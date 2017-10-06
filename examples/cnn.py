import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.exceptions import NotFittedError

import tensorflow as tf

from muffnn.core import TFPicklingBase, affine


class ConvNetClassifier(TFPicklingBase, BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible convolutional neural network classifier.

    Parameters
    ----------
    conv_hidden_units : list of tuples, optional
        Indicates the size and number of the convolutional filters.
    max_pool_size : int
        Size of max polling regions applied after each convolutional
        layer.
    dense_hidden_units : tuple or list, optional
        A list of integers indicating the number of hidden layers and their
        sizes for the dense layers at the end of the network.
    batch_size : int, optional
        The batch size for learning and prediction. If there are fewer
        examples than the batch size during fitting, then the the number of
        examples will be used instead.
    n_epochs : int, optional
        The number of epochs (iterations through the training data) when
        fitting.
    keep_prob : float, optional
        The probability of keeping values in dropout. A value of 1.0 means that
        dropout will not be used. Only used on the dense layers.
    activation : callable, optional
        The activation function. Any elementwise TensorFlow function is
        allowed.
    random_state : int, RandomState instance or None, optional
        If int, the random number generator seed. If RandomState instance,
        the random number generator itself. If None, then `np.random` will be
        used.
    solver : a subclass of `tf.train.Optimizer`, optional
        The solver to use to minimize the loss.
    solver_kwargs : optional
        Additional keyword arguments to pass to `solver` upon construction.
        See the TensorFlow documentation for possible options. Typically,
        one would want to set the `learning_rate`.

    Attributes
    ----------
    input_size_ : int
        The dimensionality of the input (i.e., number of features).
    graph_ : tensorflow.Graph
        The TensorFlow graph for the model
    n_classes_ : int
        The total number of classes.
    classes_ : array-like
        The class labels.
    """

    def __init__(self, conv_hidden_units=((5, 32), (5, 64)), max_pool_size=2,
                 dense_hidden_units=(1024,), batch_size=64, n_epochs=1,
                 keep_prob=1.0, activation=tf.nn.relu,
                 random_state=None, solver=tf.train.AdamOptimizer,
                 solver_kwargs=None):
        self.conv_hidden_units = conv_hidden_units
        self.max_pool_size = max_pool_size
        self.dense_hidden_units = dense_hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.keep_prob = keep_prob
        self.activation = activation
        self.random_state = random_state
        self.solver = solver
        self.solver_kwargs = solver_kwargs

    def __getstate__(self):
        # Handles TF persistence
        state = super().__getstate__()

        # Add attributes of this estimator
        state.update(dict(
            conv_hidden_units=self.conv_hidden_units,
            max_pool_size=self.max_pool_size,
            dense_hidden_units=self.dense_hidden_units,
            activation=self.activation,
            batch_size=self.batch_size,
            keep_prob=self.keep_prob,
            random_state=self.random_state,
            n_epochs=self.n_epochs,
            solver=self.solver,
            solver_kwargs=self.solver_kwargs))

        # Add fitted attributes if the model has been fitted.
        if self._is_fitted:
            state['input_size_'] = self.input_size_
            state['_random_state'] = self._random_state
            state['classes_'] = self.classes_
            state['n_classes_'] = self.n_classes_
            state['_label_encoder'] = self._label_encoder
            state['_image_size'] = self._image_size
            state['_num_channels'] = self._num_channels

        return state

    def _set_up_graph(self):
        """Initialize TF objects (needed before fitting or restoring)."""

        # Inputs
        self._keep_prob = tf.placeholder(
            dtype=np.float32, shape=(), name="keep_prob")
        self._input_targets = tf.placeholder(np.int32,
                                             [None],
                                             "input_targets")
        self._input_values = tf.placeholder(
            np.float32, [None, self.input_size_], "input_values")
        t = tf.reshape(
            self._input_values,
            [-1, self._image_size, self._image_size, self._num_channels])

        # Conv. layers
        prev_feats = self._num_channels
        for i, (cdim, num_feats) in enumerate(self.conv_hidden_units):

            with tf.variable_scope('conv_layer_%d' % i):
                W = tf.get_variable(
                    "weights", [cdim, cdim, prev_feats, num_feats])
                b = tf.get_variable(
                    "bias",
                    [num_feats], initializer=tf.constant_initializer(0.0))

            t = tf.nn.conv2d(t, W, strides=[1, 1, 1, 1], padding='SAME') + b
            t = t if self.activation is None else self.activation(t)
            t = tf.nn.max_pool(
                t,
                ksize=[1, self.max_pool_size, self.max_pool_size, 1],
                strides=[1, self.max_pool_size, self.max_pool_size, 1],
                padding='SAME')

            prev_feats = num_feats

        # Flatten to final size.
        final_img_size = (
            self._image_size //
            (self.max_pool_size ** len(self.conv_hidden_units)))
        t = tf.reshape(t, [-1, final_img_size * final_img_size * num_feats])

        # Dense layers.
        for i, layer_sz in enumerate(self.dense_hidden_units):
            if self.keep_prob != 1.0:
                t = tf.nn.dropout(t, keep_prob=self._keep_prob)
            t = affine(t, layer_sz, scope='dense_layer_%d' % i)
            t = t if self.activation is None else self.activation(t)

        # Final layer.
        if self.keep_prob != 1.0:
            t = tf.nn.dropout(t, keep_prob=self._keep_prob)
        t = affine(t, self.n_classes_, scope='output_layer')

        # Probs for each class.
        self._output_layer = tf.nn.softmax(t)

        # Objective function.
        self._obj_func = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._input_targets, logits=t))

        # Training.
        sk = self.solver_kwargs if self.solver_kwargs is not None else {}
        self._train_step = self.solver(**sk).minimize(self._obj_func)

    def _make_feed_dict(self, X, y=None):
        feed_dict = {
            self._input_values: X}

        if y is None:
            feed_dict[self._keep_prob] = 1.0
        else:
            feed_dict[self._input_targets] = y
            feed_dict[self._keep_prob] = self.keep_prob

        return feed_dict

    def fit(self, X, y, monitor=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples, n_targets)
            Target values
        monitor : callable, optional
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator, and a dictionary with
            {'loss': loss_value} representing the loss calculated by the
            objective function at this iteration.
            If the callable returns True the fitting procedure is stopped.
            The monitor can be used for various things such as computing
            held-out estimates, early stopping, model introspection,
            and snapshotting.

        Returns
        -------
        self : returns an instance of self.
        """
        self._is_fitted = False
        return self.partial_fit(X, y, monitor=monitor)

    def partial_fit(self, X, y, monitor=None, classes=None, **kwargs):
        """Fit the model on a minibatch of training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples, n_targets)
            Target values
        monitor : callable, optional
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator, and a dictionary with
            {'loss': loss_value} representing the loss calculated by the
            objective function at this iteration.
            If the callable returns True the fitting procedure is stopped.
            The monitor can be used for various things such as computing
            held-out estimates, early stopping, model introspection,
            and snapshotting.
        classes : array-like
            Array of labels to use for the label encoder.

        Returns
        -------
        self : returns an instance of self.
        """
        assert self.batch_size > 0, "batch_size <= 0"

        # Initialize the model if it hasn't been already by a previous call.
        if not self._is_fitted:
            self._random_state = check_random_state(self.random_state)

            self.input_size_ = int(np.prod(X.shape[1:]))
            if X.ndim == 2:
                self._image_size = int(np.sqrt(self.input_size_))
                self._num_channels = 1
            elif X.ndim == 3:
                if X.shape[1] == X.shape[2]:
                    self._image_size = X.shape[1]
                    self._num_channels = 1
                else:
                    self._image_size = int(np.sqrt(X.shape[1]))
                    self._num_channels = X.shape[2]
            elif X.ndim == 4:
                self._image_size = X.shape[1]
                self._num_channels = X.shape[3]

            if classes is not None:
                self._label_encoder = LabelEncoder()
                self._label_encoder.classes_ = np.sort(classes)
            else:
                self._label_encoder = LabelEncoder().fit(y)
            self.classes_ = self._label_encoder.classes_
            self.n_classes_ = len(self.classes_)

            self.graph_ = tf.Graph()
            with self.graph_.as_default():
                tf.set_random_seed(self._random_state.randint(0, 10000000))
                tf.get_variable_scope().set_initializer(
                    tf.contrib.layers.xavier_initializer())
                self._build_tf_graph()
                self._session.run(tf.global_variables_initializer())

            self._is_fitted = True

        yenc = self._label_encoder.transform(y)
        Xrs = X.reshape((X.shape[0], -1))

        # Train the model with the given data.
        with self.graph_.as_default():
            n_examples = X.shape[0]
            indices = np.arange(n_examples)

            for epoch in range(self.n_epochs):
                self._random_state.shuffle(indices)

                for start_idx in range(0, n_examples, self.batch_size):
                    batch_ind = indices[
                        start_idx:min(start_idx + self.batch_size, n_examples)]
                    feed_dict = self._make_feed_dict(
                        Xrs[batch_ind], yenc[batch_ind])
                    obj_val, _ = self._session.run(
                        [self._obj_func, self._train_step],
                        feed_dict=feed_dict)

                if monitor:
                    stop_early = monitor(epoch, self, {'loss': obj_val})
                    if stop_early:
                        print("stopping early due to monitor function!")
                        return self

        return self

    def predict_proba(self, X):
        """Predict probabilities for each class.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Examples to make predictions about.

        Returns
        -------
        C : array, shape = (n_samples, output_size_)
            Predicted probabilities for each class.
        """

        if not self._is_fitted:
            raise NotFittedError("Call fit before prediction!")

        Xrs = X.reshape((X.shape[0], -1))

        # Make predictions in batches.
        pred_batches = []
        start_idx = 0
        n_examples = X.shape[0]
        with self.graph_.as_default():
            while start_idx < n_examples:
                X_batch = Xrs[
                    start_idx:min(start_idx + self.batch_size, n_examples)]
                feed_dict = self._make_feed_dict(X_batch)
                start_idx += self.batch_size
                pred_batches.append(self._session.run(
                    self._output_layer, feed_dict=feed_dict))
        y_pred = np.concatenate(pred_batches)

        if len(y_pred.shape) == 1:
            y_pred = np.column_stack((1.0 - y_pred, y_pred))

        return y_pred

    def predict(self, X):
        """Make predictions.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Examples to make predictions about.

        Returns
        -------
        C : array, shape = (n_samples, output_size_)
            Predicted classes for each label.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

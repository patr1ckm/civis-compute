# You can run this example via
#
#    $ civis-compute submit iris.py
#    $ <JOBID>
#    $ civis-compute status
#    $ civis-compute get <JOBID>
#

import os
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Civis Platform container configuration.
#CIVIS name=my iris example
#CIVIS required_resources={'cpu': 1024, 'memory': 8192, 'disk_space': 10.0}

# Load and shuffle data.
iris = load_iris()
X = iris.data
y = iris.target

# Shuffle the data.
idx = np.arange(X.shape[0])
np.random.seed(45687)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# Fit and score.
rf = RandomForestClassifier(n_estimators=10)
clf = rf.fit(X, y)
score = clf.score(X, y)
print("score:", score)

# Now lets save the results.
# Just write the data to the location given by the environment
# variable CIVIS_JOB_DATA
with open(os.path.expandvars(
        os.path.join('${CIVIS_JOB_DATA}', 'iris.pkl')), 'wb') as fp:
    pickle.dump(rf, fp)

# This data will get tar gziped, put in the files endpoint and then attached to
# the job state. You can get it by running civis-compute get {scriptid}.

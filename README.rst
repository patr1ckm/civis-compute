civis-compute
=============

.. image:: https://www.travis-ci.org/civisanalytics/civis-compute.svg?branch=master
    :target: https://www.travis-ci.org/civisanalytics/civis-compute

Batch computing in the cloud with Civis Platform.

.. contents:: :local:

Installation
------------

Install from pip like this::

    pip install civis-compute


Quick Start Example
-------------------

Suppose we have a Python script that fits a Random Forest to the Iris dataset and pickles the estimator::

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

This script fits a Random Forest to the Iris dataset and pickles the estimator.

You can submit this script to Civis Platform via ``civis-compute submit``::

    civis-compute submit iris.py
    # STDOUT: script_id

The ``civis-compute submit`` command prints out the ID of the created script/container
to ``STDOUT``.

The log and the output data are attached to the script in the outputs field,
visible in Civis Platform under ``Run History``.

You can also download the outputs via ``civis-compute get``::

    civis-compute get SCRIPTID
    # default path is the current working directory
    # STDOUT: /path/to/archive.tar.gz

You can unpack the gzipped tar archive with::

    tar -xzvf /path/to/archive.tar.gz

Finally, you can recover the estimator like this::

    with open('/path/to/archive/iris.pkl', 'rb') as fp:
        rf = pickle.load(fp)

Note that any data written to the directory ``${CIVIS_JOB_DATA}`` in the job will be put into a
gzipped tar archive, put on the files endpoint and attached as an output to the script. This
behavior means that you can write any type of file, including CSV, pickled python objects, plots,
images, etc., and possibly more than one file to this directory and easily pull the results back
to your local machine via ``civis-compute get``.

Bring Your Own Container
------------------------

To use the  CLI, you must have the Civis Python API client pre-installed in the container.
You can get it via ``pip install civis`` or from https://github.com/civisanalytics/civis-python.

Support for Jupyter Notebooks
-----------------------------

The CLI can execute jupyter notebooks on Civis Platform. Locally, your notebook is converted to a
python script and then executed via ``ipython`` in a container script. This allows you to use and execute
ipython magics (e.g., ``%timeit``, etc.) in your notebooks. IPython magics that are jupyter specific
(i.e., ``%matplotlib inline`` and ``%matplotlib notebook``) are replaced with ``pass`` before
executing the notebook.

Support for R
-------------

We have installed the Python API client into our ``datascience-r`` container. This container
can be used to execute R scripts.

Use ``snake_case``, not ``CamelCase`` for Input Parameters
----------------------------------------------------------

All input parameters in comments (like ``#CIVIS required_resources=...`` above)
and the CLI are in ``snake_case``. This includes parameters not at the top level
(e.g., the ``disk_space`` option for ``required_resources``).

For the command line, ``required_resources`` is written as ``required-resources`` in keeping with
\*nix conventions.

Use YAML to Specify API Parameters That Require Lists or Hash Maps
------------------------------------------------------------------

For example, in a comment in a script use::

    #CIVIS required_resources={'cpu': 1024}

or on the command line use::

    civis-compute submit --required-resources="{'cpu': 1024}" <script.py>

for the ``required_resources`` hash map.

Available CLI Utilities
-----------------------

``civis-compute submit``
----------------------

To submit a local bash, python script, R script or jupyter notebook to Civis Platform, you can simply type::

    civis-compute submit SCRIPT [ARGS]

This command uploads the script to Civis Platform using the files endpoint and then executes it in a
container using a default setup (which gives you 1024 CPUs, 8192 MB of RAM, 16 GB of disk space, and
uses the latest version of the ``datascience-python`` or the ``datascience-r`` docker image). You
can pass arguments to the script and they will be reproduced on Civis Platform. Any arguments which
are files are automatically uploaded to the files endpoint.

Note that you can also execute bash on Civis Platform directly by simply putting the commands right after
``civis-compute submit``. For example::

    civis-compute submit sleep 3600

would make a container script execute ``sleep 3600``.

If you want to adjust these defaults or set any other parameters that can be set via the API,
you can simply add comments to your script that look like this::

    #CIVIS name=iris

This command would set the name of the custom script to 'iris'. Parameters can also be set from
the command line as options to ``civis-compute submit``. See the rest of the parameters that can be set here
https://platform.civisanalytics.com/api#v1_post_scripts_containers.

Note that special keys can be added to these comments or the command line for civis-compute CLI specific behavior

- **Run a Shell Command Before the Script**

  You can run a shell command via::

      #CIVIS shell_cmd=pip install -q tqdm

  This shell command will execute after all data has been uploaded to the container
  script but before any python packages are installed.

- **Upload Additional Files**

  To upload additional files, put them in a comment like this::

      #CIVIS files=data.csv,module.py

  These files will be put in the container job at the same relative path they are to the
  script that is uploaded.

- **Caching File Uploads**

  The civis-compute CLI can maintain a local cache of MD5 checksums and file IDs on the Civis files
  endpoint. When you specify a file dependency, this local cache is checked first. If a file
  will not expire for at least two weeks and has the same checksum, then the already uploaded
  file is used. To turn on caching, you can specify a comment like this::

      #CIVIS use_file_cache=True

- **Custom Repo Installs**

  If you specify a Git repo via the ``repo_http_uri`` option, then the ``repo_cmd`` option
  will determine how the repo is handled. By default, it is set to ``python setup.py install``.
  You can change this via::

      #CIVIS repo_cmd=python setup.py develop

- **Adding AWS Credentials**

  You can pass AWS credentials (which are stored on Civis Platform) into your job by default using::

      #CIVIS add_aws_creds=True

  You can specify your AWS credential ID from Civis Platform like this::

      #CIVIS aws_cred_id=ID

  If you do not give a credential ID, the first one found in your list of AWS credentials in
  Civis Platform is used.

Finally, any thing that can be set in the comments can be passed as a command line argument to
``civis-compute submit``. Command line arguments override anything set in the script via the
comments.

You can do a dry run of a script via the command line via::

    civis-compute submit --dry-run

This command prints out the container config and command to be run. This feature can be used
to help debug scripts before they run on Civis Platform.

``civis-compute get``
-------------------

To get the outputs of a script which has finished::

    civis-compute get SCRIPTID

where ``SCRIPTID`` is the ID of the Civis Platform script, printed to STDOUT by ``civis-compute submit``.
This command will pull the outputs from the latest run. You can specify a specific run with the
``--run-id=RUNID`` option.

To change the output directory::

    civis-compute get SCRIPTID path/to/output

To specify a specific run::

    civis-compute get SCRIPTID --run-id=RUNID

``civis-compute status``
----------------------

To view scripts that are running (and you have permissions to view)::

    civis-compute status

To see just your scripts::

    civis-compute status --mine

To see info about the most recent run of a specific container::

    civis-compute status SCRIPTID

where ``SCRIPTID`` is the ID of the Civis Platform script, printed to STDOUT
by ``civis-compute submit``.

Note that only container scripts are listed by ``civis-compute status``, up
to ~50 scripts.

``civis-compute cancel``
----------------------

To cancel a script running on Civis Platform::

    civis-compute cancel SCRIPTID

where ``SCRIPTID`` is the ID of the Civis Platform script, printed to STDOUT
by ``civis-compute submit``.

Note that only containers which you are running (i.e., ``running_as`` is set you) can be canceled. This
command will cancel both hidden and non-hidden scripts.

``civis-compute cache``
---------------------

The civis-compute CLI can cache the MD5 checksums and files endpoint IDs of your files to avoid uploading
them more than once.

To see the files in your local cache::

    civis-compute cache list

To clear the local cache::

    civis-compute cache clear

The actual cache is a simple sqlite database stored at ``~/.civiscompute/fileidcache.db``.

To turn on this feature, either set ``use_file_cache: True`` in your ``~/.civiscompute/config.yml``, or pass
this argument to your script via the command line or a configuration comment.


Changing the Default Script Submission Parameters
-------------------------------------------------

You can change the default script submission parameters and turn on the file
cache by default by editing your ``~/.civiscompute/config.yml`` file.

Here is an example::

    # my civis-compute CLI config
    use_file_cache: False
    required_resources:
      cpu: 256
      memory: 1024
      disk_space: 1.0
    docker_image_name:
      python: civisanalytics/datascience-python
      r: civisanalytics/datascience-r
    repo_cmd:
      python: 'python setup.py install'
    add_aws_creds: False
    # put a default AWS credential ID here
    # aws_cred_id:

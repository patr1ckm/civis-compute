import os
from setuptools import find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as _in:
        return _in.read()


__version__ = None
exec(read('civiscompute/version.py'))

setup(
    name="civis-compute",
    version=__version__,
    author="Civis Analytics Inc",
    author_email="opensource@civisanalytics.com",
    url="https://www.civisanalytics.com",
    description="Batch computing in the cloud with Civis Platform.",
    packages=find_packages(),
    long_description=read('README.rst'),
    include_package_data=True,
    license="BSD-3",
    install_requires=read('requirements.txt').strip().split('\n'),
    entry_points={
        'console_scripts': ['civis-compute = civiscompute.cli:cli']})

"""minimal package setup"""
import os
from setuptools import setup, find_packages

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="magpysv",
    version="2.0",
    author="Grace Cox",
    author_email="grace.alex.cox@gmail.com",
    license="MIT",
    url="https://github.com/gracecox/MagPySV",
    description="Download, process and denoise geomagnetic observatory data",
    long_description=read('readme.rst'),
    packages=find_packages(),
    include_package_data=True,
    package_data={"magpysv": ["baseline_records"]},
    classifiers=["Programming Language :: Python :: 3"],
    zip_safe=False,
    install_requires=['aacgmv2>=2.5.2','cartopy==0.17.0','chaosmagpy','datetime','glob',
   'jupyter>=1.0.0','matplotlib>=2.0.0','notebook>=4.3.1',
    'numpy>=1.12.0','pandas>=0.19.2','requests>=2.12.4','scikit-learn<=0.21.3',
    'scipy>=0.18.1','gmdata_webinterface'],
    extras_require={'develop': ['jupyter>=1.0.0','matplotlib>=2.0.0','notebook>=4.3.1',
    'numpy>=1.12.0','pandas>=0.19.2','requests>=2.12.4','scikit-learn<=0.21.3',
    'scipy>=0.18.1','prospector>=0.12.7','pytest>=3.0.6',
    'pytest-cov<2.6','Sphinx>=1.5.1','sphinx-rtd-theme>=0.1.9','gmdata_webinterface']}
)

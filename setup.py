"""minimal package setup"""
import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="magpysv",
    version="0.1",
    author="Grace Cox",
    license="MIT",
    description="Download, process and denoise geomagnetic observatory data",
    long_description=read('readme.rst'),
    packages=find_packages(),
    classifiers=["Programming Language :: Python :: 3"],
    zip_safe=False,
    install_requires=['jupyter>=1.0.0','matplotlib>=2.0.0','notebook>=4.3.1',
    'numpy>=1.12.0','pandas>=0.19.2','requests>=2.12.4','scikit-learn>=0.18.1',
    'scipy>=0.18.1','geomag-webapp-interface'],
    extras_require={'develop': ['jupyter>=1.0.0','matplotlib>=2.0.0','notebook>=4.3.1',
    'numpy>=1.12.0','pandas>=0.19.2','requests>=2.12.4','scikit-learn>=0.18.1',
    'scipy>=0.18.1','prospector>=0.12.7','pytest>=3.0.6',
    'pytest-cov>=2.4.0','Sphinx>=1.5.1','sphinx-rtd-theme>=0.1.9','geomag-webapp-interface']},
    dependency_links=['git+https://github.com/willjbrown88/geomag_wdc_web_app_interface.git@master#egg=geomag_wdc_web_app_interface']
)

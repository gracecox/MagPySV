Welcome to MagPySV!
===================================
|build-status| |docs-status| |code-health| |coverage| |license|

Full documentation is hosted at http://magpysv.readthedocs.io/en/latest/

Introduction
------------

MagPySV is an open-source Python package designed to provide a consistent, and automated as far as possible, means of generating high resolution SV time series from raw observatory hourly means distributed by the World Data Centre (WDC) for Geomagnetism at the British Geological Survey (BGS), Edinburgh. It uses a second Python package developed by BGS to download data for a given time range and list of observatories.

The package may be run on the command line or within an interactive Jupyter notebook, and allows the user to obtain data in WDC format from BGS servers for a user-specified time period and list of observatories. It produces time series of the X, Y and Z components of the internally-generated magnetic field and secular variation (SV) at the desired frequency (typically monthly or annual means), and applies corrections for all documented baseline changes. Optionally, the user may exclude data using the `ap index`_, which removes effects from documented high solar activity periods such as geomagnetic storms. Robust statistics are used to identify and remove outliers. 

The software develops previously published denoising methods, which aim to remove external field contamination from the internal field, using principal component analysis, a method that uses the covariance matrix of the residual between the observed SV and that predicted by a global field model to create and remove a proxy for external field signal from the data. This method, based on `Wardinski & Holme (2011)`_, creates a single covariance matrix for all observatories of interest combined and applies the external field correction to all locations simultaneously, resulting in cleaner time series of the internally-generated SV.

Installation
------------

MagPySV can be installed via the Python Package Index (PyPI) using the command
``pip install magpysv``. This also installs all required dependencies, including the BGS data downloading app. The example Juypyter notebooks (see below) are not included in the PyPI installation and need to be downloaded from a separate GitHub project page after the software is installed.

Example workflows
-----------------

In the paper accompanying this software (in prep), we present two case studies of cleaned data in different geographic regions and discuss their application to geomagnetic jerks: monthly first differences for Europe, and annual differences for northern high latitude regions. A second GitHub repository accompanying this package includes two notebooks that can be used to download the relevant hourly data from BGS and reproduce the figures for these case studies. Note that MagPySV must be installed before using the notebooks (preferably using ``pip``).

To obtain the notebooks, go to their GitHub project page at https://github.com/gracecox/MagPySV-examples and click the green "clone or download" button. From the dropdown menu, select "download zip" and save the file to the desired location. Unzip the file and navigate to the resulting folder. To open the notebooks, launch a Jupyter notebook from the folder (e.g. by typing ``jupyter notebook`` into the command line from that directory) and then select the desired notebook from the list that appears in your web browser.

The `Jupyter documentation`_ contains a step-by-step tutorial on installing and running notebooks, and is aimed at new users who have no familiarity with Python.

Contributing
------------

We hope others in the geomagnetism community find this code useful and welcome suggestions, feedback and contributions. Requesting new features or reporting bugs can be done by creating a `GitHub issue`_ for the repository.

If you would like to fix bugs or implement new features yourself, this is very welcome! This is done by

1. Forking MagPySV's GitHub repository
2. Creating a branch for your changes
3. Making your changes to the code
4. Submitting a pull request to the repository

Those unfamiliar with this process might find `GitHub's tutorials`_ useful. If this still looks too complicated, you are welcome to create a GitHub issue or get in touch with us directly for help.

Reference
---------

A manuscript describing MagPySV is currently in preparation. The paper also presents two case studies of cleaned data at European and high latitude observatories, and their application to geomagnetic jerks.

.. _ap index: https://www.gfz-potsdam.de/en/kp-index/
.. _Wardinski & Holme (2011): https://doi.org/10.1111/j.1365-246X.2011.04988.x
.. _Jupyter documentation: https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/
.. _GitHub issue: https://github.com/gracecox/MagPySV/issues
.. _GitHub's tutorials: https://guides.github.com/


.. |build-status| image:: https://travis-ci.org/gracecox/MagPySV.svg?branch=master
    :target: https://travis-ci.org/gracecox/MagPySV
    :alt: Build Status

.. |docs-status| image:: https://readthedocs.org/projects/magpysv/badge/?version=latest
    :target: http://magpysv.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |coverage| image:: https://coveralls.io/repos/github/gracecox/MagPySV/badge.svg?branch=master
   :target: https://coveralls.io/github/gracecox/MagPySV?branch=master
   :alt: Coverage

.. |license| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   
.. |code-health| image:: https://api.codacy.com/project/badge/Grade/e22cd43f7e364892ab42c874ada808ad
   :alt: Codacy Health
   :target: https://app.codacy.com/app/gracecox/MagPySV?utm_source=github.com&utm_medium=referral&utm_content=gracecox/MagPySV&utm_campaign=badger

The project's main directory contains the readme file, license and various setup files. The `magpysv` directory contains all of the Python modules that make up the package, detailed descriptions of all functions contained within each module are found in the full documentation hosted at http://magpysv.readthedocs.io/en/latest/. The code documentation can also be accessed using Python's help function by typing a command in this format: ``help(magpysv.modulename.functionname)``


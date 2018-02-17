# -*- coding: utf-8 -*-
#    Copyright (C) 2016  Grace Cox (University of Liverpool)
#
#    Released under the MIT license, a copy of which is located at the root of
#    this project.
"""Module containing functions to run the COV-OBS (Gillet et al) field model.

Part of the MagPySV package for geomagnetic data analysis. Contains a function
to obtain a complete list of geomagnetic observatory locations from the WDC
webserver and another function to run the COV-OBS magnetic field model by
Gillet et al. (2013, Geochem. Geophys. Geosyst.,
https://doi.org/10.1002/ggge.20041; 2015, Earth, Planets and Space,
https://doi.org/10.1186/s40623-015-0225-z2013) to obtain model
predictions for these observatory locations. The code can be obtained from
http://www.spacecenter.dk/files/magnetic-models/COV-OBSx1/ and no modifications
are necessary to run it using MagPySV.
"""

import os
from subprocess import Popen, PIPE
import numpy as np
import requests


def get_observatory_list():
    """Obtain the complete list of observatory locations held by the WDC.

    Obtains a dictionary containing information for all geomagnetic
    observatories known to the WDC using the BGS website at:
    http://app.geomag.bgs.ac.uk/wdc/

    The following information is given for each location:

    'AAA':
         {
         'code': 'AAA',

         'country': 'Kazakhstan',

         'dataAvailability': {'hour': {'earliest': 1963, 'latest': 2015},

         'minute': {'earliest': 2005, 'latest': 2015}},

         'dateClosed': None,

         'dateOpened': [1963, 1, 1],

         'elevation': 1300.0,

         'latitude': 43.18,

         'longitude': 76.92,

         'name': 'Alma Ata'
         }

    Returns:
        stations (dict):
            dictionary containing information about each geomagnetic
            observatory.
    """

    url_base = r'http://app.geomag.bgs.ac.uk/wdc/'
    station_resource = 'stations'
    stations_url = url_base + station_resource

    response = requests.get(stations_url)
    # Dictionary with IAGA codes as primary keys
    stations = response.json()
    return stations


def run_covobs(*, stations, model_path, output_path):
    """Use observatory latitude, longitude and elevation to run COV-OBS.

    Uses the dictionary of observatory information obtained from the WDC site
    to run the COV-OBS field model Gillet et al. (2013, Geochem. Geophys.
    Geosyst.,
    https://doi.org/10.1002/ggge.20041; 2015, Earth, Planets and Space,
    https://doi.org/10.1186/s40623-015-0225-z2013) for a given location given
    in geodetic coordinates (model output is also in geodetic coordinates).
    Converts latitude in degrees to colatitude in radians, longitude in degrees
    (0 to 360) into radians (-pi to pi) and elevation in m to km. It then runs
    the fortran exectuable for the field model and passes the location data as
    command line arguments. The output files are stored as mf_obs.dat and
    sv_obs.dat for magnetic field and secular variation predictions
    respectively (e.g. mf_ngk.dat and sv_ngk.dat for Niemegk.)

    Assumes that the user has compiled the fortran source code and called the
    executable "a.out".  No modification to the fortran source code is
    required (code can be downloaded from
    http://www.spacecenter.dk/files/magnetic-models/COV-OBSx1/).

    The COV-OBS code can also be used to run other field models if modified to
    accept a different spline file as input, rather than the supplied
    COV-OBS.x1-int file.

    Args:
        stations (dict): dictionary containing information about each
                geomagnetic observatory.
        model_path (str): path to the compiled COV-OBS executable.
        output_path (str): path to the directory in which the model output
            should be stored.
    """
    mycwd = os.getcwd()
    os.chdir(model_path)
    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for ob in stations.keys():
        print(ob)
        # Convert from latitude in degrees to colatitude in radians
        colatitude = np.deg2rad(90.0 - stations.get(ob).get('latitude'))
        # Convert from longitude in degrees (0 to 360) to radians (-pi to pi)
        longitude = np.deg2rad(stations.get(ob).get('longitude') - 360.0)
        # Convert elevation from m to km
        if stations.get(ob).get('elevation') is None:
            altitude = 0.0
        else:
            altitude = stations.get(ob).get('elevation')/1000.0
        # Create a string containing the inputs to the COV-OBS field model
        model_inputs = "%s\n%s\n%s\n" % (str(altitude), str(colatitude),
                                         str(longitude))
        # Create a process so python can interact with the model executable
        p = Popen('./a.out', stdin=PIPE,
                  stdout=PIPE)
        # Pass the altitude, colatitude and longitude to the field model
        p.communicate(model_inputs.encode())
        p.wait()
        # Rename the output files so they contain the observatory name
        os.rename('mfpred.dat', os.path.join(output_path,
                  'mf_%s.dat' % ob.lower()))
        os.rename('svpred.dat', os.path.join(output_path,
                  'sv_%s.dat' % ob.lower()))
    # Return to previous working directory
    os.chdir(mycwd)

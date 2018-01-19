# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:35:54 2017
Miscellaneous functions that will be included in MagPySV but need documenting,
testing etc
@author: gracecox
"""
import pandas as pd
import numpy as np
from math import pi


def get_baseline_info(file_path):
    col_names = ['observatory', 'jump_year', 'x_jump', 'y_jump', 'z_jump']
    data = pd.read_csv(file_path, sep=',', names=col_names)
    data['jump_year'] = pd.to_datetime(dict(year=data['jump_year'], month=1, day=1))
    return data


def correct_baseline_change(*, observatory, field_data, jump_data):
    obs_jumps = jump_data.loc[jump_data['observatory'] == str.upper(observatory)]
    print(obs_jumps)
    for jump in obs_jumps.index:
        if (obs_jumps.loc[jump].x_jump == 0 & obs_jumps.loc[jump].y_jump == 0 & obs_jumps.loc[jump].z_jump == 0):
            print("Field jump of unknown magnitude: ", obs_jumps.loc[jump].jump_year)
        else:
            field_data.loc[field_data['date'] < obs_jumps.loc[jump].jump_year, 'X'] = field_data.loc[
                field_data['date'] < obs_jumps.loc[jump].jump_year, 'X'] - obs_jumps.loc[jump].x_jump
            field_data.loc[field_data['date'] < obs_jumps.loc[jump].jump_year, 'Y'] = field_data.loc[
                field_data['date'] < obs_jumps.loc[jump].jump_year, 'Y'] - obs_jumps.loc[jump].y_jump
            field_data.loc[field_data['date'] < obs_jumps.loc[jump].jump_year, 'Z'] = field_data.loc[
                field_data['date'] < obs_jumps.loc[jump].jump_year, 'Z'] - obs_jumps.loc[jump].z_jump 


def geod2geoc(geodetic_latitude, altitude, data):
    """ conversion from geodetic X,Z components to geocentric B_r, B_theta
     Input:   geodetic latitude alpha (rad)
          altitude h [km]
          X, Z
     Output:  theta (rad)
          r (km)
          B_r, B_theta

     Nils Olsen, DSRI Copenhagen, September 2001.
     After Langel (1987), eq. (52), (53), (56), (57)
    """

    # Ellipsoid after World Geodetic System of 1984 (WGS84)
    # semimajor axis in km
    a = 6378.14
    # semiminor axis in km
    #b = 6356.75231424518
    b = 6356.75
    sin_lat_sq = (np.sin(geodetic_latitude))**2
    cos_lat_sq = (np.cos(geodetic_latitude))**2

    tmp = altitude * np.sqrt((a**2 * cos_lat_sq) + (b**2 * sin_lat_sq))
    beta = np.arctan((tmp + b**2) / (tmp + a**2) * np.tan(geodetic_latitude))
    # Geocentric coordinates (r and theta in spherical coordinates)
    theta = pi / 2 - beta
    r = np.sqrt(altitude**2 + 2*tmp + a**2*(1 - (1 - (b/a)**4)*sin_lat_sq) / (1 - (1 - (b/a)**2)*sin_lat_sq))
    psi = np.sin(geodetic_latitude) * np.sin(theta) - np.cos(geodetic_latitude) * np.cos(theta)
    data['Br'] = -np.sin(psi) * data['X'] - np.cos(psi) * data['Z']
    data['Btheta'] = -np.cos(psi) * data['X'] + np.sin(psi) * data['Z']
    return data, r, theta

def dms2dd(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
    if direction == 'E' or direction == 'N':
        dd *= -1
    return dd

def geoc2geod(*, r, theta, data):
    """ [alpha, h]       = geoc2geod(r, theta);
    [alpha, h, X, Z] = geoc2geod(r, theta, B_r, B_theta);

    Input:   geographic co-latitude theta (rad)
             geocentric radius r (km)
             B_r, B_theta
    Output:  geodetic latitude alpha (rad)
             geodetic altitude h [km]
             X, Z
"""
    # Nils Olsen, DTU Space, June 2011

    # Ellipsoid GRS 80 (identical for WGS84)
    a = 6378.137
    b = 6356.752
    rad = pi / 180

    RTOD = 57.2957795130823
    DTOR = 0.01745329251994330

    E2 = 1. - (b / a)**2
    E4 = E2 * E2
    E6 = E4 * E2
    E8 = E4 * E4
    OME2REQ = (1. - E2) * a
    A21 =     (512. * E2 + 128. * E4 + 60. * E6 + 35. * E8) / 1024.
    A22 =     (                        E6 +     E8) /  32.
    A23 = -3. * (                     4.*E6 +  3. * E8) / 256.
    A41 =    -(           64. * E4 + 48. * E6 + 35. * E8) / 1024.
    A42 =     (            4. * E4 +  2. * E6 +     E8) /  16.
    A43 =                                   15. * E8 / 256.
    A44 =                                      -E8 /  16.
    A61 =  3. * (                     4. * E6 +  5. * E8) / 1024.
    A62 = -3. * (                        E6 +     E8) /  32.
    A63 = 35. * (                     4.* E6 +  3. * E8) / 768.
    A81 =                                   -5. * E8 / 2048.
    A82 =                                   64. * E8 / 2048.
    A83 =                                 -252. * E8 / 2048.
    A84 =                                  320. * E8 / 2048.

    GCLAT = 90 - theta / rad
    SCL = np.sin(GCLAT * DTOR)

    RI = a / r
    A2 = RI * (A21 + RI * (A22 + RI * A23))
    A4 = RI * (A41 + RI * (A42 + RI * (A43 + RI * A44)))
    A6 = RI * (A61 + RI * (A62 + RI * A63))
    A8 = RI * (A81 + RI * (A82 + RI * (A83 + RI * A84)))

    CCL = np.sqrt(1 - SCL**2)
    S2CL = 2. * SCL * CCL
    C2CL = 2. * CCL * CCL - 1.
    S4CL = 2. * S2CL * C2CL
    C4CL = 2. * C2CL * C2CL - 1.
    S8CL = 2. * S4CL * C4CL
    S6CL = S2CL * C4CL + C2CL * S4CL

    DLTCL = S2CL * A2 + S4CL * A4 + S6CL * A6 + S8CL * A8
    alpha = (DLTCL * RTOD + GCLAT) * rad
    h = r * np.cos(DLTCL) - a * np.sqrt(1 - E2 * np.sin(alpha)**2)

    # convert also magnetic components
    psi = np.sin(alpha) * np.sin(theta) - np.cos(alpha) * np.cos(theta)
    if set(['Br', 'Btheta']).issubset(data.columns):
        psi = np.sin(alpha) * np.sin(theta) - np.cos(alpha) * np.cos(theta)
        data['X']  = -np.cos(psi) * data['Btheta'] - np.sin(psi) * data['Br'] # X
        data['Y']  = +np.sin(psi) * data['Btheta'] - np.cos(psi) * data['Br'] # Z
    return alpha, h, data

"""todays_date = datetime.datetime.now().date()
index = pd.date_range(todays_date-datetime.timedelta(3), periods=3, freq='D')
#test = np.array([[706, -4030, 55830], [695, -4028, 55850], [695, -4030, 55866]])
test = np.array([[-43102.848351, -18287.577671, -1740], [-43102.838570, -18284.577686, -1733], [-43107.858104, -18290.561354, -1730]])
columns = ['Br','Btheta', 'Bphi']
df1 = pd.DataFrame(index=index, columns=columns, data=test)
r = 6364.9525701447783
theta = 0.66526349488693726 
print(df1)
geodetic_latitude, geodetic_altitude, df1 = geoc2geod(data=df1, theta=0.66526349488693726, r=r)
print(df1)
print(np.rad2deg(geodetic_latitude))
print(geodetic_altitude)"""
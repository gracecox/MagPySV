# -*- coding: utf-8 -*-
"""
Part of the MagPy package for geomagnetic data analysis. This module provides
various functions to read, parse and manipulate the contents of World Data
Centre (WDC) formatted files containing geomagnetic data.

    Copyright (C) 2016  Grace Cox

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
# Need functions to:
# 1. Remove stormy days, save to file
# 2. Get monthly or annual differences of the monthly means (SV)

import datetime as dt
import pandas as pd
import numpy as np
import wdc_io

obs_names = ['aqu']   # ['aqu','clf','hrb','ngk']
model_name = 'cov-obs'


def wdc_readfile(fname):
    """ Load a WDC datafile and place the contents into a dataframe.

    Load a datafile of WDC hourly geomagnetic data for a single observatory and
    extract the contents. Can parse both the current WDC file format and the
    previous format containing international quiet (Q) or disturbed (D) day
    designation in place of the century field.

    Args:
        fname (str): path to a WDC datafile.

    Returns:
        data (dataframe): dataframe containing averaged geomagnetic data. First
        column is a series of datetime objects (in the format yyyy-mm-dd)
        and subsequent columns are the X, Y and Z components of the
        magnetic field at the specified times.
    """

    try:
        # New WDC file format
        cols = [(0, 3), (3, 5), (5, 7), (7, 8), (8, 10), (14, 16),
                (16, 20), (116, 120)]
        col_names = [
            'code', 'yr', 'month', 'component', 'day', 'century',
            'base', 'daily_mean_temp']
        types = {
            'code': str, 'year': int, 'month': int, 'component': str,
            'day': int, 'century': int, 'base': int, 'daily_mean': float}
        data = pd.read_fwf(fname, colspecs=cols, names=col_names,
                           converters=types, header=None)
    except ValueError:
        # Old WDC format (century value is missing. Col 15 = International D
        # or Q days.)
        cols = [(0, 3), (3, 5), (5, 7), (7, 8), (8, 10), (16, 20), (116, 120)]
        col_names = [
            'code', 'yr', 'month', 'component', 'day', 'base',
            'daily_mean_temp']
        types = {
            'code': str, 'year': int, 'month': int, 'component': str,
            'day': int, 'base': int, 'daily_mean': float}
        data = pd.read_fwf(fname, colspecs=cols, names=col_names,
                           converters=types, header=None)
        data['century'] = 19

    return data


def wdc_datetimes(data):
    """ Create datetime objects from the fields extracted from a WDC datafile.

    Args:
        data (dataframe): needs columns for century, year (yy format), month
            and day. Called by wdc_readfile.

    Returns:
        data (dataframe): the same dataframe with a series of datetime objects
        (in the format yyyy-mm-dd) in the first column.
    """

    # Convert the century/yr columns to a year
    data['year'] = 100*data['century'] + data['yr']

    # Create datetime objects from the century, year, month and day columns of
    # the WDC format data file
    dates = data.apply(lambda x: dt.datetime.strptime(
        "{0} {1} {2}".format(x['year'], x['month'], x['day']),
        "%Y %m %d"), axis=1)
    data.insert(0, 'date', dates)
    data.drop(['year', 'yr', 'century', 'code', 'day', 'month'], axis=1,
              inplace=True)

    return data


def wdc_xyz(data):
    """ Convert extracted WDC data to daily averages of X, Y and Z components.

    Missing values (indicated by 9999 in the datafiles) are replaced with NaNs.

    Args:
        data (dataframe): dataframe containing columns for datetime objects,
            magnetic field component (D, I, F, H, X, Y or Z), the tabular base
            and daily mean.

    Returns:
        data (dataframe): the same dataframe with datetime objects in the first
        column and columns for X, Y and Z components of magnetic field (in nT).
    """

    # Replace missing values with NaNs
    data.replace(9999, np.nan, inplace=True)

    data = data.groupby('component').apply(wdc_io.daily_mean_conversion)
    data.reset_index(drop=True, inplace=True)
    data.drop(['base', 'daily_mean_temp'], axis=1, inplace=True)
    data = data.pivot(index='date', columns='component', values='daily_mean')
    data.reset_index(inplace=True)

    # Call helper function to convert D and H components to X and Y
    if 'D' in data.columns and 'H' in data.columns:
        data = wdc_io.angles_to_geographic(data)
        data = data[['date', 'X', 'Y', 'Z']]

    else:
        data = data[['date', 'X', 'Y', 'Z']]

    return data


def daily_mean_conversion(df):
    """ Use the tabular base to calculate daily means in nT or degrees (D, I)

    Uses the tabular base and daily value from the WDC file to calculate the
    daily means of magnetic field components. Value is in nT for H, F, X, Y or
    Z components and in degrees for D or I components. Called by wdc_xyz.

    daily_mean = tabular_base*100 + wdc_daily_value (for components in nT)

    daily_mean = tabular_base + wdc_daily_value/600 (for D and I components)

    Args:
        df (dataframe): dataframe containing columns for datetimeobjects,
            magnetic field component (D, I, F, H, X, Y or Z), the tabular base
            and daily mean.

    Returns:
        df (dataframe): the same dataframe with datetime objects in the first
        column and daily means of the field components in either nT or degrees
        (depending on the component).
    """

    grp = pd.DataFrame()
    for group in df.groupby('component'):

        if group[0] == 'D' or group[0] == 'I':
            group[1]['daily_mean'] = group[1]['base'] + \
                (1/600.0)*group[1]['daily_mean_temp']
            grp = grp.append(group[1], ignore_index=True)
        else:
            group[1]['daily_mean'] = 100.0*group[1]['base'] + \
                group[1]['daily_mean_temp']
            grp = grp.append(group[1], ignore_index=True)
    return grp


def angles_to_geographic(data):
    """ Use D and H values to calculate the X and Y field components.

    The declination (D) and horizontal intensity (H) relate to the north (Y)
    and east (X) components as follows:

    X = H*cos(D)

    Y = H*sin(D)

    Args:
        data (dataframe): dataframe containing columns for datetimeobjects and
            daily means of the magnetic field components (D, I, F, H, X, Y or
            Z).

    Returns:
        data (dataframe): the same dataframe with datetime objects in the first
        column and daily means of the field components in either nT or degrees
        (depending on the component).
    """

    data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'X'] = data.loc[(
        ~np.isnan(data['D']) & ~np.isnan(data['H'])), 'H']*np.cos(np.deg2rad(
            data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'D']))

    data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'Y'] = data.loc[(
        ~np.isnan(data['D']) & ~np.isnan(data['H'])), 'H']*np.sin(np.deg2rad(
            data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'D']))

    return data


def data_resampling(df, sampling='M'):
    """ Resample the daily geomagnetic data to a specified frequency.

        Args:
            df (dataframe): dataframe containing datetime objects and daily
                means of magnetic data.
            sampling (str): new sampling frequency. Default value is 'M'
                (monthly means), which averages data for each month and sets
                the datetime object to the final day of that month. Use 'MS'
                to set the datetime object to the first day of the month.
                Another useful option is 'A' (annual means), which averages
                data for a whole year and sets the datetime object to the final
                day of the year. Use 'AS' to set the datetime object to the
                first day of the year.

        Returns:
            df (dataframe): dataframe of datetime objects and monthly/annual
            means of observatory data.
    """

    resampled = df.set_index('date', drop=False).resample(sampling, how='mean')
    resampled.reset_index(inplace=True)

    return resampled

# =============================================================================
# df = pd.DataFrame()
# for observatory in obs_names:
#     path = '/Users/Grace/Dropbox/BGS_hourly/hourval/single_obs/%s/*.wdc' \
#         % observatory
#     print(path)
#     filenames = glob.glob(path)
#     for f in filenames:
#         try:
#             frame = wdc_io.wdc_to_dataframe(f)
#             df = df.append(frame, ignore_index=True)
#         except StopIteration:
#             pass
# =============================================================================

# -*- coding: utf-8 -*-
"""Module containing functions to parse World Data Centre (WDC) files

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
    with this program.  If not, see <http://www.gnu.org/licenses/>."""
# Need functions to:
# 1. Remove stormy days, save to file
# 2. Get monthly or annual differences of the monthly means (SV)

import datetime as dt
import pandas as pd
import numpy as np
import glob


def wdc_parsefile(fname):
    """Load a WDC datafile and place the contents into a dataframe.

    Load a datafile of WDC hourly geomagnetic data for a single observatory and
    extract the contents. Can parse both the current WDC file format and the
    previous format containing international quiet (Q) or disturbed (D) day
    designation in place of the century field.

    Args:
        fname (str): path to a WDC datafile.

    Returns:
        data (pandas.DataFrame): dataframe containing daily geomagnetic
            data. First column is a series of datetime objects (in the format
            yyyy-mm-dd) and subsequent columns are the X, Y and Z components of
            the magnetic field at the specified times."""

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
        # or Q days. Col 16 = Blank for data since 1900, 8 for data before.)
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
    """Create datetime objects from the fields extracted from a WDC datafile.

    Args:
        data (pandas.DataFrame): needs columns for century, year (yy format),
            month and day. Called by wdc_parsefile.

    Returns:
        data (pandas.DataFrame): the same dataframe with a series of datetime
            objects (in the format yyyy-mm-dd) in the first column."""

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
    """Convert extracted WDC data to daily averages of X, Y and Z components.

    Missing values (indicated by 9999 in the datafiles) are replaced with NaNs.

    Args:
        data (pandas.DataFrame): dataframe containing columns for datetime
            objects, magnetic field component (D, I, F, H, X, Y or Z), the
            tabular base and daily mean.

    Returns:
        data (pandas.DataFrame): the same dataframe with datetime objects in
            the first column and columns for X, Y and Z components of magnetic
            field (in nT)."""

    # Replace missing values with NaNs
    data.replace(9999, np.nan, inplace=True)

    data = data.groupby('component').apply(daily_mean_conversion)
    data.reset_index(drop=True, inplace=True)
    data.drop(['base', 'daily_mean_temp'], axis=1, inplace=True)
    data = data.pivot(index='date', columns='component', values='daily_mean')
    data.reset_index(inplace=True)

    # Call helper function to convert D and H components to X and Y
    if 'D' in data.columns and 'H' in data.columns:
        data = angles_to_geographic(data)
        data = data[['date', 'X', 'Y', 'Z']]

    else:
        data = data[['date', 'X', 'Y', 'Z']]

    return data


def daily_mean_conversion(data):
    """Use the tabular base to calculate daily means in nT or degrees (D, I)

    Uses the tabular base and daily value from the WDC file to calculate the
    daily means of magnetic field components. Value is in nT for H, F, X, Y or
    Z components and in degrees for D or I components. Called by wdc_xyz.

    daily_mean = tabular_base*100 + wdc_daily_value (for components in nT)

    daily_mean = tabular_base + wdc_daily_value/600 (for D and I components)

    Args:
        data (pandas.DataFrame): dataframe containing columns for datetime
            objects, magnetic field component (D, I, F, H, X, Y or Z), the
            tabular base and daily mean.

    Returns:
        data (pandas.DataFrame): the same dataframe with datetime objects in
            the first column and daily means of the field components in either
            nT or degrees (depending on the component)."""

    grp = pd.DataFrame()
    for group in data.groupby('component'):

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
    """Use D and H values to calculate the X and Y field components.

    The declination (D) and horizontal intensity (H) relate to the north (Y)
    and east (X) components as follows:

    X = H*cos(D)

    Y = H*sin(D)

    Args:
        data (pandas.DataFrame): dataframe containing columns for datetime
            objects and daily means of the magnetic field components (D, I, F,
            H, X, Y or Z).

    Returns:
        data (pandas.DataFrame): the same dataframe with datetime objects in
            the first column and daily means of the field components in either
            nT or degrees (depending on the component)."""

    data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'X'] = data.loc[(
        ~np.isnan(data['D']) & ~np.isnan(data['H'])), 'H']*np.cos(np.deg2rad(
            data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'D']))

    data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'Y'] = data.loc[(
        ~np.isnan(data['D']) & ~np.isnan(data['H'])), 'H']*np.sin(np.deg2rad(
            data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'D']))

    return data


def data_resampling(data, sampling='M'):
    """Resample the daily geomagnetic data to a specified frequency.

        Args:
            data (pandas.DataFrame): dataframe containing datetime objects and
                daily means of magnetic data.
            sampling (str): new sampling frequency. Default value is 'M'
                (monthly means), which averages data for each month and sets
                the datetime object to the final day of that month. Use 'MS'
                to set the datetime object to the first day of the month.
                Another useful option is 'A' (annual means), which averages
                data for a whole year and sets the datetime object to the final
                day of the year. Use 'AS' to set the datetime object to the
                first day of the year.

        Returns:
            data (pandas.DataFrame): dataframe of datetime objects and
                monthly/annual means of observatory data."""

    resampled = data.set_index('date', drop=False).resample(
            sampling, how='mean')
    resampled.reset_index(inplace=True)

    return resampled


def wdc_readfile(fname):
    """Wrapper function to call wdc_parsefile, wdc_readfile and wdc_xyz.

    Args:
        fname (str): path to a WDC datafile.

    Returns:
        data (pandas.DataFrame): dataframe containing the data read from the
            WDC file. First column is a series of datetime objects (in the
            format yyyy-mm-dd) and subsequent columns are the X, Y and Z
            components of the magnetic field at the specified times."""

    rawdata = wdc_parsefile(fname)
    rawdata = wdc_datetimes(rawdata)
    data = wdc_xyz(rawdata)

    return data


def append_wdc_data(obs_name,
                    path='/Users/Grace/Dropbox/BGS_data/hourval/\
single_obs/%s/*.wdc'):

    """Append all WDC data for an observatory into a single dataframe.

    Args:
        obs_name (str): observatory name (as 3-digit IAGA code).

    Returns:
        data (pandas.DataFrame): dataframe containing all available daily
            geomagnetic data at a single observatory. First column is a series
            of datetime objects (in the format yyyy-mm-dd) and subsequent
            columns are the X, Y and Z components of the magnetic field at the
            specified times."""

    df = pd.DataFrame()

    data_path = path % obs_name

    filenames = glob.glob(data_path)
    for f in filenames:
        try:
            frame = wdc_readfile(f)
            df = df.append(frame, ignore_index=True)
        except StopIteration:
            pass

    return df


def covobs_parsefile(fname):
    """Load a datafile containing SV predictions from a field model.

    Args:
        fname (str): path to a COV-OBS datafile.

    Returns:
        data (pandas.DataFrame): dataframe containing daily geomagnetic
            data. First column is a series of datetime objects (in the format
            yyyy-mm-dd) and subsequent columns are the X, Y and Z components of
            the SV at the specified times."""

    model_data = pd.read_csv(fname, sep='\s+', header=None,
                             usecols=[0, 1, 2, 3])

    model_data.columns = ["year_decimal", "dx", "dy", "dz"]


def covobs_datetimes(data):
    """Create datetime objects from the year column of a COV-OBS output file.

    The format output by the field model is year.decimalmonth e.g. 1960.08 is
    Jan 1960

    Args:
        data (pandas.DataFrame): needs a column for decimal year (in yyyy.mm
            format). Called by covobs_parsefile.

    Returns:
        data (pandas.DataFrame): the same dataframe with the decimal year
            column replced with a series of datetime objects in the format
            yyyy-mm-dd"""

    year_temp = np.floor(data.year_decimal.values.astype(
            'float64')).astype('int')

    months = (12*(data.year_decimal - year_temp) + 1).round().astype('int')

    data.drop(data.columns[[0]], axis=1, inplace=True)
    data.insert(0, 'year', year_temp)
    data.insert(1, 'month', months)

    date = data.apply(lambda x: dt.datetime.strptime(
            "{0} {1}".format(int(x['year']), int(x['month'])), "%Y %m"),
            axis=1)

    data.insert(0, 'date', date)
    data.drop(data.columns[[1, 2]], axis=1, inplace=True)

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


def covobs_readfile(fname):
    """Wrapper function to call covobs_parsefile and covobs_datetimes.

    Args:
        fname (str): path to a COV-OBS format datafile.

    Returns:
        data (pandas.DataFrame): dataframe containing the data read from the
            file. First column is a series of datetime objects (in the
            format yyyy-mm-dd) and subsequent columns are the X, Y and Z
            components of the SV at the specified times."""

    rawdata = covobs_parsefile(fname)
    data = covobs_datetimes(rawdata)

    return data


def write_csv_data(data, data_path, obs_name):

            fpath = data_path + obs_name + '.csv'
            data.to_csv(fpath, sep=' ', header=True, index=False)


def read_csv_data(fname):

    col_names = ['date', 'component', 'mean']
    data = pd.read_csv(fname, sep=' ', header=0, names=col_names,
                       parse_dates=[0], dayfirst=True)
    return data


def combine_csv_data(obs_list, data_path, model_path):

    for observatory in obs_list:

        obs_fname = data_path + observatory + '.csv'
        model_fname = model_path + observatory + '.csv'
        obs_data_temp = read_csv_data(obs_fname)
        model_data_temp = read_csv_data(model_fname)
        # Combine the current observatory data with those of other
        # observatories
        if observatory == obs_list[0]:
            obs_data = obs_data_temp
            model_data = model_data_temp

        else:
            obs_data = pd.merge(
                               left=obs_data, right=obs_data_temp,
                               how='left', on='date', suffixes=obs_list)
            model_data = pd.merge(
                               left=model_data, right=model_data_temp,
                               how='left', on='date', suffixes=obs_list)

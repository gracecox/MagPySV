# -*- coding: utf-8 -*-
#    Copyright (C) 2016  Grace Cox (University of Liverpool)
#
#    Released under the MIT license, a copy of which is located at the root of
#    this project.
"""Module containing functions to parse World Data Centre (WDC) files.

Part of the MagPySV package for geomagnetic data analysis. This module provides
various functions to read, parse and manipulate the contents of World Data
Centre (WDC) formatted files containing geomagnetic data and output data to
comma separated values (CSV) files. Also contains functions to read output of
the COV-OBS magnetic field model series by Gillet et al. (2013, Geochem.
Geophys. Geosyst.; 2015, Earth, Planets and Space).
"""


import datetime as dt
import glob
import os
import pandas as pd
import numpy as np


def wdc_parsefile(fname):
    """Load a WDC datafile and place the contents into a dataframe.

    Load a datafile of WDC hourly geomagnetic data for a single observatory and
    extract the contents. Can parse both the current WDC file format and the
    previous format containing international quiet (Q) or disturbed (D) day
    designation in place of the century field.

    Args:
        fname (str): path to a WDC datafile.

    Returns:
        data (pandas.DataFrame):
            dataframe containing daily geomagnetic data. First column is a
            series of datetime objects (in the format yyyy-mm-dd) and
            subsequent columns are the X, Y and Z components of the magnetic
            field at the specified times.
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
        data (pandas.DataFrame):
            the same dataframe with a series of datetime objects (in the format
            yyyy-mm-dd) in the first column.
    """
    # Convert the century/yr columns to a year
    data['year'] = 100 * data['century'] + data['yr']

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
        data (pandas.DataFrame):
            the same dataframe with datetime objects in the first column and
            columns for X, Y and Z components of magnetic field (in nT).
    """
    # Replace missing values with NaNs
    data.replace(9999, np.nan, inplace=True)
    # Group the data by field component, calculate the daily means and form
    # a dataframe with separate columns for each field component
    data = data.groupby('component').apply(daily_mean_conversion)
    data.reset_index(drop=True, inplace=True)
    data.drop(['base', 'daily_mean_temp'], axis=1, inplace=True)
    data = data.pivot_table(index='date', columns='component',
                            values='daily_mean')
    data.reset_index(inplace=True)

    # Call helper function to convert D and H components to X and Y
    if 'D' in data.columns and 'H' in data.columns:
        data = angles_to_geographic(data)
        if 'X' in data.columns and 'Y' in data.columns and 'Z' in data.columns:
            data = data[['date', 'X', 'Y', 'Z']]

    else:
        if 'X' in data.columns and 'Y' in data.columns and 'Z' in data.columns:
            data = data[['date', 'X', 'Y', 'Z']]
    # Make sure that the dataframe contains columns for X, Y and Z components,
    # and create a column of NaN values if a component is missing
    if 'X' not in data.columns:
        data['X'] = np.NaN
    if 'Y' not in data.columns:
        data['Y'] = np.NaN
    if 'Z' not in data.columns:
        data['Z'] = np.NaN

    return data


def daily_mean_conversion(data):
    """Use the tabular base to calculate daily means in nT or degrees (D, I).

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
        data (pandas.DataFrame):
            the same dataframe with datetime objects in the first column and
            daily means of the field components in either nT or degrees
            (depending on the component).
    """
    grp = pd.DataFrame()
    for group in data.groupby('component'):

        if group[0] == 'D' or group[0] == 'I':
            group[1]['daily_mean'] = group[1]['base'] + \
                (1 / 600.0) * group[1]['daily_mean_temp']
            grp = grp.append(group[1], ignore_index=True)
        else:
            group[1]['daily_mean'] = 100.0 * group[1]['base'] + \
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
        data (pandas.DataFrame):
            the same dataframe with datetime objects in the first column and
            daily means of the field components in either nT or degrees
            (depending on the component).
    """
    data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'X'] = data.loc[(
        ~np.isnan(data['D']) & ~np.isnan(data['H'])), 'H'] * np.cos(np.deg2rad(
            data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'D']))

    data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'Y'] = data.loc[(
        ~np.isnan(data['D']) & ~np.isnan(data['H'])), 'H'] * np.sin(np.deg2rad(
            data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'D']))

    return data


def wdc_readfile(fname):
    """Wrapper function to call wdc_parsefile, wdc_readfile and wdc_xyz.

    Args:
        fname (str): path to a WDC datafile.

    Returns:
        data (pandas.DataFrame):
            dataframe containing the data read from the WDC file. First column
            is a series of datetime objects (in the format yyyy-mm-dd) and
            subsequent columns are the X, Y and Z components of the magnetic
            field at the specified times.
    """
    rawdata = wdc_parsefile(fname)
    rawdata = wdc_datetimes(rawdata)
    data = wdc_xyz(rawdata)

    return data


def append_wdc_data(*, obs_name,
                    path='./data/BGS_data/hourval/single_obs/%s/*.wdc'):

    """Append all WDC data for an observatory into a single dataframe.

    Args:
        obs_name (str): observatory name (as 3-digit IAGA code).
        path (str): path to a directory containing WDC datafiles for a single
            observatory.

    Returns:
        data (pandas.DataFrame):
            dataframe containing all available daily geomagnetic data at a
            single observatory. First column is a series of datetime objects
            (in the format yyyy-mm-dd) and subsequent columns are the X, Y and
            Z components of the magnetic field at the specified times.
    """
    data = pd.DataFrame()

    data_path = path % obs_name

    filenames = glob.glob(data_path)
    for file in filenames:
        print(file)
        try:
            frame = wdc_readfile(file)
            data = data.append(frame, ignore_index=True)
        except StopIteration:
            pass

    return data


def covobs_parsefile(fname):
    """Loads MF and SV predictions from the COV-OBS geomagnetic field model.

    Load a datafile containing SV/MF predictions from the COV-OBS magnetic
    field model series by Gillet et al. (2013, Geochem. Geophys. Geosyst.;
    2015, Earth, Planets and Space) field model.

    Args:
        fname (str): path to a COV-OBS datafile.

    Returns:
        model_data (pandas.DataFrame):
            dataframe containing daily geomagnetic data. First column is a
            series of datetime objects (in the format yyyy-mm-dd) and
            subsequent columns are the X, Y and Z components of the SV/MF at
            the specified times.
    """
    model_data = pd.read_csv(fname, sep=r'\s+', header=None,
                             usecols=[0, 1, 2, 3])

    model_data.columns = ["year_decimal", "dx", "dy", "dz"]

    return model_data


def covobs_datetimes(data):
    """Create datetime objects from COV-OBS field model output file.

    The format output by the field model is year.decimalmonth e.g. 1960.08 is
    Jan 1960

    Args:
        data (pandas.DataFrame): needs a column for decimal year (in yyyy.mm
            format).

    Returns:
        data (pandas.DataFrame):
            the same dataframe with the decimal year column replced with a
            series of datetime objects in the format yyyy-mm-dd.
    """
    year_temp = np.floor(data.year_decimal.values.astype(
        'float64')).astype('int')

    months = (12 * (data.year_decimal - year_temp) + 1).round().astype('int')

    data.insert(0, 'year', year_temp)
    data.insert(1, 'month', months)

    date = data.apply(lambda x: dt.datetime.strptime(
        "{0} {1}".format(int(x['year']), int(x['month'])), "%Y %m"),
        axis=1)

    data.insert(0, 'date', date)

    data.drop(['year', 'year_decimal', 'month'], axis=1, inplace=True)

    return data


def covobs_readfile(fname):
    """Wrapper function to call covobs_parsefile and covobs_datetimes.

    Args:
        fname (str): path to a COV-OBS format datafile.

    Returns:
        data (pandas.DataFrame):
            dataframe containing the data read from the file. First column is a
            series of datetime objects (in the format yyyy-mm-dd) and
            subsequent columns are the X, Y and Z components of the SV/MF at
            the specified times.
    """

    rawdata = covobs_parsefile(fname)
    data = covobs_datetimes(rawdata)

    return data


def wdc_to_daily_csv(*, fpath='./data/BGS_hourly/', write_path,
                     print_obs=True):
    """Converts hourly WDC data to X, Y and Z daily means and save to CSV file.

    Finds WDC hourly data files for all observatories in a directory path
    (assumes data for each observatory is located inside a directory named
    after the observatory). The WDC distributes data inside directories
    with the naming convention /hourval/single_obs/obs/obsyear.wdc where obs is
    a three digit observatory name, year is a four digit year and the string
    /hourval/single_obs prepends each directory name. e.g.
    /hourval/single_obs/ngk/ngk1990.wdc or /hourval/single_obs/clf/clf2013.wdc.
    This function converts the hourly data to daily X, Y and Z means, appends
    all years of data for a single observatory into a single dataframe and
    saves the dataframe to a CSV file.

    Args:
        fpath (str): path to the datafiles. Assumes data for each observatory
            is stored in a directory named after the observatory.
        write_path (str): path to which the output CSV files are written.
        print_obs (bool): choose whether to print each observatory name as the
            function goes through the directories. Useful for checking progress
            as it can take a while to read the whole WDC dataset.
    """
    wdc_path = fpath + 'hourval/single_obs/*'
    dir_list = glob.glob(wdc_path)
    obs_names = [os.path.basename(obs_path) for obs_path in dir_list]
    for observatory in obs_names:
        if print_obs is True:
            print(observatory)
        wdc_data = append_wdc_data(
            obs_name=observatory,
            path=fpath + '/hourval/single_obs/%s/*.wdc')
        write_csv_data(data=wdc_data, write_path=write_path,
                       obs_name=observatory)


def write_csv_data(*, data, write_path, obs_name, file_prefix=None):
    """Write dataframe to a CSV file.

    Args:
        data (pandas.DataFrame): data to be written to file.
        write_path (str): path to which the output CSV file is written.
        obs_name (str): name of observatory at which the data were obtained.
        file_prefix (str): optional string to prefix the output CSV filenames
            (useful for specifying parameters used to create the dataset etc).
    """
    if file_prefix is not None:
        fpath = write_path + file_prefix + obs_name + '.csv'
    else:
        fpath = write_path + obs_name + '.csv'
    data.to_csv(fpath, sep=',', na_rep='NA', header=True, index=False)


def read_csv_data(fname):
    """Read dataframe from a CSV file.

    Args:
        fname (str): path to a CSV datafile.

    Returns:
        data (pandas.DataFrame):
            dataframe containing the data read from the CSV file.
    """
    col_names = ['date', 'X', 'Y', 'Z']
    data = pd.read_csv(fname, sep=',', header=0, names=col_names,
                       parse_dates=[0])
    return data


def combine_csv_data(*, start_date, end_date, sampling_rate='MS',
                     obs_list, data_path, model_path):
    """Read and combine observatory and model SV data for several locations.

    Calls read_csv_data to read observatory data and field model predictions
    for each observatory in a list. The data and predictions for individual
    observatories are combined into their respective large dataframes. The
    first column contains datetime objects and subsequent columns contain X, Y
    and Z secular variation/field components (in groups of three) for all
    observatories.

    Args:
        start_date (datetime.datetime): the start date of the data analysis.
        end_date (datetime.datetime): the end date of the analysis.
        sampling_rate (str): the sampling rate for the period of interest. The
            default is 'MS', which creates a range of dates between the
            specified values at monthly intervals with the day fixed as the
            first of each month. Use 'M' for the final day of each month. Other
            useful options are 'AS' (a series of dates at annual intervals,
            with the day and month fixed at 01 and January respectively) and
            'A' (as for 'AS' but with the day/month fixed as 31 December.)
        obs_list (list): list of observatory names (as 3-digit IAGA codes).
        data_path (str): path to the CSV files containing observatory data.
        model_path (str): path to the CSV files containing model SV data.

    Returns:
        (tuple): tuple containing:

        - obs_data (*pandas.DataFrame*):
            dataframe containing SV data for all observatories in obs_list.
        - model_sv_data (*pandas.DataFrame*):
            dataframe containing SV predictions for all observatories in
            obs_list.
        - model_mf_data (*pandas.DataFrame*):
            dataframe containing magnetic field predictions for all
            observatories in obs_list.
    """
    # Initialise the dataframe with the appropriate date range
    obs_data = pd.DataFrame({'date': pd.date_range(start_date, end_date,
                                                   freq=sampling_rate)})
    model_sv_data = pd.DataFrame({'date': pd.date_range(
                                start_date, end_date,
                                freq=sampling_rate)})
    model_mf_data = pd.DataFrame({'date': pd.date_range(
                                start_date, end_date,
                                freq=sampling_rate)})

    for observatory in obs_list:

        obs_file = observatory + '.csv'
        model_sv_file = 'sv_' + observatory + '.dat'
        model_mf_file = 'mf_' + observatory + '.dat'
        obs_data_temp = read_csv_data(os.path.join(data_path, obs_file))
        model_sv_data_temp = covobs_readfile(os.path.join(model_path,
                                                          model_sv_file))
        model_mf_data_temp = covobs_readfile(os.path.join(model_path,
                                                          model_mf_file))
        obs_data_temp.rename(
            columns={'X': 'X' + '_' + observatory,
                     'Y': 'Y' + '_' + observatory,
                     'Z': 'Z' + '_' + observatory}, inplace=True)
        model_sv_data_temp.rename(
            columns={'dx': 'dx' + '_' + observatory,
                     'dy': 'dy' + '_' + observatory,
                     'dz': 'dz' + '_' + observatory}, inplace=True)
        model_mf_data_temp.rename(
            columns={'dx': 'x' + '_' + observatory,
                     'dy': 'y' + '_' + observatory,
                     'dz': 'z' + '_' + observatory}, inplace=True)
        # Combine the current observatory data with those of other
        # observatories
        if observatory == obs_list[0]:
            obs_data = pd.merge(
                left=obs_data, right=obs_data_temp,
                how='left', on='date')
            model_sv_data = pd.merge(
                left=model_sv_data, right=model_sv_data_temp,
                how='left', on='date')
            model_mf_data = pd.merge(
                left=model_mf_data, right=model_mf_data_temp,
                how='left', on='date')

        else:
            obs_data = pd.merge(
                left=obs_data, right=obs_data_temp,
                how='left', on='date')
            model_sv_data = pd.merge(
                left=model_sv_data, right=model_sv_data_temp,
                how='left', on='date')
            model_mf_data = pd.merge(
                left=model_mf_data, right=model_mf_data_temp,
                how='left', on='date')
    return obs_data, model_sv_data, model_mf_data

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
code used for the COV-OBS magnetic field model series by Gillet et al. (links
below).
"""


import datetime as dt
import glob
import os
import pandas as pd
import numpy as np


def wdc_parsefile(fname):
    """Load a WDC datafile and place the contents into a dataframe.

    Load a datafile of WDC hourly geomagnetic data for a single observatory and
    extract the contents. Parses the current WDC file format, but not the
    previous format containing international quiet (Q) or disturbed (D) day
    designation in place of the century field - only the newer format is
    downloaded from the BGS servers. Detailed file format description
    can be found at http://www.wdc.bgs.ac.uk/catalog/format.html

    Args:
        fname (str): path to a WDC datafile.

    Returns:
        data (pandas.DataFrame):
            dataframe containing hourly geomagnetic data. First column is a
            series of datetime objects (in the format yyyy-mm-dd hh:30:00) and
            subsequent columns are the X, Y and Z components of the magnetic
            field at the specified times.
    """
    # New WDC file format
    cols = [(0, 3), (3, 5), (5, 7), (7, 8), (8, 10), (14, 16),
            (16, 20), (20, 116)]
    col_names = [
        'code', 'yr', 'month', 'component', 'day', 'century',
        'base', 'hourly_values']
    types = {
        'code': str, 'year': int, 'month': int, 'component': str,
        'day': int, 'century': int, 'base': int, 'hourly_values': str}
    data = pd.read_fwf(fname, colspecs=cols, names=col_names,
                       converters=types, header=None)
    data['hourly_values'] = data['hourly_values'].apply(
                                                      separate_hourly_vals)
    data = data.set_index(['code', 'yr', 'month', 'component', 'day',
                           'century', 'base'])['hourly_values'].apply(
                           pd.Series).stack()
    data = data.reset_index()
    data.columns = ['code', 'yr', 'month', 'component', 'day', 'century',
                    'base', 'hour', 'hourly_mean_temp']
    data['hourly_mean_temp'] = data['hourly_mean_temp'].astype(float)

    return data


def separate_hourly_vals(hourstring):
    """Separate individual hourly field means from the string containing all
    24 values in the WDC file. Called by wdc_parsefile.

    Args:
        hourstring (str): string containing the hourly magnetic field means
            parsed from a WDC file for a single day.

    Returns:
        hourly_vals_list (list):
            list containing the hourly field values.
    """
    n = 4
    hourly_vals_list = [hourstring[i:i+n] for i in range(0, len(hourstring),
                        n)]
    return hourly_vals_list


def wdc_datetimes(data):
    """Create datetime objects from the fields extracted from a WDC datafile.

    Args:
        data (pandas.DataFrame): needs columns for century, year (yy format),
            month, day and hour. Called by wdc_parsefile.

    Returns:
        data (pandas.DataFrame):
            the same dataframe with a series of datetime objects (in the format
            yyyy-mm-dd hh:30:00) in the first column.
    """
    # Convert the century/yr columns to a year
    data['year'] = 100 * data['century'] + data['yr']

    # Create datetime objects from the century, year, month and day columns of
    # the WDC format data file. The hourly mean is given at half past the hour
    dates = data.apply(lambda x: dt.datetime.strptime(
        "{0} {1} {2} {3} {4}".format(x['year'], x['month'], x['day'],
                                     x['hour'], 30), "%Y %m %d %H %M"), axis=1)
    data.insert(0, 'date', dates)
    data.drop(['year', 'yr', 'century', 'code', 'day', 'month', 'hour'],
              axis=1, inplace=True)

    return data


def wdc_xyz(data):
    """Convert extracted WDC data to hourly X, Y and Z components in nT.

    Missing values (indicated by 9999 in the datafiles) are replaced with NaNs.

    Args:
        data (pandas.DataFrame): dataframe containing columns for datetime
            objects, magnetic field component (D, I, F, H, X, Y or Z), the
            tabular base and hourly mean.

    Returns:
        data (pandas.DataFrame):
            the same dataframe with datetime objects in the first column and
            columns for X, Y and Z components of magnetic field (in nT).
    """
    # Replace missing values with NaNs
    data.replace(9999, np.nan, inplace=True)
    # Group the data by field component, calculate the hourly means and form
    # a dataframe with separate columns for each field component
    data = data.groupby('component').apply(hourly_mean_conversion)
    data.reset_index(drop=True, inplace=True)
    data.drop(['base', 'hourly_mean_temp'], axis=1, inplace=True)
    data = data.pivot_table(index='date', columns='component',
                            values='hourly_mean')
    data.reset_index(inplace=True)

    # Call helper function to convert D and H components to X and Y
    if 'D' in data.columns and 'H' in data.columns:
        data = angles_to_geographic(data)

    # Make sure that the dataframe contains columns for X, Y and Z components,
    # and create a column of NaN values if a component is missing
    if 'X' not in data.columns:
        data['X'] = np.NaN
    if 'Y' not in data.columns:
        data['Y'] = np.NaN
    if 'Z' not in data.columns:
        data['Z'] = np.NaN

    data = data[['date', 'X', 'Y', 'Z']]

    return data


def hourly_mean_conversion(data):
    """Use the tabular base to calculate hourly means in nT or degrees (D, I).

    Uses the tabular base and hourly value from the WDC file to calculate the
    hourly means of magnetic field components. Value is in nT for H, F, X, Y or
    Z components and in degrees for D or I components. Called by wdc_xyz.

    hourly_mean = tabular_base*100 + wdc_hourly_value (for components in nT)

    hourly_mean = tabular_base + wdc_hourly_value/600 (for D and I components)

    Args:
        data (pandas.DataFrame): dataframe containing columns for datetime
            objects, magnetic field component (D, I, F, H, X, Y or Z), the
            tabular base and hourly mean.

    Returns:
        obs_data (pandas.DataFrame):
            dataframe with datetime objects in the first column and hourly
            means of the field components in either nT or degrees (depending on
            the component).
    """
    obs_data = pd.DataFrame()
    for group in data.groupby('component'):

        if group[0] == 'D' or group[0] == 'I':
            group[1]['hourly_mean'] = group[1]['base'] + \
                (1 / 600.0) * group[1]['hourly_mean_temp']
            obs_data = obs_data.append(group[1], ignore_index=True)
        else:
            group[1]['hourly_mean'] = 100.0 * group[1]['base'] + \
                group[1]['hourly_mean_temp']
            obs_data = obs_data.append(group[1], ignore_index=True)
    return obs_data


def angles_to_geographic(data):
    """Use D and H values to calculate the X and Y field components.

    The declination (D) and horizontal intensity (H) relate to the north (Y)
    and east (X) components as follows:

    X = H*cos(D)

    Y = H*sin(D)

    Args:
        data (pandas.DataFrame): dataframe containing columns for datetime
            objects and hourly means of the magnetic field components (D, I, F,
            H, X, Y or Z).

    Returns:
        data (pandas.DataFrame):
            the same dataframe with datetime objects in the first column and
            hourly means of the field components in either nT or degrees
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
    """Wrapper function to call wdc_parsefile, wdc_datetimes and wdc_xyz.

    Args:
        fname (str): path to a WDC datafile.

    Returns:
        data (pandas.DataFrame):
            dataframe containing the data read from the WDC file. First column
            is a series of datetime objects (in the format yyyy-mm-dd hh:30:00)
            and subsequent columns are the X, Y and Z components of the
            magnetic field at the specified times (hourly means).
    """
    rawdata = wdc_parsefile(fname)
    rawdata = wdc_datetimes(rawdata)
    data = wdc_xyz(rawdata)

    return data


def append_wdc_data(*, obs_name, path=None):
    """Append all WDC data for an observatory into a single dataframe.

    Args:
        obs_name (str): observatory name (as 3-digit IAGA code).
        path (str): path to directory containing WDC datafiles. All files for
            the observatory should be located in the same directory.

    Returns:
        data (pandas.DataFrame):
            dataframe containing all available hourly geomagnetic data at a
            single observatory. First column is a series of datetime objects
            (in the format yyyy-mm-dd hh:30:00) and subsequent columns are the
            X, Y and Z components of the magnetic field at the specified times.
    """
    data = pd.DataFrame()

    data_path = os.path.join(path, obs_name.lower() + '*.wdc')
    # Obtain a list of all files containing the observatory name and ending
    # .wdc in the specified directory
    filenames = glob.glob(data_path)
    # Iterate over the files and append them to previous files
    for file in filenames:
        print(file)
        try:
            frame = wdc_readfile(file)
            data = data.append(frame, ignore_index=True)
        except StopIteration:
            pass

    return data


def covobs_parsefile(*, fname, data_type):
    """Loads MF and SV predictions from the COV-OBS geomagnetic field model.

    Load a datafile containing SV/MF predictions from the COV-OBS magnetic
    field model series by Gillet et al. (2013, Geochem. Geophys. Geosyst.,
    https://doi.org/10.1002/ggge.20041;
    2015, Earth, Planets and Space, https://doi.org/10.1186/s40623-015-0225-z)
    field model.

    Args:
        fname (str): path to a COV-OBS datafile.
        data_type (str): specify whether the file contains magnetic field data
            ('mf') or or secular variation data ('sv')

    Returns:
        model_data (pandas.DataFrame):
            dataframe containing hourly geomagnetic data. First column is a
            series of datetime objects (in the format yyyy-mm-dd) and
            subsequent columns are the X, Y and Z components of the SV/MF at
            the specified times.
    """
    model_data = pd.read_csv(fname, sep=r'\s+', header=None,
                             usecols=[0, 1, 2, 3])
    if data_type is 'mf':
        model_data.columns = ["year_decimal", "X", "Y", "Z"]
    else:
        model_data.columns = ["year_decimal", "dX", "dY", "dZ"]
    return model_data


def covobs_datetimes(data):
    """Create datetime objects from COV-OBS field model output file.

    The format output by the field model is year.decimalmonth e.g. 1960.08 is
    Jan 1960.

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


def covobs_readfile(*, fname, data_type):
    """Wrapper function to call covobs_parsefile and covobs_datetimes.

    The COV-OBS code (publically available) can be used to produce synthetic
    observatory time series for other field models if the appropriate spline
    file is used. The output will be of the same format as COV-OBS output and
    can be read using MagPySV.

    Args:
        fname (str): path to a COV-OBS format datafile.
        data_type (str): specify whether the file contains magnetic field data
            ('mf') or or secular variation data ('sv')
    Returns:
        data (pandas.DataFrame):
            dataframe containing the data read from the file. First column is a
            series of datetime objects (in the format yyyy-mm-dd) and
            subsequent columns are the X, Y and Z components of the SV/MF at
            the specified times.
    """

    rawdata = covobs_parsefile(fname=fname, data_type=data_type)
    data = covobs_datetimes(rawdata)

    return data


def wdc_to_hourly_csv(*, wdc_path=None, write_dir, obs_list,
                      print_obs=True):
    """Convert WDC file to X, Y and Z hourly means and save to CSV file.

    Finds WDC hourly data files for all observatories in a directory path
    (assumes data for all observatories are located inside the same directory).
    The BGS downloading app distributes data inside a single directory
    with the naming convention obsyear.wdc where obs is a three digit
    observatory name in lowercase and year is a four digit year,
    e.g. ngk1990.wdc or clf2013.wdc. This function converts the hourly WDC
    format data to hourly X, Y and Z means, appends all years of data for a
    single observatory into a single dataframe and saves the dataframe to a
    CSV file.

    Args:
        wdc_path (str): path to the directory containing datafiles.
        write_dir (str): directory path to which the output CSV files are
            written.
        obs_list (list): list of observatory names (as 3-digit IAGA codes).
        print_obs (bool): choose whether to print each observatory name as the
            function goes through the directories. Useful for checking progress
            as it can take a while to read the whole WDC dataset. Defaults to
            True.
    """
    # Create the output directory if it does not exist
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    # Iterate over each given observatory and call append_wdc_data
    for observatory in obs_list:
        if print_obs is True:
            print(observatory)
        wdc_data = append_wdc_data(
            obs_name=observatory,
            path=wdc_path)
        write_csv_data(data=wdc_data, write_dir=write_dir,
                       obs_name=observatory)


def write_csv_data(*, data, write_dir, obs_name, file_prefix=None,
                   decimal_dates=False, header=True):
    """Write dataframe to a CSV file.

    Args:
        data (pandas.DataFrame): data to be written to file.
        write_dir (str): directory path to which the output CSV file is
            written.
        obs_name (str): name of observatory at which the data were obtained.
        file_prefix (str): optional string to prefix the output CSV filenames
            (useful for specifying parameters used to create the dataset etc).
        decimal_dates (bool): optional argument to specify that dates should be
            written in decimal format rather than datetime objects. Defaults to
            False.
        header (bool): option to include header in file. Defaults to True.
    """

    # Create the output directory if it does not exist
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    # Convert datetime objects to decimal dates if required
    if decimal_dates is True:
        data.date = data.date.apply(datetime_to_decimal)
    if file_prefix is not None:
        fpath = os.path.join(write_dir, file_prefix + obs_name + '.csv')
    else:
        fpath = os.path.join(write_dir, obs_name + '.csv')
    data.to_csv(fpath, sep=',', na_rep='NA', header=header, index=False)


def read_csv_data(*, fname, data_type):
    """Read dataframe from a CSV file.

    Args:
        fname (str): path to a CSV datafile.
        data_type (str): specify whether the file contains magnetic field data
            ('mf') or or secular variation data ('sv')

    Returns:
        data (pandas.DataFrame):
            dataframe containing the data read from the CSV file.
    """
    if data_type is 'mf':
        col_names = ['date', 'X', 'Y', 'Z']
    else:
        col_names = ['date', 'dX', 'dY', 'dZ']
    data = pd.read_csv(fname, sep=',', header=0, names=col_names,
                       parse_dates=[0])
    return data


def combine_csv_data(*, start_date, end_date, sampling_rate='MS',
                     obs_list, data_path, model_path, day_of_month=1):
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
        day_of_month (int): For SV data, first differences of
            monthly means have dates at the start of the month (i.e. MF of
            mid-Feb minus MF of mid-Jan should give SV at Feb 1st. For annual
            differences of monthly means the MF of mid-Jan year 2 minus MF of
            mid-Jan year 1 gives SV at mid-July year 1. The dates of COV-OBS
            output default to the first day of the month (compatible with dates
            of monthly first differences SV data, but not with those of
            annual differences). This option is used to set the day part of the
            dates column if required. Default to 1 (all output dataframes
            will have dates set at the first day of the month.)

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
    dates = pd.date_range(start_date, end_date, freq=sampling_rate)
    obs_data = pd.DataFrame({'date': dates})
    model_sv_data = pd.DataFrame({'date': dates})
    model_mf_data = pd.DataFrame({'date': dates})

    for observatory in obs_list:

        obs_file = observatory + '.csv'
        model_sv_file = 'sv_' + observatory + '.dat'
        model_mf_file = 'mf_' + observatory + '.dat'
        obs_data_temp = read_csv_data(fname=os.path.join(data_path, obs_file),
                                      data_type='sv')
        model_sv_data_temp = covobs_readfile(fname=os.path.join(model_path,
                                             model_sv_file), data_type='sv')
        model_mf_data_temp = covobs_readfile(fname=os.path.join(model_path,
                                             model_mf_file), data_type='mf')

        model_sv_data_temp['date'] = model_sv_data_temp['date'].apply(
            lambda dt: dt.replace(day=1))

        obs_data_temp.rename(
            columns={'dX': 'dX' + '_' + observatory,
                     'dY': 'dY' + '_' + observatory,
                     'dZ': 'dZ' + '_' + observatory}, inplace=True)
        obs_data_temp['date'] = obs_data_temp['date'].apply(
            lambda dt: dt.replace(day=1))
        model_sv_data_temp.rename(
            columns={'dX': 'dX' + '_' + observatory,
                     'dY': 'dY' + '_' + observatory,
                     'dZ': 'dZ' + '_' + observatory}, inplace=True)
        model_mf_data_temp.rename(
            columns={'X': 'X' + '_' + observatory,
                     'Y': 'Y' + '_' + observatory,
                     'Z': 'Z' + '_' + observatory}, inplace=True)
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
    if day_of_month is not 1:
        model_sv_data['date'] = model_sv_data['date'].apply(
            lambda dt: dt.replace(day=day_of_month))
        model_mf_data['date'] = model_sv_data['date']
        obs_data['date'] = model_sv_data['date']
    return obs_data, model_sv_data, model_mf_data


def datetime_to_decimal(date):
    """Convert a datetime object to a decimal year.

    Args:
        date (datetime.datetime): datetime object representing an observation
            time.

    Returns:
        date (float): the same date represented as a decimal year.
    """
    year_start = dt.datetime(date.year, 1, 1)
    year_end = year_start.replace(year=date.year + 1)
    decimal_year = date.year + (date - year_start) / (year_end - year_start)
    return decimal_year


def ae_parsefile(fname):
    """Load a WDC-like format AE file and place contents into a dataframe.

    Load a file of AE (Auroral Electroject)
    index hourly data in the format distributed by the Kyoto WDC at
    http://wdc.kugi.kyoto-u.ac.jp/dstae/index.html and extract the contents.

    Args:
        fname (str): path to a WDC-like formatted AE file.

    Returns:
        data (pandas.DataFrame):
            dataframe containing hourly AE data. First column is a
            series of datetime objects (in the format yyyy-mm-dd hh:30:00) and
            second column contains theAE values at the specified times.
    """
    # AE WDC file format
    cols = [(0, 2), (3, 5), (5, 7), (8, 10), (14, 16),
            (16, 20), (20, 116)]
    col_names = [
        'code', 'yr', 'month', 'day', 'century',
        'base', 'hourly_values']
    types = {
        'code': str, 'year': int, 'month': int,
        'day': int, 'century': int, 'base': int, 'hourly_values': str}
    data = pd.read_fwf(fname, colspecs=cols, names=col_names,
                       converters=types, header=None)
    data = data.loc[data['code'] == "AE"]
    # Separate the hourly values
    try:
        data['hourly_values'] = data['hourly_values'].apply(
                                                    separate_hourly_vals)
    except ValueError:
        data['hourly_values'] = data['hourly_values'].apply(
            separate_hourly_vals_ae)
    data = data.set_index(['code', 'yr', 'month', 'day',
                           'century', 'base'])['hourly_values'].apply(
                           pd.Series).stack()
    data = data.reset_index()
    data.columns = ['code', 'yr', 'month', 'day', 'century',
                    'base', 'hour', 'hourly_mean_temp']
    data['hourly_mean_temp'] = data['hourly_mean_temp'].astype(float)
    return data


def separate_hourly_vals_ae(hourstring):
    """Separate individual hourly field means from the string containing all
    24 values in the AE file. Called by ae_parsefile.

    Args:
        hourstring (str): string containing the hourly AE means parsed from
            a Kyoto WDC-like file for a single day.

    Returns:
        hourly_vals_list (list):
            list containing the hourly AE values.
    """
    n = 4
    if hourstring[0] is not '-' and hourstring[0] is not ' ':
        hourstring = ' ' + hourstring
    hourly_vals_list = [hourstring[i:i+n] for i in range(0, len(hourstring),
                        n)]
    return hourly_vals_list


def ae_readfile(fname):
    """Wrapper function to call ae_parsefile and wdc_datetimes.

    Args:
        fname (str): path to a AE file in Kyoto WDC-like format. Assumes data
            for all years are contained within this file.

    Returns:
        data (pandas.DataFrame):
            dataframe containing the data read from the WDC file. First column
            is a series of datetime objects (in the format yyyy-mm-dd hh:30:00)
            and second column contains AE values at the specified times
            (hourly means).
    """
    data = ae_parsefile(fname)
    data = wdc_datetimes(data)
    data['hourly_mean'] = 100.0 * data['base'] + \
        data['hourly_mean_temp']
    data.drop(['hourly_mean_temp', 'base'], axis=1, inplace=True)
    return data


def append_ae_data(ae_data_path):
    """Append AE data into a single dataframe containing all years.

    Data downloaded from
    ftp://ftp.ngdc.noaa.gov/STP/GEOMAGNETIC_DATA/INDICES/AURORAL_ELECTROJET/HOURLY/
    come in WDC-like format files with one file per year named aeyyyy.wdc (data
    provided by the WDC at Kyoto. Can be downloaded directly from
    http://wdc.kugi.kyoto-u.ac.jp/dstae/index.html)

    Args:
        ae_data_path (str): path to directory containing WDC-like format AE
            datafiles. All AE files should be located in the same directory.

    Returns:
        data (pandas.DataFrame):
            dataframe containing all available hourly AE data. First column is
            a series of datetime objects (in the format yyyy-mm-dd hh:30:00)
            and second column contains AE values at the specified times.
    """
    data = pd.DataFrame()
    # Obtain a list of all files containing 'ae' and ending in .wdc in the
    # specified directory
    filenames = glob.glob(ae_data_path + 'ae*.txt')
    # Iterate over the files and append them to previous files
    for file in filenames:
        print(file)
        try:
            frame = ae_readfile(file)
            data = data.append(frame, ignore_index=True)
        except StopIteration:
            pass

    return data


def ap_readfile(fname):
    """Load an kp/ap file and place the hourly ap values into a dataframe.

    Load a datafile of 3-hourly ap data and extract the contents. Each of the
    3-hourly values for a given day is repeated three times to give an hourly
    mean for all 24 hours of the day. This function is designed to read files
    downloaded from the GFZ, Potsdam server at
    ftp://ftp.gfz-potsdam.de/pub/home/obs/kp-ap/.

    Args:
        fname (str): path to an ap datafile.

    Returns:
        data (pandas.DataFrame):
            dataframe containing hourly ap data. First column is a
            series of datetime objects (in the format yyyy-mm-dd hh:30:00) and
            second column contains ap values at the specified times.
    """
    col_names = ['full_string']
    types = {'full_string': str}
    # Parse the file
    if fname[-8] == '2':
        cols = [(1, 55)]
        data = pd.read_fwf(fname, colspecs=cols, names=col_names,
                           converters=types, header=None)
        data['month'] = data.full_string.str[1:3]
        data['day'] = data.full_string.str[3:5]
        data['hourly_values'] = data.full_string.str[30:]
    else:
        cols = [(0, 55)]
        data = pd.read_fwf(fname, colspecs=cols, names=col_names,
                           converters=types, header=None)
        data['month'] = data.full_string.str[2:4]
        data['day'] = data.full_string.str[4:6]
        data['hourly_values'] = data.full_string.str[32:]
    data.drop(['full_string'], axis=1, inplace=True)
    data['hourly_values'] = data['hourly_values'].apply(
        separate_three_hourly_vals)
    data = data.set_index(['month', 'day'])['hourly_values'].apply(
                               pd.Series).stack()
    data = data.reset_index()
    data.columns = ['month', 'day', 'hour', 'hourly_mean']
    data['hourly_mean'] = data['hourly_mean'].astype(float)
    data['year'] = int(fname[-8:-4])
    dates = data.apply(lambda x: dt.datetime.strptime(
        "{0} {1} {2} {3} {4}".format(x['year'], x['month'], x['day'],
                                     x['hour'], 30), "%Y %m %d %H %M"), axis=1)
    data.insert(0, 'date', dates)
    data.drop(['year', 'day', 'month', 'hour'],
              axis=1, inplace=True)
    return data


def separate_three_hourly_vals(hourstring):
    """Separate 3-hourly ap means from the string containing all 8 values.

    Separate the 8 individual 3-hourly ap means from the string containing all
    values for the day. Each value is repeated 3 times to give a value for each
    hour. Called by ap_readfile.

    Args:
        hourstring (str): string containing the 3-hourly ap means parsed from
            a Kyoto WDC-like file for a single day.

    Returns:
        hourly_vals_list (list):
            list containing the hourly ap values.
    """
    n = 3
    hourly_vals_list = [hourstring[i:i+n] for i in range(0, len(hourstring),
                        n)]
    hourly_vals_list = np.repeat(hourly_vals_list, n)
    return hourly_vals_list


def append_ap_data(ap_data_path):
    """Append ap data into a single dataframe containing all years.

    Data downloaded from ftp://ftp.gfz-potsdam.de/pub/home/obs/kp-ap/wdc/
    come in WDC-like format files with one file per year named kpyyyy.wdc. This
    function concatenates all years into a single dataframe.

    Args:
        ap_data_path (str): path to directory containing WDC-like format ap
            datafiles. All ap files should be located in the same directory.

    Returns:
        data (pandas.DataFrame):
            dataframe containing all available hourly ap data. First column is
            a series of datetime objects (in the format yyyy-mm-dd hh:30:00)
            and second column contains ap values at the specified times.
    """
    data = pd.DataFrame()
    # Obtain a list of all files containing 'ap' and ending in .wdc in the
    # specified directory
    filenames = glob.glob(ap_data_path + 'kp*.wdc')
    # Iterate over all files and append data
    for file in filenames:
        print(file)
        try:
            frame = ap_readfile(file)
            data = data.append(frame, ignore_index=True)
        except StopIteration:
            pass

    return data

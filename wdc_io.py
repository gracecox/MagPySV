# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:46:51 2016

@author: Grace
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

    try:
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
        # Old WDC format (century value is missing. col 15 = International D
        # or Q days. )

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

    # Replace missing values with NaNs
    data.replace(9999, np.nan, inplace=True)

    data = data.groupby('component').apply(wdc_io.daily_mean_conversion)
    data.reset_index(drop=True, inplace=True)

    data.drop(['base', 'daily_mean_temp'], axis=1, inplace=True)

    data = data.pivot(index='date', columns='component', values='daily_mean')
    data.reset_index(inplace=True)
    if 'D' in data.columns and 'H' in data.columns:
        data = wdc_io.angles_to_geographic(data)
        data = data[['date', 'X', 'Y', 'Z']]

    else:
        data = data[['date', 'X', 'Y', 'Z']]

    return data


def daily_mean_conversion(df):
    """Calculate the daily mean using the tabular base"""
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
    """ Use the declination and horizontal intensity to calculate X and Y"""
    data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'X'] = data.loc[(
        ~np.isnan(data['D']) & ~np.isnan(data['H'])), 'H']*np.cos(np.deg2rad(
            data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'D']))

    data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'Y'] = data.loc[(
        ~np.isnan(data['D']) & ~np.isnan(data['H'])), 'H']*np.sin(np.deg2rad(
            data.loc[(~np.isnan(data['D']) & ~np.isnan(data['H'])), 'D']))

    return data


def data_resampling(df, sampling='M'):
    """Resample the daily data to a specified frequency. Default is 'M'
    (month end) for monthly means. Another useful option is 'A' (year end)
    for annual means."""

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

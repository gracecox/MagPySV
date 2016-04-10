# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:25:47 2016

@author: Grace
"""
import pandas as pd
import datetime as dt
import numpy as np


def covobs_parsefile(fname):

    model_data = pd.read_csv(fname, sep='\s+', header=None,
                             usecols=[0, 1, 2, 3])

    model_data.columns = ["year_decimal", "dx", "dy", "dz"]


def covobs_datetimes(data):
    """Create datetime objects from the year column of a covobs output file.

    The format output by the field model is year.decimalmonth e.g. 1960.08 is
    Jan 1960

    Args:
        data (dataframe): needs a column for year (yyyy.mm format). Called by
        covobs_parsefile.

    Returns:
        data (dataframe): the same dataframe with a series of datetime objects
        (in the format yyyy-mm-dd) in the first column."""

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

    rawdata = covobs_parsefile(fname)
    data = covobs_datetimes(rawdata)

    return data

# -*- coding: utf-8 -*-
#    Copyright (C) 2016  Grace Cox (University of Liverpool)
#
#    Released under the MIT license, a copy of which is located at the root of
#    this project.
"""Module containing functions to parse files output by magnetic field models.

Part of the MagPySV package for geomagnetic data analysis. This module provides
various functions to read SV files output by geomagnetic field models.
"""


import pandas as pd
import numpy as np
import os


def calculate_sv(obs_data, mean_spacing=1):
    """Calculate the secular variation from the observed magnetic field values.

    Uses monthly means of geomagnetic observatory data to calculate the SV
    according to user-specified sampling. The typical choices are monthly
    differences of monthly means and annual differences of monthly means. For
    samplings other than monthly differences, the datetime objects of the
    calculated SV are taken the midpoint of the datetime objects of the field
    data. E.g. differencing the means of the field January 1999 and in January
    2000 yields the SV for July 1999.

    Args:
        obs_data (pandas.DataFrame): dataframe containing means (usually
            monthly) of observed geomagnetic field values.
        mean_spacing (int): the number of months separating the monthly mean
            values used to calculate the SV. Set to 1 (default) to use adjacent
            months of data (first differences of monthly means) or set to 12
            to calculate annual differences of monthly means.

    Returns:
        obs_sv (pandas.DataFrame): dataframe containing SV data.
    """
    obs_sv = pd.DataFrame()
    # Set the SV month to halfway between the MF measurement dates
    obs_sv['date'] = obs_data['date'] - pd.tseries.offsets.DateOffset(
        months=int(np.floor(mean_spacing/2)), day=1)
    if mean_spacing % 2 == 1:
        obs_sv['date'] = obs_sv['date'].apply(lambda dte: dte.replace(day=1))
    else:
        obs_sv['date'] = obs_sv['date'].apply(lambda dte: dte.replace(day=15))

    # Calculate scale required to give SV in nT/yr
    scaling_factor = 12 / mean_spacing
    obs_sv['dx'] = scaling_factor * obs_data['X'].diff(periods=mean_spacing)
    obs_sv['dy'] = scaling_factor * obs_data['Y'].diff(periods=mean_spacing)
    obs_sv['dz'] = scaling_factor * obs_data['Z'].diff(periods=mean_spacing)
    obs_sv.drop(obs_sv.head(mean_spacing).index, inplace=True)

    return obs_sv


def calculate_residuals(*, obs_data, model_data):
    """Calculate SV residuals (observed - prediction) using datetime objects.

    Args:
        obs_data (pandas.DataFrame): dataframe containing means (usually
            monthly) of SV calculated from observed geomagnetic field values.
        model_data (pandas.DataFrame): dataframe containing the SV predicted by
            a geomagnetic field model.

    Returns:
        residuals (pandas.DataFrame): dataframe containing SV residuals.
    """
    dates = obs_data['date']
    model_data = model_data[model_data['date'].isin(dates)]
    obs_data.drop(obs_data.columns[[0]], axis=1, inplace=True)
    model_data.drop(model_data.columns[[0]], axis=1, inplace=True)

    # Calculate SV residuals (data - model prediction) for all observatories
    residuals = pd.DataFrame(
        obs_data.values - model_data.values,
        columns=obs_data.columns)
    obs_data.insert(0, 'date', dates)

    return residuals


def data_resampling(data, sampling='MS', average_date=True):
    """Resample the hourly geomagnetic data to a specified frequency.

    Args:
        data (pandas.DataFrame): dataframe containing datetime objects and
            hourly means of magnetic data.
        sampling (str): new sampling frequency. Default value is 'MS'
            (monthly means), which averages data for each month and sets
            the datetime object to the first day of that month. Use 'M'
            to set the datetime object to the final day of the month.
            Another useful option is 'A' (annual means), which averages
            data for a whole year and sets the datetime object to the final
            day of the year. Use 'AS' to set the datetime object to the
            first day of the year.
        average_date (bool): the specified resampling intervals only have
            options for setting the date to the first ('MS' and 'AS') or final
            ('M' and 'A') day of the month or year. For monthly averages, a
            more appropriate representative date is the middle of that month
            (i.e. the 15th day of the month). For annual averages, an
            appropriate representative date is the middle of that year (taken
            as July 1st of the year.) This option is used to set the dates to
            the centre of the sampling interval. Defaults to True.

    Returns:
        data (pandas.DataFrame):
            dataframe of datetime objects and monthly/annual means of
            observatory data.
    """
    resampled = data.set_index('date', drop=False).resample(sampling).mean()
    resampled.reset_index(inplace=True)
    if average_date is True and 'A' in sampling:
        resampled['date'] = resampled['date'] + pd.tseries.offsets.DateOffset(
                            month=7, day=1)
    if average_date is True and 'M' in sampling:
        resampled['date'] = resampled['date'] + pd.tseries.offsets.DateOffset(
                            day=15)
    return resampled


def apply_Ap_threshold(*, Ap_file=None, obs_data, threshold):
    """Remove observatory data for times with ap values above threshold value.

    Args:
        Ap_file (str): path to file containing hourly values for the ap index.
        obs_data (pandas.DataFrame): dataframe containing hourly means of
            observed geomagnetic field values.
        threshold (int): the threshold Ap value. Data for days with a higher Ap
            value will be replaced with NaNs and omitted from monthly (or
            annual etc) means.

    Returns:
        obs_data (pandas.DataFrame): data with ap threshold applied.
    """
    col_names = ['date', 'Ap']
    Ap_hourly = pd.read_csv(Ap_file, names=col_names,
                            dtype={'date': 'str', 'Ap': 'float'},
                            parse_dates=[0], skiprows=0, index_col=None)
    # Merge the two dataframes so that only dates contained within both are
    # retained
    obs_data = pd.merge(obs_data, Ap_hourly, on='date', how='inner')
    # Use the threshold to discard data from days with high external field
    # activity
    obs_data.loc[obs_data.Ap > threshold, 'X'] = np.NaN
    obs_data.loc[obs_data.Ap > threshold, 'Y'] = np.NaN
    obs_data.loc[obs_data.Ap > threshold, 'Z'] = np.NaN
    obs_data.drop(['Ap'], axis=1, inplace=True)

    return obs_data


def remove_selected_points(*, data, fname):
    """Remove specified points from dataset based on list of points in a file.

    Reads a list of unwanted points from a file and removes them from the
    dataframe if present. E.g. If the user had monthly SV means and wished to
    exclude the X value at NGK in January 2015 from the analysis, the following
    line would be written in a file

    2015-01-01,ngk,X

    It is preferable (and more repeatable) to use the included outlier
    detection algorithms to remove outliers.

    Args:
        data (pandas.DataFrame): dataframe containing datetime objects and
              daily means of magnetic data.
        fname (str): path to file containing a list of unwanted data points.

    Returns:
        data (pandas.DataFrame): the same dataframe with the points removed.
    """
    col_names = ['date', 'observatory', 'component']
    points = pd.read_csv(fname, sep=',', header=0, names=col_names,
                         parse_dates=[0])

    for data_point in points.itertuples():
        col_name = data_point.component + '_' + data_point.observatory
        try:
            data.loc[data['date'] == data_point.date, col_name] = np.nan
        except KeyError:
            pass

    return data


def calculate_sv_index(obs_data, mean_spacing=1):
    """Calculate the secular variation of a geomagnetic index.

    Uses monthly means of a geomagnetic index (e.g. Dst) to calculate the SV
    according to user-specified sampling. The typical choices are monthly
    differences of monthly means and annual differences of monthly means. For
    samplings other than monthly differences, the datetime objects of the
    calculated SV are taken the midpoint of the datetime objects of the field
    data. E.g. differencing the means of the field January 1999 and in January
    2000 yields the SV for July 1999.

    Args:
        obs_data (pandas.DataFrame): dataframe containing means (usually
            monthly) of observed geomagnetic field values.
        mean_spacing (int): the number of months separating the monthly mean
            values used to calculate the SV. Set to 1 (default) to use adjacent
            months of data (first differences of monthly means) or set to 12 to
            calculate annual differences of monthly means.

    Returns:
        obs_sv (pandas.DataFrame): dataframe containing SV data.
    """
    obs_sv = pd.DataFrame()
    # Set the SV month to halfway between the MF measurement dates
    obs_sv['date'] = obs_data['date'] - pd.tseries.offsets.DateOffset(
        months=int(np.floor(mean_spacing/2)), day=1)
    if mean_spacing % 2 == 1:
        obs_sv['date'] = obs_sv['date'].apply(lambda x: x.replace(day=1))
    else:
        obs_sv['date'] = obs_sv['date'].apply(lambda x: x.replace(day=15))
    # Calculate scale required to give SV in nT/yr
    scaling_factor = 12 / mean_spacing
    obs_sv['mean'] = scaling_factor * obs_data['mean'].diff(
            periods=mean_spacing)
    obs_sv.drop(obs_sv.head(mean_spacing).index, inplace=True)

    return obs_sv


def calculate_correlation_index(*, dates, signal, index_file):
    """Calculate correlation coefficient between signal and a geomagnetic index

    Args:
        dates (datetime.datetime): dates of time series measurements.
        signal (time series): data to be compared with geomagnetic index.
        index_file (str): path to file containing geomagnetic index.

    Returns:
        (tuple): tuple containing:

        - coeff (*float*):
            correlation coefficient.
        - merged (*pandas.DataFrame*):
            dataframe containing dates, signal and the geomagnetic index.
    """
    col_names = ['date', 'index_vals']
    index_df = pd.read_csv(index_file, names=col_names,
                           dtype={'date': 'str', 'index_vals': 'float'},
                           parse_dates=[0], skiprows=1, index_col=None)
    df = pd.DataFrame({'date': dates, 'proxy': signal})
    # Merge the two dataframes so that only dates contained within both are
    # retained
    merged = pd.merge(df.dropna(), index_df.dropna(), on='date', how='inner')
    coeff = np.corrcoef(merged.index_vals, merged.proxy)

    return np.abs(coeff.data[0, 1]), merged


def get_baseline_info(*, fname=None):
    """Read documented baseline changes from a file.
    Args:
        fname (str): location of file containing documented baseline changes.

    Returns:
        data (pandas.DataFrame): baseline change data.
    """
    if fname is None:
        fname = os.path.join(os.path.dirname(__file__), 'baseline_records')

    col_names = ['observatory', 'jump_year', 'x_jump', 'y_jump', 'z_jump']
    data = pd.read_csv(fname, sep=',', names=col_names)
    data['jump_year'] = pd.to_datetime(dict(year=data['jump_year'],
                                       month=1, day=1))
    return data


def correct_baseline_change(*, observatory, field_data, baseline_data,
                            print_data):
    """Correct documented baseline changes.

    Args:
        observatory (str): observatory name given as a 3-digit IAGA code.
        field_data (pandas.DataFrame): uncorrected magnetic field data.
        baseline_data (pandas.DataFrame): baseline discontinuity data in the
            format output by get_baseline_info.
        print_data (bool): option to print the corrections made.
    """
    # Extract baseline data for the specified observatory
    obs_jumps = baseline_data.loc[
        baseline_data['observatory'] == str.upper(observatory)]

    if print_data is True:
        print(obs_jumps)
    # Loop over a baseline changes and make the correction
    for jump in obs_jumps.index:
        # Check if a change of zero is recorded
        if (obs_jumps.loc[jump].x_jump == 0 & obs_jumps.loc[jump].y_jump == 0 &
                obs_jumps.loc[jump].z_jump == 0):
            print("Baseline change of unknown magnitude: ",
                  obs_jumps.loc[jump].jump_year)
        else:
            field_data.loc[field_data['date'] < obs_jumps.loc[jump].jump_year,
                           'X'] = field_data.loc[
                field_data['date'] < obs_jumps.loc[jump].jump_year, 'X'] - \
                obs_jumps.loc[jump].x_jump
            field_data.loc[field_data['date'] < obs_jumps.loc[jump].jump_year,
                           'Y'] = field_data.loc[
                field_data['date'] < obs_jumps.loc[jump].jump_year, 'Y'] - \
                obs_jumps.loc[jump].y_jump
            field_data.loc[field_data['date'] < obs_jumps.loc[jump].jump_year,
                           'Z'] = field_data.loc[
                field_data['date'] < obs_jumps.loc[jump].jump_year, 'Z'] - \
                obs_jumps.loc[jump].z_jump

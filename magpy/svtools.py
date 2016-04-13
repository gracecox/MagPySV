# -*- coding: utf-8 -*-
"""Module containing functions to parse files output by magnetic field models.

Part of the MagPy package for geomagnetic data analysis. This module provides
various functions to read SV files output by geomagnetic field models.

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

import pandas as pd


def calculate_sv(obs_data, mean_spacing=1):
    """Calculate the secular variation from the observed magnetic field values.

    Uses monthly means of geomagnetic observatory data to calculate the SV
    according to user-specified sampling. The typical choices are monthly
    differences of monthly means and annual differences of monthly means. For
    samplings other than monthly differences, the datetime objects of the
    calculated SV are taken the midpoint of the datetime objects of the field
    data. E.g. differencing the means of the field January 1999 and in January
    2000 yields the SV for August 1999.

    Args:
        obs_data (pandas.DataFrame): dataframe containing means (usually
            monthly) of observed geomagnetic field values.
        mean_spacing (int): the number of months separating the monthly mean
            values used to calculate the SV. Set to 1 to use adjacent months of
            data (monthly differences of monthly means) or set to 12 to
            calculate annual differences of monthly means.

    Returns:
        obs_data (pandas.DataFrame): dataframe containing SV data.
    """

    # write function to calculate sv here
    obs_sv = pd.DataFrame()
    obs_sv['date'] = obs_data['date'] - pd.tseries.offsets.DateOffset(
        months=mean_spacing - 1)
    obs_sv['dx'] = obs_data['X'].diff(periods=mean_spacing)
    obs_sv['dy'] = obs_data['Y'].diff(periods=mean_spacing)
    obs_sv['dz'] = obs_data['Z'].diff(periods=mean_spacing)
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

    model_data = model_data[model_data['date'].isin(obs_data['date'])]
    obs_data.drop(obs_data.columns[[0]], axis=1, inplace=True)
    model_data.drop(model_data.columns[[0]], axis=1, inplace=True)

    # Calculate SV residuals (data - model prediction) for all observatories
    residuals = pd.DataFrame(
        obs_data.values - model_data.values,
        columns=obs_data.columns)
    return residuals

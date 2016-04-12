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

    # write function to calculate sv here
    obs_sv = pd.DataFrame()
    obs_sv['date'] = obs_data['date'] - pd.tseries.offsets.DateOffset(
        months=mean_spacing-1)
    obs_sv['dx'] = obs_data['X'].diff(periods=mean_spacing)
    obs_sv['dy'] = obs_data['Y'].diff(periods=mean_spacing)
    obs_sv['dz'] = obs_data['Z'].diff(periods=mean_spacing)
    obs_sv.drop(obs_sv.head(mean_spacing).index, inplace=True)
    return obs_sv


def calculate_residuals(obs_data, model_data):
    """ Calculate residuals between model and observatory data.

    Use datetime objects to ensure that we have both an observatory SV value
    and a model prediction for each date in the analysis, and that these values
    are located in the same position within their respective dataframes """

    model_data = model_data[model_data['date'].isin(obs_data['date'])]
    obs_data.drop(obs_data.columns[[0]], axis=1, inplace=True)
    model_data.drop(model_data.columns[[0]], axis=1, inplace=True)

    # Calculate SV residuals (data - model prediction) for all observatories
    residuals = pd.DataFrame(
                            obs_data.values-model_data.values,
                            columns=obs_data.columns)
    return residuals

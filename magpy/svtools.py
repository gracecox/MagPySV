# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:40:59 2016

@author: Grace
"""

import pandas as pd
import glob
import wdc_io


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

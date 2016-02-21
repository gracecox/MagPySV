# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:29:31 2016

@author: Grace
"""

import numpy as np
import datetime as dt
import pandas as pd

def load_monthly_means(obs_names,model_name):
    
    """ Load SV values and calculate the residuals
    
    Loads secular variation data for the specified
    observatories from file. Also loads the SV predicted at the same locations 
    by the specified field model and calculates the SV residuals 
    (model - prediction).
    
    Args: 
        obs_names: List of observatory names (as 3-digit IAGA codes)
        model_name: Name of field model used to calculate the residuals
         
    Returns:
        A dataframe containing the residuals between the monthly means data and
        model values. First column is a list of datetime objects (in the 
        format yyyy-mm-01. The 01 is spcified because the datetime objects need
        a day). Subsequent columns contain the SV residuals in x, y and z for
        each observatory.              
    """
    
    for observatory in obs_names:
        
        # Load the SV data for a single observatory
        data_filename = (
                        '~/Dropbox/BGS_data/monthly_means/data/sv_ap_%s.txt'
                        % observatory)
                        
        obs_data_temp = pd.read_csv(data_filename,sep='\s+',header=None)
        
        obs_data_temp.columns = ["year", "month", "dx", "dy", "dz"]
        
        # Create datetime objects from the year and month columns of the data file
        dates = obs_data_temp.apply(
                                   lambda x:dt.datetime.strptime(
                                   "{0} {1}".format(int(x['year']),
                                   int(x['month'])), "%Y %m"),axis=1)
                                   
        obs_data_temp.insert(0, 'date', dates)
        obs_data_temp.drop(obs_data_temp.columns[[1, 2]], axis=1, inplace=True)
      
        # Load the SV predicted by the COV-OBS model at the same observatory
        model_filename = (
                         '~/Dropbox/cov-obs_x1/monthly_vals/sv_%s.dat'
                         % observatory)
                         
        model_data_temp = pd.read_csv(model_filename,sep='\s+',
                                      header=None,usecols=[0,1,2,3])
        model_data_temp.columns = ["year_decimal", "dx", "dy", "dz"]
        
        # Create datetime objects from the year column of the data file. The format
        # output by the field model is year.decimalmonth e.g. 1960.08 is Jan 1960
        year_temp = np.floor(
                         model_data_temp.year_decimal.values.astype(
                         'float64')).astype('int')
                         
        months = (
                 12*(model_data_temp.year_decimal - year_temp) + 1).round(
                 ).astype('int')
                 
        model_data_temp.drop(model_data_temp.columns[[0]], axis=1, inplace=True)
        model_data_temp.insert(0, 'year', year_temp)
        model_data_temp.insert(1, 'month', months)
        
        date = model_data_temp.apply(
                                    lambda x:dt.datetime.strptime(
                                    "{0} {1}".format(int(x['year']),
                                    int(x['month'])), "%Y %m"),axis=1)
          
        model_data_temp.insert(0, 'date', date)
        model_data_temp.drop(model_data_temp.columns[[1, 2]], axis=1, inplace=True)
        
        # Combine the current observatory data with those of other observatories
        if observatory == obs_names[0]:
            obs_data = obs_data_temp
            model_data = model_data_temp
        else:
            obs_data = pd.merge(
                               left=obs_data, right=obs_data_temp,
                               how='left', on='date', suffixes=obs_names)
            model_data = pd.merge(
                                 left=model_data,right=model_data_temp,
                                 how='left', on='date', suffixes=obs_names)
    
    # Use datetime objects to ensure that we have both an observatory SV value and 
    # a model prediction for each date in the analysis, and that these values are
    # located in the same position within their respective dataframes
      
    model_data=model_data[model_data['date'].isin(obs_data['date'])]    
    obs_data.drop(obs_data.columns[[0]], axis=1, inplace=True)
    model_data.drop(model_data.columns[[0]], axis=1, inplace=True)
    
    # Calculate the SV residuals (data - model prediction) for all observatories
    residuals = pd.DataFrame(
                            obs_data.values-model_data.values,
                            columns=obs_data.columns)
    return residuals
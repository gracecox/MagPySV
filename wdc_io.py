# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:46:51 2016

@author: Grace
"""
# Need functions to:
# 1. Convert wdc hourly data into monthly means, remove stormy days, save to file
# 2. Get monthly or annual differences of the monthly means (SV)

import glob    
import datetime as dt
import pandas as pd
import numpy as np

obs_names = ['aqu'] #['aqu','clf','hrb','ngk']
model_name = 'cov-obs'

def wdc_readfile(fname):
    
    try:
        cols = [(0,3),(3,5),(5,7),(7,8),(8,10),(14,16),(16,20),(116,120)]
        col_names = [
                    'code','yr','month','component','day','century','base',
                    'daily_mean_temp']
        types = {
                'code': str,'year':int,'month':int,'component':str,
                'day':int,'century':int,'base':int,'daily_mean':float}
        data = pd.read_fwf(fname,colspecs=cols,names=col_names,converters=types,
                           header=None)
    except ValueError:
        # Old WDC format (century value is missing. col 15= International D or Q days. )
        cols = [(0,3),(3,5),(5,7),(7,8),(8,10),(16,20),(116,120)]
        col_names = [
                    'code','yr','month','component','day','base',
                    'daily_mean_temp']
        types = {
                'code': str,'year':int,'month':int,'component':str,
                'day':int,'base':int,'daily_mean':float}
        data = pd.read_fwf(fname,colspecs=cols,names=col_names,converters=types,
                           header=None)
        data['century'] = 19
        
    return data
                           
def wdc_datetimes(data):
    
    # Convert the century/yr columns to a year
    data['year'] = 100*data['century'] + data['yr']
       
    # Create datetime objects from the century, year, month and day columns of
    # the WDC format data file
    dates = data.apply(
                      lambda x:dt.datetime.strptime(
                      "{0} {1} {2}".format(x['year'],
                      x['month'],x['day']), "%Y %m %d"),axis=1)
    data.insert(0, 'date', dates)
    return data
    
def data_averaging(data, sampling=True):
    
    #Replace missing values with NaNs
    data.replace(9999, np.nan, inplace=True)
    
    data['daily_mean']=data.apply(daily_mean_conversion,axis=1)
    
    data.drop(data.columns[[2, 3, 5, 6, 7, 8, 9]], axis=1, inplace=True)
    
    data = data.pivot(index='date', columns='component', values='daily_mean')
    
    if 'D' in data.columns and 'H' in data.columns:
        data = data.merge(
                          data.apply(angles_to_geographic, axis=1),
                          left_index=True, right_index=True)
        data.reset_index(inplace=True)
        data.drop(data.columns[[1,2]], axis=1, inplace=True)
        
    else:
        data.reset_index(inplace=True)
        
    """Resample the daily data to a specified frequency. Default is 'M' (month end)
       for monthly means. Another useful option is 'A' (year end) for annual means."""
        
    data = data.set_index(['date']).resample(sampling,how='mean') 
    data.reset_index(inplace=True)
    
    return data.reindex_axis(['date','X','Y','Z'],axis=1) 


def daily_mean_conversion(row):
     """Calculate the daily mean using the tabular base"""
     
     if np.isnan(row['daily_mean_temp']): 
         return np.nan
     elif row['component'] == 'D' or row['component']=='I':
         return row['base'] + (1/600.0)*row['daily_mean_temp']
     else:
         return 100*row['base'] + row['daily_mean_temp']
    
def angles_to_geographic(row):
    """ Use the declination and horizontal intensity to calculate X and Y"""
    if np.isnan(row['H']) or np.isnan(row['D']):
        x = np.nan
        y = np.nan
        
    else:
    
        x = row['H']*np.cos(np.deg2rad(row['D']))
        y = row['H']*np.sin(np.deg2rad(row['D']))
      
    return pd.Series({'X':x, 'Y':y})                
                      
#def data_resampling(df, sampling='M'):
#    """Resample the daily data to a specified frequency. Default is 'M' (month end)
#       for monthly means. Another useful option is 'A' (year end) for annual means."""
#      
#    resampled = df.set_index('date', drop=False).resample(sampling,how='mean')  
#                 
#    return resampled                 
#                      
##for observatory in obs_names:
##    path = '/Users/Grace/Dropbox/BGS_hourly/hourval/single_obs/%s/*.wdc' % observatory
##    print(path)
##    filenames = glob.glob(path)                      
#     for f in filenames:
#        try:
#            frame = wdc_to_dataframe(f)
#            df = df.append(frame,ignore_index=True)
#        except StopIteration:
#            pass                     
#                      
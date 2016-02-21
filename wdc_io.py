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

obs_names = ['aqu'] #['aqu','clf','hrb','ngk']
model_name = 'cov-obs'




def wdc_to_dataframe(f):    
    cols = [(0,3),(3,5),(5,7),(7,8),(8,10),(14,16),(16,20),(116,120)]
    col_names = [
                'code','yr','month','element','day','century','base',
                'daily_mean_temp']
    types = {
            'code': str,'year':int,'month':int,'element':str,
            'day':int,'century':int,'base':int,'daily_mean':int}
    data=pd.read_fwf(f,colspecs=cols,names=col_names,converters=types,header=None)
    
    # Convert the century/yr columns to a year
    data['year'] = 100*data['century'] + data['yr']
    
    # Calculate the daily mean using the tabular base
    data['daily_mean'] = (100*data['base'] + data['daily_mean_temp']).astype(float)
    
    # Create datetime objects from the century, year, month and day columns of
    # the WDC format data file
    dates = data.apply(
                      lambda x:dt.datetime.strptime(
                      "{0} {1} {2}".format(x['year'],
                      x['month'],x['day']), "%Y %m %d"),axis=1)
    data.insert(0, 'date', dates)
    
    data.drop(data.columns[[2, 3, 5, 6, 7, 8, 9]], axis=1, inplace=True)
    return data                      
                      
def daily_to_monthly(df):
    monthly = df.set_index('date').groupby('element')['daily_mean'].resample('M',how='mean')
    monthly.rename(columns={'daily_mean':'monthly_mean'}, inplace=True)
    monthly.reset_index(inplace=True)                      
    monthly.columns = ['date', 'element', 'monthly_mean']
    return monthly                 
                      
for observatory in obs_names:
    path = '/Users/Grace/Dropbox/BGS_hourly/hourval/single_obs/%s/*.wdc' % observatory
    print(path)
    filenames = glob.glob(path)
    df = pd.concat((wdc_to_dataframe(f) for f in filenames))                      
                      
                      
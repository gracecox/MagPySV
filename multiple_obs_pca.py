# -*- coding: utf-8 -*-
"""
Script to perform principal component analysis (PCA) on secular variation 
residuals (the difference between the observed SV and that predicted by a 
geomagnetic field model) calculated from annual differences of monthly 
means at several observatories. Uses the imputer from sklearn.preprocessing to 
fill in missing data points and calculates the eigenvalues/vectors of the
(3nx3n) covariance matrix for n observatories. The residuals are rotated
into the eigendirections and denoised using the method detailed in
Wardinski & Holme (2011). The SV residuals of the noisy component for all 
observatories combined are used as a proxy for the unmodelled external signal. 
The denoised data are then rotated back into geographic coordinates and saved
to file.
Author: Grace Cox
Date written: 09/01/16

@author: Grace
"""
import numpy as np
import scipy as sp
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import Imputer
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
cpm = importr('cpm')

obs_names = ['aqu','clf'] #['aqu','clf','hrb','ngk']
model_name = 'cov-obs'
obs_no = len(obs_names)

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
                        
# Fill in missing SV values (indicated as NaN in the data files)                     
imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
imputed_residuals = imp.fit_transform(residuals)

# Principal component analysis - find the eigenvalues and eigenvectors of
# the covariance matrix of the residuals. Project the SV residuals into the 
# eigenvector directions. The pca algorithm outputs the eigenvalues sorted from
# largest to smallest, so the corresponding eigenvector matrix has the 'noisy'
# direction in the first column and the 'clean' direction in the final column 
# Smallest eigenvalue: 'quiet' direction
# Largest eiegenvalue: 'noisy' direction 

proxy_number = 1;  # How many eigendirections to use as the proxy?
pca = sklearnPCA()
projected_residuals = pca.fit_transform(imputed_residuals)
eig_values = pca.explained_variance_ 
eig_vectors = pca.components_

# Use the method of Wardinski & Holme (2011) to remove unmodelled external
# signal in the SV residuals. The variable 'proxy' contains the noisy
# component residual for all observatories combined
noisy_direction = eig_vectors[0,:]
proxy = projected_residuals[:,0]

if proxy_number>1:
    for direction in range(proxy_number):
        proxy = proxy + projected_residuals[:,direction]
    

corrected_residuals = []

for idx in range(len(proxy)):
    corrected_residuals.append(imputed_residuals[idx,:] - proxy[idx]*noisy_direction)
    
corrected_residuals = pd.DataFrame(corrected_residuals,columns=obs_data.columns)
denoised_sv = pd.DataFrame(
                        corrected_residuals.values+model_data.values,
                        columns=obs_data.columns)
denoised_sv.insert(0, 'date', dates)                        
                        
                        
def plot_eigenvalues(values):
    plt.figure(figsize=(8, 6))
    plt.plot(values)
    plt.axis('tight')
    plt.xlabel('$i$', fontsize=16)
    plt.ylabel('$\lambda_i$', fontsize=16)
        
def plot_denoised(dates,denoised,model):
    
    plt.figure(figsize=(8, 6))
    plt.subplot(3,1,1)
    plt.gca().xaxis_date()
    plt.plot(
             dates,denoised[:,0],'b',
             dates,model[:,0],'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.ylabel('$\dot{x}$ (nT/yr)', fontsize=16)
    plt.subplot(3,1,2)
    plt.gca().xaxis_date()
    plt.plot(
             dates,denoised[:,1],'b',
             dates,model[:,1],'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.ylabel('$\dot{y}$ (nT/yr)', fontsize=16)
    plt.subplot(3,1,3)
    plt.gca().xaxis_date()
    plt.plot(
             dates,denoised[:,2],'b',
             dates,model[:,2],'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('$\dot{z}$ (nT/yr)', fontsize=16)
    #plt.legend(['denoised','cov-obs'],loc='upper right',frameon=False)

def plot_dcx(date,signal):
    dcx = pd.read_csv(
                     '~/Dropbox/BGS_data/monthly_means/Dcx/Dcx_mm_monthly_diff.txt',
                     sep='\s+',header=None)
    dcx.columns = ["year", "month", "monthly_mean"]
    dates = dcx.apply(
                     lambda x:dt.datetime.strptime(
                     "{0} {1}".format(int(x['year']),
                     int(x['month'])), "%Y %m"),axis=1)
                               
    dcx.insert(0, 'date', dates)
    dcx.drop(dcx.columns[[1, 2]], axis=1, inplace=True)
    plt.figure(figsize=(8, 6))
    plt.gca().xaxis_date()
    plt.plot(
             dcx.date,sp.stats.mstats.zscore(dcx.monthly_mean),'b',
             date,sp.stats.mstats.zscore(signal),'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Dcx (nT/yr)', fontsize=16)
    plt.legend(['Dcx','proxy'],loc='upper right',frameon=False)

def plot_dcx_fft(dates,signal):
    dcx = pd.read_csv(
                     '~/Dropbox/BGS_data/monthly_means/Dcx/Dcx_mm_monthly_diff.txt',
                     sep='\s+',header=None)
    dcx.columns = ["year", "month", "monthly_mean"]
    dates = dcx.apply(
                     lambda x:dt.datetime.strptime(
                     "{0} {1}".format(int(x['year']),
                     int(x['month'])), "%Y %m"),axis=1)
                               
    dcx.insert(0, 'date', dates)
    dcx.drop(dcx.columns[[1, 2]], axis=1, inplace=True)
    
    T = 1/12.0   # Sampling time in years
    
    # Find the next power of two higher than the length of the time series and
    # perform the FFT with the series padded with zeroes to this length
    N = int(pow(2,np.ceil(np.log2(len(proxy)))))
    
    dcx_fft = sp.fft(dcx.monthly_mean, N)
    proxy_fft = sp.fft(signal, N)
    freq = np.linspace(0.0, 1.0/(2.0*T), N/2)
    dcx_power = 2.0/N * np.abs(dcx_fft[:N/2])
    proxy_power = 2.0/N * np.abs(proxy_fft[:N/2])
    
    plt.figure(figsize=(10, 7))
    plt.subplot(2,1,1)
    plt.gca().xaxis_date()
    plt.plot(dcx.date,dcx.monthly_mean,'b',
             dates,signal,'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Dcx (nT/yr)', fontsize=16)
    plt.subplot(2,1,2)
    plt.plot(freq,dcx_power,'b',
             freq,proxy_power,'r')
    plt.xlabel('Frequency (cycles/year)', fontsize=16)
    plt.ylabel('Power', fontsize=16)
    plt.legend(['Dcx','proxy'],loc='upper right',frameon=False)
    

def change_point_analysis(signal,cpm_method):
    res = cpm.detectChangePoint(
                                ro.FloatVector(signal),cpmType=cpm_method,
                                ARL0=500,startup=20)
    # Convert the ListVector returned by cpm to a python dictionary                             
    results = { key : res.rx2(key)[0] for key in res.names }        
        
    return results
 
def plot_cpa_results(dates,signal,results):   
    if results['changeDetected']:
        plt.figure(figsize=(10, 7))
        plt.gca().xaxis_date()
        plt.plot(dates,signal,'b')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xlabel('Year', fontsize=16)
        plt.ylabel('SV (nT/yr)', fontsize=16)
        # Draw vertical lines at each change point location
        if isinstance(results['changePoint'], int):     # Only 1 change point
            plt.axvline(dates[results['changePoint']],color='r',linestyle='--')
        else:
            for chgpoint in results['changePoint']:  # More than 1 change point
                plt.axvline(dates[chgpoint],color='r',linestyle='--')





                  
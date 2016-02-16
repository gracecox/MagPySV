# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:04:47 2016

@author: Grace
"""
import numpy as np
import scipy as sp
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
                        
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
    N = int(pow(2,np.ceil(np.log2(len(signal)))))
    
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
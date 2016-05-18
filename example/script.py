# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:01:58 2016

@author: Grace
"""


import magpy.changepoint as changepoint
import magpy.denoise as denoise
import magpy.inputoutput as inputoutput
import magpy.svplots as svplots
import magpy.svtools as svtools

# List of observatories to include
obs_names = ['aqu', 'had']

# Extract data from WDC files and save to CSV files
for observatory in obs_names:
    field_data = inputoutput.append_wdc_data(observatory)
    inputoutput.write_csv_data(field_data, "./data", observatory)

# Concatenate magnetic field data for these observatories
obs_data, model_data = inputoutput.combine_csv_data(obs_list=obs_names,
                data_path="./data/",
                model_path="Users/Grace/Dropbox/cov-obs_x1/monthly_vals/")

# Plot the SV

# Calculate the SV residuals

# Use principal component analysis to remove correlated external signal (noise)

# Compare the signal used for noise removal with Dst index

# Plot the denoised SV

# Identify jumps in the time series using changepoint analysis

# Plot the CPA results

# Plot the final SV series

# Save the data to CSV files
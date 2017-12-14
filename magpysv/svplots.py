# -*- coding: utf-8 -*-
#    Copyright (C) 2016  Grace Cox (University of Liverpool)
#
#    Released under the MIT license, a copy of which is located at the root of
#    this project.
"""Module containing plotting functions.

Part of the MagPySV package for geomagnetic data analysis. This module provides
various plotting functions.
"""


import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sp

# Setup matplotlib to use latex fonts in figure labels if needed
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                       r'\usepackage{helvet}',
                                       r'\usepackage{sansmath}',
                                       r'\sansmath']


def plot_eigenvalues(*, values, fig_size=(8, 6), font_size=12, label_size=16):
    """Plot eigenvalues of the covariance matrix of SV residuals.

    Produces a plot of the eigenvalues obtained during the principal component
    analysis (PCA) of SV residuals. The largest eigenvalue represents the
    eigendirection with the largest contribution to the residuals (i.e. the
    "noisy" direction.). The smallest eigenvalue represents the
    eigendirection with the smallest contribution to the residuals (the "clean"
    direction). See Wardinski & Holme (2011, GJI) for further details.

    Args:
        values (array): the eigenvalues obtained from the principal component
            analysis of the SV residuals.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
    """
    plt.figure(figsize=fig_size)
    plt.plot(values, 'bx-')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
#    plt.axis('tight')
    plt.xlabel(r'$i$', fontsize=label_size)
    plt.ylabel(r'$\lambda_i$', fontsize=label_size)


def plot_eigenvectors(*, obs_names, eigenvecs, fig_size=(8, 6), font_size=12,
                      label_size=16):
    """Plot eigenvectors of the covariance matrix of SV residuals.

    Produces a plot of the eigenvectors corresponding to the n largest
    eigenvalues of the covariance matrix obtained during PCA of SV residuals,
    where n is the number of eigenvalues used as a proxy for unmodelled
    external field signal. The n eigenvectors corresponding to the n largest
    eigenvalue represent the directions with the largest contribution
    to the residuals (i.e. the "noisiest" directions). See Wardinski & Holme
    (2011, GJI) for further details.

    Args:
        obs_names (list): list of observatory names given as three digit IAGA
            codes.
        eigenvecs (array): the eigenvalues obtained from the principal
        component analysis of the SV residuals.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
    """
    plt.figure(figsize=fig_size)
    # Loop over directions and plot each eigenvector on a separate subplot
    for direction in range(eigenvecs.shape[1]):
        plt.subplot(eigenvecs.shape[1], 1, direction + 1)
        plt.plot(np.abs(eigenvecs[::3, direction]), 'bx',
                 np.abs(eigenvecs[1::3, direction]), 'rx',
                 np.abs(eigenvecs[2::3, direction]), 'cx')
        plt.ylim(0, 1)
        plt.yticks(fontsize=font_size)
        plt.xticks(range(len(obs_names)), obs_names, fontsize=font_size)
        plt.ylabel(r'$\mathbf{{v}}_{}$'.format(direction), fontsize=label_size)
    plt.legend(['x direction', 'y direction', 'z direction'],
               loc='upper right', frameon=False)
    plt.xlabel('Location', fontsize=label_size)


def plot_mf(*, dates, mf, model, obs, fig_size=(8, 6), font_size=12,
            label_size=16, plot_legend=True):
    """Plot the SV and model prediction for a single observatory.

    Produces a plot of the X, Y and Z components of the SV and field
    model prediction for a single observatory.

    Args:
        dates (datetime.datetime): dates of time series measurements.
        mf (time series): X, Y and Z components of magnetic field at a single
            location.
        model (time series): X, Y and Z components of the field predicted by a
            field model for the same location as the data.
        obs (str): observatory name given as three digit IAGA code.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        plot_legend (bool): option to include a legend on the plot. Defaults
            to True.
    """
    plt.figure(figsize=fig_size)
    # X component
    plt.subplot(3, 1, 1)
    plt.gca().xaxis_date()
    plt.plot(dates, mf.ix[:, 0], 'b', dates, model.ix[:, 0], 'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylabel(r'$X$ (nT)', fontsize=label_size)
    # Y component
    plt.subplot(3, 1, 2)
    plt.gca().xaxis_date()
    plt.plot(dates, mf.ix[:, 1], 'b', dates, model.ix[:, 1], 'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylabel(r'$Y$ (nT)', fontsize=label_size)
    # Z component
    plt.subplot(3, 1, 3)
    plt.gca().xaxis_date()
    plt.plot(dates, mf.ix[:, 2], 'b', dates, model.ix[:, 2], 'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel('Year', fontsize=label_size)
    plt.ylabel(r'$Z$ (nT)', fontsize=label_size)
    if plot_legend is True:
        plt.legend([obs, 'COV-OBS'], loc='upper right', frameon=False)


def plot_sv(*, dates, sv, model, obs, fig_size=(8, 6), font_size=12,
            label_size=16, plot_legend=False, plot_average=False,
            window_length=12, min_samples=3):
    """Plot the SV and model prediction for a single observatory.

    Produces a plot of the X, Y and Z components of the SV and field
    model prediction for a single observatory.

    Args:
        dates (datetime.datetime): dates of time series measurements.
        sv (time series): X, Y and Z components of SV at a single location.
        model (time series): X, Y and Z components of the SV predicted by a
            field model for the same location as the data.
        obs (str): observatory name given as three digit IAGA code.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        plot_legend (bool): option to include a legend on the plot. Defaults
            to False.
        plot_average (bool): option to include a running average of the SV time
            series on the plot. Defaults to False.
        window_length (int): number of months over which to take the running
            average if this is plotted. Defaults to 12 months.
        min_samples (int): minimum number of non-NaN values that must be
            present in the window in order for the running average to be
            calculated rather than set to NaN. Defaults to 3 (e.g. for monthly
            first differences this means that at least 3 months of data per
            window are required to calculate the 12-month running average.)
    """
    if plot_average is True:
        plt.figure(figsize=fig_size)
        # X component
        plt.subplot(3, 1, 1)
        plt.gca().xaxis_date()
        plt.plot(dates, sv.ix[:, 0], 'b', dates, sv.ix[:, 0].rolling(
            window=window_length, center=True, min_periods=min_samples).mean(),
            'c', dates, model.ix[:, 0], 'r')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.ylabel(r'$\dot{x}$ (nT/yr)', fontsize=label_size)
        # Y component
        plt.subplot(3, 1, 2)
        plt.gca().xaxis_date()
        plt.plot(dates, sv.ix[:, 1], 'b', dates, sv.ix[:, 1].rolling(
            window=window_length, center=True, min_periods=min_samples).mean(),
            'c', dates, model.ix[:, 1], 'r')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.ylabel(r'$\dot{y}$ (nT/yr)', fontsize=label_size)
        # Z component
        plt.subplot(3, 1, 3)
        plt.gca().xaxis_date()
        plt.plot(dates, sv.ix[:, 2], 'b', dates, sv.ix[:, 2].rolling(
            window=window_length, center=True, min_periods=min_samples).mean(),
            'c', dates, model.ix[:, 2], 'r')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xlabel('Year', fontsize=label_size)
        plt.ylabel(r'$\dot{z}$ (nT/yr)', fontsize=label_size)
        if plot_legend is True:
            plt.legend([obs, 'Running average', 'COV-OBS'], loc='upper right',
                       frameon=False)
    else:
        plt.figure(figsize=fig_size)
        # X component
        plt.subplot(3, 1, 1)
        plt.gca().xaxis_date()
        plt.plot(dates, sv.ix[:, 0], 'b', dates, model.ix[:, 0], 'r')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.ylabel(r'$\dot{x}$ (nT/yr)', fontsize=label_size)
        # Y component
        plt.subplot(3, 1, 2)
        plt.gca().xaxis_date()
        plt.plot(dates, sv.ix[:, 1], 'b', dates, model.ix[:, 1], 'r')
        plt.gcf().autofmt_xdate()
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.axis('tight')
        plt.ylabel(r'$\dot{y}$ (nT/yr)', fontsize=label_size)
        # Z component
        plt.subplot(3, 1, 3)
        plt.gca().xaxis_date()
        plt.plot(dates, sv.ix[:, 2], 'b', dates, model.ix[:, 2], 'r')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xlabel('Year', fontsize=label_size)
        plt.ylabel(r'$\dot{z}$ (nT/yr)', fontsize=label_size)

        if plot_legend is True:
            plt.legend([obs, 'COV-OBS'], loc='upper right', frameon=False)


def plot_sv_comparison(*, dates, noisy_sv, denoised_sv, model, obs,
                       fig_size=(8, 6), font_size=12, label_size=16,
                       plot_legend=False, plot_average=False,
                       window_length=12, min_samples=3):
    """Plot noisy/denoised SV and model prediction for a single observatory.

    Produces a plot of the X, Y and Z components of the noisy SV, the denoised
    SV and field model prediction for a single observatory.

    Args:
        dates (datetime.datetime): dates of time series measurements.
        noisy_sv (time series): X, Y and Z components of uncorrected SV at a
            single location.
        denoised_sv (time series): X, Y and Z components of denoised SV at a
            single location.
        model (time series): X, Y and Z components of the SV predicted by a
            field model for the same location as the data.
        obs (str): observatory name given as three digit IAGA code.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        plot_legend (bool): option to include a legend on the plot. Defaults
            to False.
        plot_average (bool): option to include a running average of the SV time
            series on the plot. Defaults to False.
        window_length (int): number of months over which to take the running
            average if this is plotted. Defaults to 12 months.
        min_samples (int): minimum number of non-NaN values that must be
            present in the window in order for the running average to be
            calculated rather than set to NaN. Defaults to 3 (e.g. for monthly
            first differences this means that at least 3 months of data per
            window are required to calculate the 12-month running average.)
    """
    plt.figure(figsize=fig_size)
    if plot_average is True:
        # X component
        plt.subplot(3, 1, 1)
        plt.gca().xaxis_date()
        plt.plot(dates, noisy_sv.ix[:, 0], 'b', dates, denoised_sv.ix[:, 0],
                 'r', dates, denoised_sv.ix[:, 0].rolling(window=window_length,
                 center=True, min_periods=min_samples).mean(), 'c',
                 dates, model.ix[:, 0], 'k')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.ylabel(r'$\dot{x}$ (nT/yr)', fontsize=label_size)
        # Y component
        plt.subplot(3, 1, 2)
        plt.gca().xaxis_date()
        plt.plot(dates, noisy_sv.ix[:, 1], 'b', dates, denoised_sv.ix[:, 1],
                 'r', dates, denoised_sv.ix[:, 1].rolling(window=window_length,
                 center=True, min_periods=min_samples).mean(), 'c',
                 dates, model.ix[:, 1], 'k')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.ylabel(r'$\dot{y}$ (nT/yr)', fontsize=label_size)
        # Z component
        plt.subplot(3, 1, 3)
        plt.gca().xaxis_date()
        plt.plot(dates, noisy_sv.ix[:, 2], 'b', dates, denoised_sv.ix[:, 2],
                 'r', dates, denoised_sv.ix[:, 2].rolling(window=window_length,
                 center=True, min_periods=min_samples).mean(), 'c',
                 dates, model.ix[:, 2], 'k')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xlabel('Year', fontsize=label_size)
        plt.ylabel(r'$\dot{z}$ (nT/yr)', fontsize=label_size)

        if plot_legend is True:
            plt.legend([obs, 'Denoised SV', 'Running average', 'COV-OBS'],
                       loc='upper right', frameon=False)
    else:
        # X component
        plt.subplot(3, 1, 1)
        plt.gca().xaxis_date()
        plt.plot(dates, noisy_sv.ix[:, 0], 'b', dates, denoised_sv.ix[:, 0],
                 'r', dates, model.ix[:, 0], 'k')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.ylabel(r'$\dot{x}$ (nT/yr)', fontsize=label_size)
        # Y component
        plt.subplot(3, 1, 2)
        plt.gca().xaxis_date()
        plt.plot(dates, noisy_sv.ix[:, 1], 'b', dates, denoised_sv.ix[:, 1],
                 'r', dates, model.ix[:, 1], 'k')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.ylabel(r'$\dot{y}$ (nT/yr)', fontsize=label_size)
        # Z component
        plt.subplot(3, 1, 3)
        plt.gca().xaxis_date()
        plt.plot(dates, noisy_sv.ix[:, 2], 'b', dates, denoised_sv.ix[:, 2],
                 'r', dates, model.ix[:, 2], 'k')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xlabel('Year', fontsize=label_size)
        plt.ylabel(r'$\dot{z}$ (nT/yr)', fontsize=label_size)

        if plot_legend is True:
            plt.legend([obs, 'Denoised SV', 'COV-OBS'],
                       loc='upper right', frameon=False)


def plot_dcx(*, dates, signal, fig_size=(8, 6), font_size=12, label_size=16,
             plot_legend=True):
    """Compare the proxy used to denoise the SV data with the Dst index.

    Loads Dcx data (extended, corrected Dst) and plots it alongside the signal
    used as a proxy for unmodelled external signal. Both time series are
    reduced to zero mean and unit variance (i.e. their zscore) for plotting.

    Args:
        dates (datetime.datetime): dates of time series measurements.
        signal (time series): proxy for unmodelled external signal used in the
            denoising process (principal component analysis). The proxy is the
            residual in the noisiest eigendirection(s).
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        plot_legend (bool): option to include a legend on the plot. Defaults
            to True.
    """
    # Read the Dcx data and put into a dataframe
    data_path = '../../Dropbox/BGS_data/monthly_means/Dcx/'
    data_file = 'Dcx_mm_monthly_diff.txt'
    dcx = pd.read_csv(os.path.join(data_path, data_file), sep=r'\s+',
                      header=None)
    dcx.columns = ["year", "month", "monthly_mean"]
    dcx_dates = dcx.apply(lambda x: dt.datetime.strptime(
        "{0} {1}".format(int(x['year']), int(x['month'])), "%Y %m"), axis=1)

    # Create datetime objects for the series
    dcx.insert(0, 'date', dcx_dates)
    dcx.drop(dcx.columns[[1, 2]], axis=1, inplace=True)
    # Only keep Dcx data for dates during the period of interest
    dcx = dcx[dcx['date'].isin(dates)]

    # Plot the zscore of the two time series
    plt.figure(figsize=fig_size)
    plt.gca().xaxis_date()
    plt.plot(dcx.date, sp.stats.mstats.zscore(dcx.monthly_mean), 'b',
             dates, sp.stats.mstats.zscore(signal), 'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.xlabel('Year', fontsize=label_size)
    plt.ylabel('Dcx (nT/yr)', fontsize=label_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    if plot_legend is True:
        plt.legend(['Dcx', 'proxy'], loc='upper right', frameon=False)


def plot_dcx_fft(*, dates, signal, fig_size=(8, 6), font_size=12,
                 label_size=16, plot_legend=True):
    """Compare the DFTs of the proxy signal with that of the Dst index.

    Loads Dcx data (extended, corrected Dst), calculates its DFT and plots it
    alongside the DFT of the signal used as a proxy for unmodelled external
    signal. The length of the time series are padded with zeroes up to the next
    power of two.

    Args:
        dates (datetime.datetime): dates of time series measurements.
        signal (time series): proxy for unmodelled external signal used in the
            denoising process (principal component analysis). The proxy is the
            residual in the noisiest eigendirection(s).
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        plot_legend (bool): option to include a legend on the plot. Defaults
            to True.
    """
    data_path = '../../Dropbox/BGS_data/monthly_means/Dcx/'
    data_file = 'Dcx_mm_monthly_diff.txt'
    dcx = pd.read_csv(os.path.join(data_path, data_file), sep=r'\s+',
                      header=None)
    dcx.columns = ["year", "month", "monthly_mean"]
    dcx_dates = dcx.apply(lambda x: dt.datetime.strptime(
        "{0} {1}".format(int(x['year']), int(x['month'])),
        "%Y %m"), axis=1)

    dcx.insert(0, 'date', dcx_dates)
    dcx.drop(dcx.columns[[1, 2]], axis=1, inplace=True)
    # Only keep Dcx data for dates during the period of interest
    dcx = dcx[dcx['date'].isin(dates)]

    sampling_period = 1 / 12.0   # Sampling time in years

    # Find the next power of two higher than the length of the time series and
    # perform the FFT with the series padded with zeroes to this length
    sample_length = int(pow(2, np.ceil(np.log2(len(signal)))))

    dcx_fft = sp.fft(dcx.monthly_mean, sample_length)
    proxy_fft = sp.fft(signal, sample_length)
    freq = np.linspace(0.0, 1.0 / (2.0 * sampling_period), sample_length / 2)
    dcx_power = (2.0 / sample_length) * np.abs(dcx_fft[:sample_length // 2])
    proxy_power = (2.0 / sample_length) * np.abs(
        proxy_fft[:sample_length // 2])

    plt.figure(figsize=fig_size)
    # Time domain
    plt.subplot(2, 1, 1)
    plt.gca().xaxis_date()
    plt.plot(dcx.date, dcx.monthly_mean, 'b', dates, signal, 'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylabel('Dcx (nT/yr)', fontsize=label_size)
    # Frequency domain
    plt.subplot(2, 1, 2)
    plt.plot(freq, dcx_power, 'b', freq, proxy_power, 'r')
    plt.axis('tight')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel('Frequency (cycles/year)', fontsize=label_size)
    plt.ylabel('Power', fontsize=label_size)
    if plot_legend is True:
        plt.legend(['Dcx', 'proxy'], loc='upper right', frameon=False)


def plot_outliers(*, dates, signal, obs_name, outliers, fig_size=(8, 6),
                  font_size=12, label_size=16):
    """Plot the SV and identified outliers.

    Args:
        dates (datetime.datetime): dates of time series measurements.
        signal (time series): single component of SV at a single location.
        obs_name (str): states the SV component and observatory name given as
            three digit IAGA code. For example, the X component at NGK would be
            x_ngk if obs_name is taken from the pandas.DataFrame containing
            SV data for all observatories combined.
        outliers (array): outliers identified by the denoise.detect_outliers
            function
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
    """
    plt.figure(figsize=fig_size)
    plt.plot(dates, signal, 'k', dates, outliers, 'r^')
    plt.axis('tight')
    plt.xlabel('Year', fontsize=label_size)
    plt.ylabel('SV', fontsize=label_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend([obs_name, 'outlier'], loc='upper right', frameon=False)

# -*- coding: utf-8 -*-
#    Copyright (C) 2016  Grace Cox
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program.  If not, see <http://www.gnu.org/licenses/>."""
"""Module containing plotting functions.

Part of the MagPy package for geomagnetic data analysis. This module provides
various plotting functions."""


import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sp


def plot_eigenvalues(*, values, fig_size=(8, 6), font_size=16, label_size=20):
    """Plot eigenvalues of the SV residuals.

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
        font_size (int): font size for axes. Defaults to 16 pt.
        label_size (int): font size for axis labels. Defaults to 20 pt.
    """

    plt.figure(figsize=fig_size)
    plt.plot(values)
    plt.axis('tight')
    plt.xlabel(r'$i$', fontsize=font_size)
    plt.ylabel(r'$\lambda_i$', fontsize=label_size)


def plot_mf(*, dates, mf, model, obs, fig_size=(8, 6), font_size=16,
            label_size=20, plot_legend=False):
    """Plot the SV and model prediction for a single observatory.

    Produces a plot of the X, Y and Z components of the SV and field
    model prediction for a single observatory.

    Args:
        dates (datetime.datetime series): dates of time series measurements.
        mf (series): X, Y and Z components of magnetic field at a single
            location.
        model (series): X, Y and Z components of the field predicted by a
            field model for the same location as the data.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 16 pt.
        label_size (int): font size for axis labels. Defaults to 20 pt.
        plot_legend (bool): option to include a legend on the plot. Defaults
            to False.
    """

    plt.figure(figsize=fig_size)
    # X component
    plt.subplot(3, 1, 1)
    plt.gca().xaxis_date()
    plt.plot(dates, mf.ix[:, 0], 'b', dates, model.ix[:, 0], 'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.ylabel(r'$X$ (nT)', fontsize=font_size)
    # Y component
    plt.subplot(3, 1, 2)
    plt.gca().xaxis_date()
    plt.plot(dates, mf.ix[:, 1], 'b', dates, model.ix[:, 1], 'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.ylabel(r'$Y$ (nT)', fontsize=font_size)
    # Z component
    plt.subplot(3, 1, 3)
    plt.gca().xaxis_date()
    plt.plot(dates, mf.ix[:, 2], 'b', dates, model.ix[:, 2], 'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.xlabel('Year', fontsize=label_size)
    plt.ylabel(r'$Z$ (nT)', fontsize=font_size)
    if plot_legend is True:
        plt.legend([obs, 'COV-OBS'], loc='upper right', frameon=False)


def plot_sv(*, dates, sv, model, obs, fig_size=(8, 6), font_size=16,
            label_size=20, plot_legend=False):
    """Plot the SV and model prediction for a single observatory.

    Produces a plot of the X, Y and Z components of the SV and field
    model prediction for a single observatory.

    Args:
        dates (datetime.datetime series): dates of time series measurements.
        sv (series): X, Y and Z components of SV at a single location.
        model (series): X, Y and Z components of the SV predicted by a
            field model for the same location as the data.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 16 pt.
        label_size (int): font size for axis labels. Defaults to 20 pt.
        plot_legend (bool): option to include a legend on the plot. Defaults
            to False.
    """

    plt.figure(figsize=fig_size)
    # X component
    plt.subplot(3, 1, 1)
    plt.gca().xaxis_date()
    plt.plot(dates, sv.ix[:, 0], 'b', dates, model.ix[:, 0], 'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.ylabel(r'$\dot{x}$ (nT/yr)', fontsize=font_size)
    # Y component
    plt.subplot(3, 1, 2)
    plt.gca().xaxis_date()
    plt.plot(dates, sv.ix[:, 1], 'b', dates, model.ix[:, 1], 'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.ylabel(r'$\dot{y}$ (nT/yr)', fontsize=font_size)
    # Z component
    plt.subplot(3, 1, 3)
    plt.gca().xaxis_date()
    plt.plot(dates, sv.ix[:, 2], 'b', dates, model.ix[:, 2], 'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.xlabel('Year', fontsize=label_size)
    plt.ylabel(r'$\dot{z}$ (nT/yr)', fontsize=font_size)
    if plot_legend is True:
        plt.legend([obs, 'COV-OBS'], loc='upper right', frameon=False)


def plot_dcx(*, dates, signal, fig_size=(8, 6), font_size=16, label_size=20,
             plot_legend=False):
    """Compare the proxy used to denoise the SV data with the Dst index.

    Loads Dcx data (extended, corrected Dst) and plots it alongside the signal
    used as a proxy for unmodelled external signal. Both time series are
    reduced to zero mean and unit variance (i.e. their zscore) for plotting.

    Args:
        dates (datetime.datetime series): dates of time series measurements.
        signal (series): proxy for unmodelled external signal used in the
            denoising process (principal component analysis). The proxy is the
            residual in the noisiest eigendirection(s).
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 16 pt.
        label_size (int): font size for axis labels. Defaults to 20 pt.
        plot_legend (bool): option to include a legend on the plot. Defaults
            to False.
    """

    # Read the Dcx data and put into a dataframe
    data_path = '/Users/Grace/Dropbox/BGS_data/monthly_means/Dcx/'
    data_file = 'Dcx_mm_monthly_diff.txt'
    dcx = pd.read_csv(os.path.join(data_path, data_file), sep=r'\s+',
                      header=None)
    dcx.columns = ["year", "month", "monthly_mean"]
    dates = dcx.apply(lambda x: dt.datetime.strptime(
        "{0} {1}".format(int(x['year']), int(x['month'])), "%Y %m"), axis=1)

    # Create datetime objects for the series
    dcx.insert(0, 'date', dates)
    dcx.drop(dcx.columns[[1, 2]], axis=1, inplace=True)

    # Plot the zscore of the two time series
    plt.figure(figsize=fig_size)
    plt.gca().xaxis_date()
    plt.plot(dcx.date, sp.stats.mstats.zscore(dcx.monthly_mean), 'b',
             dates, sp.stats.mstats.zscore(signal), 'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.xlabel('Year', fontsize=label_size)
    plt.ylabel('Dcx (nT/yr)', fontsize=font_size)
    if plot_legend is True:
        plt.legend(['Dcx', 'proxy'], loc='upper right', frameon=False)


def plot_dcx_fft(*, dates, signal, fig_size=(8, 6), font_size=16,
                 label_size=20, plot_legend=False):
    """Compare the DFTs of the proxy signal with that of the Dst index.

    Loads Dcx data (extended, corrected Dst), calculates its DFT and plots it
    alongside the DFT of the signal used as a proxy for unmodelled external
    signal. The length of the time series are padded with zeroes up to the next
    power of two.

    Args:
        dates (datetime.datetime series): dates of time series measurements.
        signal (series): proxy for unmodelled external signal used in the
            denoising process (principal component analysis). The proxy is the
            residual in the noisiest eigendirection(s).
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 16 pt.
        label_size (int): font size for axis labels. Defaults to 20 pt.
        plot_legend (bool): option to include a legend on the plot. Defaults
            to False.
    """
    data_path = '/Users/Grace/Dropbox/BGS_data/monthly_means/Dcx/'
    data_file = 'Dcx_mm_monthly_diff.txt'
    dcx = pd.read_csv(os.path.join(data_path, data_file), sep=r'\s+',
                      header=None)
    dcx.columns = ["year", "month", "monthly_mean"]
    dates = dcx.apply(lambda x: dt.datetime.strptime(
        "{0} {1}".format(int(x['year']), int(x['month'])),
        "%Y %m"), axis=1)

    dcx.insert(0, 'date', dates)
    dcx.drop(dcx.columns[[1, 2]], axis=1, inplace=True)

    sampling_period = 1 / 12.0   # Sampling time in years

    # Find the next power of two higher than the length of the time series and
    # perform the FFT with the series padded with zeroes to this length
    sample_length = int(pow(2, np.ceil(np.log2(len(signal)))))

    dcx_fft = sp.fft(dcx.monthly_mean, sample_length)
    proxy_fft = sp.fft(signal, sample_length)
    freq = np.linspace(0.0, 1.0 / (2.0 * sampling_period), sample_length / 2)
    dcx_power = (2.0 / sample_length) * np.abs(dcx_fft[:sample_length / 2])
    proxy_power = (2.0 / sample_length) * np.abs(proxy_fft[:sample_length / 2])

    plt.figure(figsize=fig_size)
    # Time domain
    plt.subplot(2, 1, 1)
    plt.gca().xaxis_date()
    plt.plot(dcx.date, dcx.monthly_mean, 'b', dates, signal, 'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.xlabel('Year', fontsize=label_size)
    plt.ylabel('Dcx (nT/yr)', fontsize=label_size)
    # Frequency domain
    plt.subplot(2, 1, 2)
    plt.plot(freq, dcx_power, 'b', freq, proxy_power, 'r')
    plt.xlabel('Frequency (cycles/year)', fontsize=font_size)
    plt.ylabel('Power', fontsize=font_size)
    if plot_legend is True:
        plt.legend(['Dcx', 'proxy'], loc='upper right', frameon=False)

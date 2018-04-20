# -*- coding: utf-8 -*-
#    Copyright (C) 2016  Grace Cox (University of Liverpool)
#
#    Released under the MIT license, a copy of which is located at the root of
#    this project.
"""Module containing plotting functions.

Part of the MagPySV package for geomagnetic data analysis. This module provides
various plotting functions.
"""


# import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sp
import magpysv.tools as tools

# Setup matplotlib to use latex fonts in figure labels if needed
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                       r'\usepackage{sansmath}',
                                       r'\sansmath']


def plot_eigenvalues(*, values, fig_size=(8, 6), font_size=12, label_size=16,
                     save_fig=False, write_path=None):
    """Plot eigenvalues of the covariance matrix of SV residuals.

    Produces a plot of the eigenvalues obtained during the principal component
    analysis (PCA) of SV residuals. The largest eigenvalue represents the
    eigendirection with the largest contribution to the residuals (i.e. the
    "noisy" direction.). The smallest eigenvalue represents the
    eigendirection with the smallest contribution to the residuals (the "clean"
    direction). See Wardinski & Holme (2011, GJI,
    https://doi.org/10.1111/j.1365-246X.2011.04988.x) for further details.

    Args:
        values (array): the eigenvalues obtained from the principal component
            analysis of the SV residuals.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    """
    plt.figure(figsize=fig_size)
    plt.semilogy(values, 'bx-')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel(r'$i$', fontsize=label_size)
    plt.ylabel(r'$\lambda_i$', fontsize=label_size)

    if save_fig is True:
        # Create the output directory if it does not exist
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        fpath = os.path.join(write_path, 'eigenvalues.pdf')
        plt.savefig(fpath, bbox_inches='tight')
        plt.close()


def plot_eigenvectors(*, obs_names, eigenvecs, fig_size=(8, 6), font_size=12,
                      label_size=16, save_fig=False, write_path=None):
    """Plot eigenvectors of the covariance matrix of SV residuals.

    Produces a plot of the eigenvectors corresponding to the n largest
    eigenvalues of the covariance matrix obtained during PCA of SV residuals,
    where n is the number of eigenvalues used as a proxy for unmodelled
    external field signal. The n eigenvectors corresponding to the n largest
    eigenvalue represent the directions with the largest contribution
    to the residuals (i.e. the "noisiest" directions). See Wardinski & Holme
    (2011, GJI, https://doi.org/10.1111/j.1365-246X.2011.04988.x)
    for further details.

    Args:
        obs_names (list): list of observatory names given as three digit IAGA
            codes.
        eigenvecs (array): the eigenvalues obtained from the principal
        component analysis of the SV residuals.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    """
    # Loop over directions and plot each eigenvector on a separate subplot
    for direction in range(eigenvecs.shape[1]):
        plt.figure(figsize=fig_size)
        plt.plot(np.abs(eigenvecs[::3, direction]), 'bx',
                 np.abs(eigenvecs[1::3, direction]), 'rx',
                 np.abs(eigenvecs[2::3, direction]), 'cx',
                 markersize=10, mew=3)
        plt.ylim(0, 1)
        plt.grid()
        plt.yticks(fontsize=font_size)
        plt.xticks(range(len(obs_names)), obs_names, fontsize=font_size)
        plt.xticks(rotation=60)
        plt.ylabel(r'$\mathbf{{v}}_{%03d}$' % (direction), fontsize=label_size)
        plt.legend(['x direction', 'y direction', 'z direction'],
                   loc='upper right', frameon=False, fontsize=label_size)
        plt.xlabel('Location', fontsize=label_size)

        if save_fig is True:
            # Create the output directory if it does not exist
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            fpath = os.path.join(write_path, 'eigendirection%03d.pdf' % direction)
            plt.savefig(fpath, bbox_inches='tight')
            plt.close()


def plot_mf(*, dates, mf, model, obs, model_name, fig_size=(8, 6),
            font_size=12, label_size=16, plot_legend=True, save_fig=False,
            write_path=None):
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
        model_name (str): field model name.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        plot_legend (bool): option to include a legend on the plot. Defaults
            to True.
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    """
    plt.figure(figsize=fig_size)
    # X component
    plt.subplot(3, 1, 1)
    plt.title(obs, fontsize=label_size)
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
        plt.legend([obs, model_name], loc='best', frameon=False)

    if save_fig is True:
        # Create the output directory if it does not exist
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        fpath = os.path.join(write_path, obs + '.pdf')
        plt.savefig(fpath, bbox_inches='tight')
        plt.close()


def plot_sv(*, dates, sv, model, obs, model_name, fig_size=(8, 6),
            font_size=12, label_size=16, plot_legend=False, plot_average=False,
            window_length=12, min_samples=3, save_fig=False, write_path=None):
    """Plot the SV and model prediction for a single observatory.

    Produces a plot of the X, Y and Z components of the SV and field
    model prediction for a single observatory.

    Args:
        dates (datetime.datetime): dates of time series measurements.
        sv (time series): X, Y and Z components of SV at a single location.
        model (time series): X, Y and Z components of the SV predicted by a
            field model for the same location as the data.
        obs (str): observatory name given as three digit IAGA code.
        model_name (str): field model name.
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
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    """
    if plot_average is True:
        plt.figure(figsize=fig_size)
        # X component
        plt.subplot(3, 1, 1)
        plt.title(obs, fontsize=label_size)
        plt.gca().xaxis_date()
        plt.plot(dates, sv.ix[:, 0], 'b', dates, sv.ix[:, 0].rolling(
            window=window_length, center=True, min_periods=min_samples).mean(),
            'r', dates, model.ix[:, 0], 'k')
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
            'r', dates, model.ix[:, 1], 'k')
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
            'r', dates, model.ix[:, 2], 'k')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xlabel('Year', fontsize=label_size)
        plt.ylabel(r'$\dot{z}$ (nT/yr)', fontsize=label_size)

        if plot_legend is True:
            plt.legend([obs, 'Running average', model_name], loc='best',
                       frameon=False)
    else:
        plt.figure(figsize=fig_size)
        # X component
        plt.subplot(3, 1, 1)
        plt.title(obs, fontsize=label_size)
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
            plt.legend([obs, model_name], loc='best', frameon=False)

    if save_fig is True:
        # Create the output directory if it does not exist
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        fpath = os.path.join(write_path, 'clean_' + obs + '.pdf')
        plt.savefig(fpath, bbox_inches='tight')
        plt.close()


def plot_sv_comparison(*, dates, noisy_sv, denoised_sv, model, obs, model_name,
                       fig_size=(8, 6), font_size=12, label_size=16,
                       plot_legend=False, plot_average=False,
                       window_length=12, min_samples=3, save_fig=False,
                       write_path=None, corrected_residuals, residuals,
                       plot_rms=False):
    """Plot noisy/denoised SV and model prediction for a single observatory.

    Produces a plot of the X, Y and Z components of the noisy SV, the denoised
    SV and field model prediction for a single observatory.

    Args:
        dates (datetime.datetime): dates of time series measurements.
        noisy_sv (time series): X, Y and Z components of uncorrected SV at a
            single location.
        denoised_sv (time series): X, Y and Z components of denoised SV at a
            single location.
        residuals (time series): difference between modelled and observed SV.
        corrected_residuals (time series): difference between modelled and
            denoised observed SV.
        model (time series): X, Y and Z components of the SV predicted by a
            field model for the same location as the data.
        model_name (str): field model name.
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
        plot_rms (bool): option to calculate the rms before and after denoising
        and display the values on the figure. Defaults to False.
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    Returns:
        (tuple): tuple containing:

        - rms_ratio_x (*float*):
            ratio of rms values for X component residuals before and after
            denoising.
        - rms_ratio_y (*float*):
            ratio of rms values for Y component residuals before and after
            denoising.
        - rms_ratio_z (*float*):
            ratio of rms values for Z component residuals before and after
            denoising.
    """
    plt.figure(figsize=fig_size)
    plt.title(obs, fontsize=label_size)

    if plot_rms is True:
        # Calculate the rms before and after denoising
        rms_x_noisy = np.sqrt(np.nanmean(np.square(residuals.ix[:, 0])))
        rms_x_denoised = np.sqrt(np.nanmean(np.square(
            corrected_residuals.ix[:, 0])))
        rms_y_noisy = np.sqrt(np.nanmean(np.square(residuals.ix[:, 1])))
        rms_y_denoised = np.sqrt(np.nanmean(np.square(
            corrected_residuals.ix[:, 1])))
        rms_z_noisy = np.sqrt(np.nanmean(np.square(residuals.ix[:, 2])))
        rms_z_denoised = np.sqrt(np.nanmean(np.square(
            corrected_residuals.ix[:, 2])))

    if plot_average is True:
        # X component
        plt.subplot(3, 1, 1)
        plt.title(obs, fontsize=label_size)
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

        if plot_rms is True:
            plt.annotate('rms = ' + "{:.0f}".format(rms_x_noisy),
                         xy=(0.05, 0.9), xycoords='axes fraction',
                         fontsize=font_size, color='b')
            plt.annotate('rms = ' + "{:.0f}".format(rms_x_denoised),
                         xy=(0.05, 0.8), xycoords='axes fraction',
                         fontsize=font_size, color='r')
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

        if plot_rms is True:
            plt.annotate('rms = ' + "{:.0f}".format(rms_y_noisy),
                         xy=(0.05, 0.9), xycoords='axes fraction',
                         fontsize=font_size, color='b')
            plt.annotate('rms = ' + "{:.0f}".format(rms_y_denoised),
                         xy=(0.05, 0.8), xycoords='axes fraction',
                         fontsize=font_size, color='r')
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

        if plot_rms is True:
            plt.annotate('rms = ' + "{:.0f}".format(rms_z_noisy),
                         xy=(0.05, 0.9), xycoords='axes fraction',
                         fontsize=font_size, color='b')
            plt.annotate('rms = ' + "{:.0f}".format(rms_z_denoised),
                         xy=(0.05, 0.8), xycoords='axes fraction',
                         fontsize=font_size, color='r')

        if plot_legend is True:
            plt.legend([obs, 'Denoised SV', 'Running average', model_name],
                       loc='best', frameon=False)
    else:
        # X component
        plt.subplot(3, 1, 1)
        plt.title(obs, fontsize=label_size)
        plt.gca().xaxis_date()
        plt.plot(dates, noisy_sv.ix[:, 0], 'b', dates, denoised_sv.ix[:, 0],
                 'r', dates, model.ix[:, 0], 'k')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.ylabel(r'$\dot{x}$ (nT/yr)', fontsize=label_size)

        if plot_rms is True:
            plt.annotate('rms = ' + "{:.0f}".format(rms_x_noisy),
                         xy=(0.05, 0.9), xycoords='axes fraction',
                         fontsize=font_size, color='b')
            plt.annotate('rms = ' + "{:.0f}".format(rms_x_denoised),
                         xy=(0.05, 0.8), xycoords='axes fraction',
                         fontsize=font_size, color='r')
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

        if plot_rms is True:
            plt.annotate('rms = ' + "{:.0f}".format(rms_y_noisy),
                         xy=(0.05, 0.9), xycoords='axes fraction',
                         fontsize=font_size, color='b')
            plt.annotate('rms = ' + "{:.0f}".format(rms_y_denoised),
                         xy=(0.05, 0.8), xycoords='axes fraction',
                         fontsize=font_size, color='r')
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

        if plot_rms is True:
            plt.annotate('rms = ' + "{:.0f}".format(rms_z_noisy),
                         xy=(0.05, 0.9), xycoords='axes fraction',
                         fontsize=font_size, color='b')
            plt.annotate('rms = ' + "{:.0f}".format(rms_z_denoised),
                         xy=(0.05, 0.8), xycoords='axes fraction',
                         fontsize=font_size, color='r')

        if plot_legend is True:
            plt.legend([obs, 'Denoised SV', model_name],
                       loc='best', frameon=False)

    if save_fig is True:
        # Create the output directory if it does not exist
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        fpath = os.path.join(write_path, obs + '.pdf')
        plt.savefig(fpath, bbox_inches='tight')
        plt.close()

    if plot_rms is True:
        rms_ratio_x = rms_x_denoised/rms_x_noisy
        rms_ratio_y = rms_y_denoised/rms_y_noisy
        rms_ratio_z = rms_z_denoised/rms_z_noisy
        return rms_ratio_x, rms_ratio_y, rms_ratio_z
    else:
        return


def plot_index(*, index_file, dates, projected_residuals, fig_size=(8, 6),
               font_size=12, label_size=16, plot_legend=True, save_fig=False,
               write_path=None, index_name='Dst'):
    """Compare the proxy used to denoise the SV data with a geomagnetic index.

    Loads geomagnetic index and plots it alongside the signal
    used as a proxy for unmodelled external signal. Both time series are
    reduced to zero mean and unit variance (i.e. their zscore) for plotting.

    Args:
        dates (datetime.datetime): dates of time series measurements.
        index_file (str): path to the file containing index data.
        projected_residuals (time series): difference between modelled and
            SV rotated into the eigendirections obtained during denoising
            (principal component analysis). The proxy for unmodelled external
            signal is the residual projected in the noisiest eigendirection(s).
        index_name (str): name of index used in comparison e.g. Dst or ap.
            Defaults to Dst.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        plot_legend (bool): option to include a legend on the plot. Defaults
            to True.
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    """
    for direction in range(projected_residuals.shape[1]):
        signal = projected_residuals[:, direction]
        coeff, df = tools.calculate_correlation_index(
            dates=dates, signal=signal, index_file=index_file)
        # Plot the zscore of the two time series
        plt.figure(figsize=fig_size)
        plt.gca().xaxis_date()
        plt.plot(df.date, sp.stats.mstats.zscore(df.index_vals), 'b',
                 dates, sp.stats.mstats.zscore(signal), 'r')
        plt.gcf().autofmt_xdate()
        plt.axis('tight')
        plt.xlabel('Year', fontsize=label_size)
        plt.ylabel('Signal (nT/yr)', fontsize=label_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.annotate('|r| = ' + "{:.2f}".format(np.abs(coeff)), xy=(0.05, 0.9),
                     xycoords='axes fraction', fontsize=16)

        if plot_legend is True:
            plt.legend([index_name, 'proxy'], loc='best', frameon=False,
                       fontsize=16)

        if save_fig is True:
            # Create the output directory if it does not exist
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            fpath = os.path.join(write_path, index_name \
                + '_eigendirection%03d.pdf' % direction)
            plt.savefig(fpath, bbox_inches='tight')
            plt.close()


def plot_index_dft(*, index_file, dates, signal, fig_size=(8, 6), font_size=12,
                   label_size=16, plot_legend=True, save_fig=False,
                   write_path=None, index_name='Dst'):
    """Compare the DFTs of the proxy signal with that of a geomagnetic index.

    Loads index data, calculates its DFT using an FFT algorithm and plots it
    alongside the DFT of the signal used as a proxy for unmodelled external
    signal. The length of the time series are padded with zeroes up to the next
    power of two.

    Args:
        dates (datetime.datetime): dates of time series measurements.
        signal (time series): proxy for unmodelled external signal used in the
            denoising process (principal component analysis). The proxy is the
            residual in the noisiest eigendirection(s).
        index_file (str): path to the file containing index data.
        index_name (str): name of index used in comparison e.g. Dst or Dcx.
            Defaults to Dst.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        plot_legend (bool): option to include a legend on the plot. Defaults
            to True.
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    """
    coeff, df = tools.calculate_correlation_index(
        dates=dates, signal=signal, index_file=index_file)
    sampling_period = 1 / 12.0   # Sampling time in years

    # Find the next power of two higher than the length of the time series and
    # perform the DFT with the series padded with zeroes to this length
    sample_length = int(pow(2, np.ceil(np.log2(len(df.proxy)))))
    index_dft = sp.fft(df.index_vals, sample_length)
    proxy_dft = sp.fft(df.proxy, sample_length)
    freq = np.linspace(0.0, 1.0 / (2.0 * sampling_period), sample_length / 2)
    index_power = (2.0 / sample_length) * np.abs(
        index_dft[:sample_length // 2])
    proxy_power = (2.0 / sample_length) * np.abs(
        proxy_dft[:sample_length // 2])

    plt.figure(figsize=fig_size)
    # Time domain
    plt.subplot(2, 1, 1)
    plt.gca().xaxis_date()
    plt.plot(df.date, df.index_vals, 'b', dates, signal, 'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylabel('Signal (nT/yr)', fontsize=label_size)
    plt.annotate('|r| = ' + "{:.2f}".format(np.abs(coeff)), xy=(0.05, 0.9),
                 xycoords='axes fraction')
    # Frequency domain
    plt.subplot(2, 1, 2)
    plt.plot(freq, index_power, 'b', freq, proxy_power, 'r')
    plt.axis('tight')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel('Frequency (cycles/year)', fontsize=label_size)
    plt.ylabel('Power', fontsize=label_size)

    if plot_legend is True:
        plt.legend([index_name, 'proxy'], loc='best', frameon=False)

    if save_fig is True:
        # Create the output directory if it does not exist
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        fpath = os.path.join(write_path, index_name + '_dft.pdf')
        plt.savefig(fpath, bbox_inches='tight')
        plt.close()


def plot_outliers(*, dates, signal, obs_name, outliers, signal_type='SV',
                  fig_size=(8, 6), font_size=12, label_size=16, save_fig=False,
                  write_path=None):
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
        signal_type (str): specify whether magnetic field ('MF') or secular
            variation ('SV') is plotted. Defaults to SV.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    """
    plt.figure(figsize=fig_size)
    plt.plot(dates, signal, 'k', dates, outliers, 'r^')
    plt.axis('tight')
    plt.xlabel('Year', fontsize=label_size)
    if signal_type is 'SV':
        plt.ylabel('SV (nT/yr)', fontsize=label_size)
    else:
        plt.ylabel('MF (nT)', fontsize=label_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend([obs_name, 'outlier'], loc='best', frameon=False)

    if save_fig is True:
        # Create the output directory if it does not exist
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        fpath = os.path.join(write_path, obs_name + '_outliers.pdf')
        plt.savefig(fpath, bbox_inches='tight')
        plt.close()


def plot_residuals_dft(*, projected_residuals, dates, fig_size=(10, 8),
                       font_size=12, label_size=16, plot_legend=True,
                       save_fig=False, write_path=None):
    """Compare the DFTs of the projected residuals with each other.

    Calculates the DFT of the residuals in each eigendirection given and plots
    it alongside the residuals themselves. Produces a single figure with each
    eigendirection included as a subplot. Use plot_residuals_dft_all if a
    separate figure per eigendirection is desired. The length of the time
    series are padded with zeroes up to the next power of two.

    Args:
        dates (datetime.datetime): dates of time series measurements.
        projected_residuals (time series): difference between modelled and
            SV rotated into the eigendirections obtained during denoising
            (principal component analysis). The proxy for unmodelled external
            signal is the residual projected in the noisiest eigendirection(s).
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        plot_legend (bool): option to include a legend on the plot. Defaults to
            True.
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    """
    fig_count = 1
    # Create figure with shared subplot axes
    fig, ax = plt.subplots(nrows=projected_residuals.shape[1], ncols=2,
                           sharex=True, sharey=True, figsize=fig_size)

    sampling_period = 1 / 12.0   # Sampling time in years
    sample_length = int(pow(2, np.ceil(np.log2(projected_residuals.shape[0]))))

    # Iterate over the eigendirections and produce a figure for each
    for direction in range(projected_residuals.shape[1]):
        residual_dft = sp.fft(projected_residuals[:, direction], sample_length)
        freq = np.linspace(0.0, 1.0 / (2.0 * sampling_period),
                           sample_length / 2)
        residual_power = (2.0 / sample_length) * np.abs(
            residual_dft[:sample_length // 2])
        plt.subplot(projected_residuals.shape[1], 2, direction + fig_count)
        plt.gca().xaxis_date()
        plt.plot(dates, projected_residuals[:, direction], 'b')
        plt.xticks(rotation=60)
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.ylabel('Direction {}'.format(direction), fontsize=label_size-2)
        fig_count = fig_count + 1
        plt.subplot(projected_residuals.shape[1], 2, direction + fig_count)
        # Frequency domain
        plt.plot(freq, residual_power, 'b')
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
    fig.text(0.00, 0.5, 'Residuals (nT/yr)', va='center', rotation='vertical',
             fontsize=label_size)
    fig.text(0.25, 0.02, 'Date', ha='center', fontsize=label_size)
    fig.text(0.75, 0.02, 'Frequency (cycles/yr)', ha='center',
             fontsize=label_size)

    if save_fig is True:
        # Create the output directory if it does not exist
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        fpath = os.path.join(write_path, 'residuals_dft.pdf')
        plt.savefig(fpath, bbox_inches='tight')
        plt.close()


def plot_residuals_dft_all(*, projected_residuals, dates, fig_size=(10, 8),
                           font_size=12, label_size=16,
                           save_fig=False, write_path=None):
    """Compare the DFTs of the projected residuals with each other.

    Calculates the DFT of the residuals in each eigendirection given and plots
    it alongside the residuals themselves. Produces a separate figure per
    eigendirection. Use plot_residuals_dft if a single figure with each
    eigendirection included as a subplot is desired. The length of the time
    series are padded with zeroes up to the next power of two.

    Args:
        dates (datetime.datetime): dates of time series measurements.
        projected_residuals (time series): difference between modelled and
            SV rotated into the eigendirections obtained during denoising
            (principal component analysis). The proxy for unmodelled external
            signal is the residual projected in the noisiest eigendirection(s).
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        plot_legend (bool): option to include a legend on the plot. Defaults to
            True.
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    """

    sampling_period = 1 / 12.0   # Sampling time in years
    sample_length = int(pow(2, np.ceil(np.log2(projected_residuals.shape[0]))))

    # Iterate over the eigendirections and produce a figure for each
    for direction in range(projected_residuals.shape[1]):
        residual_dft = sp.fft(projected_residuals[:, direction], sample_length)
        freq = np.linspace(0.0, 1.0 / (2.0 * sampling_period),
                           sample_length / 2)
        residual_power = (2.0 / sample_length) * np.abs(
            residual_dft[:sample_length // 2])
        ax = plt.subplots(nrows=1, ncols=2, figsize=fig_size)[1]
        plt.subplot(2, 1, 1)
        plt.gca().xaxis_date()
        plt.plot(dates, projected_residuals[:, direction], 'b')
        plt.xticks(rotation=60)
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.ylabel('Residuals (nT/yr)', fontsize=label_size)
        plt.subplot(2, 1, 2)
        # Frequency domain
        plt.plot(freq, residual_power, 'b')
        plt.axis('tight')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xlabel('Frequency (cycles/yr)', fontsize=label_size)
        plt.ylabel('DFT', fontsize=label_size)

        if save_fig is True:
            # Create the output directory if it does not exist
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            fpath = os.path.join(write_path, \
                'dft_eigendirection%03d.pdf' % direction)
            plt.savefig(fpath, bbox_inches='tight')
            plt.close()


def compare_proxies(*, fname1, fname2, legend_text, fig_size=(8, 6),
                    font_size=12, label_size=16,
                    save_fig=False, write_path=None):
    """Compare proxies of unmodelled external signal for different analyses.

    Calculates the correlation coefficients of two given proxies for unmodelled
    external signals and includes it on a plot of the two series. Each
    proxy is formed of the SV residuals projected into the eigendirection(s) of
    the largest eigenvalues of the residual covariance matrix. The proxies are
    reduced to zero-mean and unit-variance on the plots (zscore).

    Args:
        fname1 (str): path to file containing a time series of proxy for noise.
        fname2 (str): path to a second file containing a proxy for noise.
        legend_text (str): text to include on the plot legend.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    """
    proxy1 = pd.read_csv(fname1, parse_dates=[0], names=['date', 'proxy1'],
                         skiprows=1, index_col=None)
    proxy2 = pd.read_csv(fname2, parse_dates=[0], names=['date', 'proxy2'],
                         skiprows=1, index_col=None)
    # Merge the two dataframes so that only dates contained within both are
    # retained
    df = pd.merge(proxy1.dropna(), proxy2.dropna(), on='date', how='inner')
    coeff = np.corrcoef(df.proxy1, df.proxy2)

    # Plot the zscore of the two time series
    plt.figure(figsize=fig_size)
    plt.gca().xaxis_date()
    plt.plot(df.date, sp.stats.mstats.zscore(df.proxy1), 'b',
             df.date, sp.stats.mstats.zscore(df.proxy2), 'r')
    plt.gcf().autofmt_xdate()
    plt.axis('tight')
    plt.xlabel('Year', fontsize=label_size)
    plt.ylabel('Proxy signal (nT/yr)', fontsize=label_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(legend_text, loc='upper right', frameon=False, fontsize=16)
    plt.annotate('|r| = ' + "{:.2f}".format(np.abs(coeff.data[0, 1])),
                 xy=(0.05, 0.9),
                 xycoords='axes fraction', fontsize=16)

    if save_fig is True:
        # Create the output directory if it does not exist
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        fpath = os.path.join(write_path, 'proxy_comparison_' + legend_text[0] + '_' \
            + legend_text[1] + '.pdf')
        plt.savefig(fpath, bbox_inches='tight')
        plt.close()


def rms_ratios(*, rms, fig_size=(8, 6), font_size=12, label_size=16,
               save_fig=False, write_path=None):
    """Plot the rms of residuals after removing successive eigendirections.

    Plots the ratio of the residuals rms values before and after denoising, for
    different numbers of eigendirections are removed from the data. Removing
    all eigendirections gives a ratio of zero as the denoised SV now equals the
    model prediction. Requires the dnoising to be run several times, each time
    using a different number of eigendirections for the external signal proxy
    (i.e. different values of the argument proxy_number in calls to
    eigenvalue_analysis.) Uses output from plot_sv_comparison when that
    function is run with the option plot_rms=True.

    Args:
        rms (dict): rms ratios for each component after running
            eigenvalue_analysis with different values for the proxy_number
            argument.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    """
    for observatory in rms.keys():
        plt.figure(figsize=fig_size)
        plt.plot(range(0, 3*len(rms.keys())), rms[observatory]['x'], 'bx-',
                 markersize=7, mew=2)
        plt.plot(range(0, 3*len(rms.keys())), rms[observatory]['y'], 'rx-',
                 markersize=7, mew=2)
        plt.plot(range(0, 3*len(rms.keys())), rms[observatory]['z'], 'cx-',
                 markersize=7, mew=2)
        plt.legend(['x', 'y', 'z'],
                   loc='best', frameon=False, fontsize=16)
        plt.xlabel('Eigendirections removed', fontsize=label_size)
        plt.ylabel('%s rms/denoised rms' % observatory, fontsize=label_size)
        plt.ylim([0, 1.2])
        plt.xlim([0, 3*len(rms.keys())-1])
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid()

        if save_fig is True:
            # Create the output directory if it does not exist
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            fpath = os.path.join(write_path, 'rms_ratio_%s.pdf' % observatory)
            plt.savefig(fpath, bbox_inches='tight')
            plt.close()

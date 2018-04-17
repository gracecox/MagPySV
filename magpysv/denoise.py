# -*- coding: utf-8 -*-
#    Copyright (C) 2016  Grace Cox (University of Liverpool)
#
#    Released under the MIT license, a copy of which is located at the root of
#    this project.
"""Module containing functions to remove external signal from geomagnetic data.

Part of the MagPySV package for geomagnetic data analysis. This module provides
various functions to denoise geomagnetic data by performing principal component
analysis and identifying and removing outliers. Also contains an outlier
detection function based on median absolute deviation from the median (MAD).
"""


import pandas as pd
import magpysv.plots as plots
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import Imputer


def eigenvalue_analysis_impute(*, dates, obs_data, model_data, residuals,
                               proxy_number=1):
    """Remove external signal from SV data using Principal Component Analysis.

    Perform principal component analysis (PCA) on secular variation
    residuals (the difference between the observed SV and that predicted by a
    geomagnetic field model) calculated from annual differences of monthly
    means at several observatories. Uses the imputer from sklearn.preprocessing
    to fill in missing data points and calculates the singular values of the
    data matrix for n observatories (uses Singular Values Decomposition, SVD).
    The residuals are rotated into the eigendirections and denoised using the
    method detailed in Wardinski & Holme (2011, GJI,
    https://doi.org/10.1111/j.1365-246X.2011.04988.x). The SV residuals of the
    noisy component for all observatories combined are used as a proxy for the
    unmodelled external signal. The denoised data are then rotated back into
    geographic coordinates. The pca algorithm outputs the singular values
    (these are equal to the square root of the eigenvalues of the covariance
    matrix) sorted from largest to smallest, so the corresponding eigenvector
    matrix has the 'noisy' direction in the first column and the 'clean'
    direction in the final column.

    Note that the SVD algorithm cannot be used if any data are missing, which
    is why imputation is needed with this method. The function
    denoise.eigenvalue_analysis permits missing values and does not
    infill them - that is the more robust function.

    Smallest eigenvalue: 'quiet' direction

    Largest eiegenvalue: 'noisy' direction

    Args:
        dates (datetime.datetime): dates of the time series measurements.
        obs_data (pandas.DataFrame): dataframe containing columns for
            monthly/annual means of the X, Y and Z components of the secular
            variation at the observatories of interest.
        model_data (pandas.DataFrame): dataframe containing columns for field
            model prediction of the X, Y and Z components of the secular
            variation at the same observatories as in obs_data.
        residuals (pandas.DataFrame): dataframe containing the SV residuals
            (difference between the observed data and model prediction).
        proxy_number (int): the number of 'noisy' directions used to create
            the proxy for the external signal removal. Default value is 1 (only
            the residual in the direction of the largest eigenvalue is used).
            Using n directions means that proxy is the sum of the SV residuals
            in the n noisiest eigendirections.

    Returns:
        (tuple): tuple containing:

        - denoised_sv (*pandas.DataFrame*):
            dataframe with dates in the first
            column and columns for the denoised X, Y and Z secular variation
            components at each of the observatories for which data were
            provided.
        - proxy (*array*):
            the signal that was used as a proxy for unmodelled
            external magnetic field in the denoising stage.
        - eig_values (*array*):
            the singular values of the obs_data matrix.
        - eig_vectors (*array*):
            the eigenvectors associated with the n largest
            singular values of the data matrix. For example, if the residuals
            in the two 'noisiest' directions are used as the proxy for external
            signal, then these two eigenvectors are returned.
        - projected_residuals (*array*):
            SV residuals rotated into the eigendirections.
        - corrected_residuals (*array*):
            SV residuals after the denoising process.
    """
    # Fill in missing SV values (indicated as NaN in the data files)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputed_residuals = imp.fit_transform(residuals)

    pca = sklearnPCA()
    projected_residuals = pca.fit_transform(imputed_residuals)
    eig_values = pca.explained_variance_
    eig_vectors = pca.components_

    # Use the method of Wardinski & Holme (2011) to remove unmodelled external
    # signal in the SV residuals. The variable 'proxy' contains the noisy
    # component residual for all observatories combined

    corrected_residuals = []

    if proxy_number == 1:
        noisy_direction = eig_vectors[:, 0]
        proxy = projected_residuals[:, 0]
        for idx in range(len(proxy)):
            corrected_residuals.append(
                imputed_residuals.data[idx, :] - proxy[idx] * noisy_direction)
    elif proxy_number > 1:
        noisy_direction = eig_vectors[:, 0:proxy_number]
        proxy = np.sum(projected_residuals[:, 0:proxy_number], axis=1)
        for idx in range(len(projected_residuals[:, 0])):
            corrected = imputed_residuals.data[idx, :]
            for direction in range(proxy_number):
                corrected = corrected - projected_residuals[idx, direction] \
                    * noisy_direction[:, direction]
            corrected_residuals.append(corrected)

    corrected_residuals = pd.DataFrame(corrected_residuals,
                                       columns=obs_data.columns)
    denoised_sv = pd.DataFrame(
        corrected_residuals.values + model_data.values,
        columns=obs_data.columns)
    denoised_sv.insert(0, 'date', dates)

    return denoised_sv, proxy, eig_values, eig_vectors, projected_residuals,\
        corrected_residuals.astype('float')


def eigenvalue_analysis(*, dates, obs_data, model_data, residuals,
                        proxy_number=1):
    """Remove external signal from SV data using principal Component Analysis.

    Perform principal component analysis (PCA) on secular variation
    residuals (the difference between the observed SV and that predicted by a
    geomagnetic field model) calculated from annual differences of monthly
    means at several observatories. Uses masked arrays to discount missing data
    points and calculates the eigenvalues/vectors of the (3nx3n) covariance
    matrix for n observatories. The residuals are rotated into the
    eigendirections and denoised using the method detailed in Wardinski & Holme
    (2011, GJI, https://doi.org/10.1111/j.1365-246X.2011.04988.x). The SV
    residuals of the noisy component for all observatories
    combined are used as a proxy for the unmodelled external signal. The
    denoised data are then rotated back into geographic coordinates. The PCA
    algorithm outputs the eigenvalues sorted from largest to smallest, so the
    corresponding eigenvector matrix has the 'noisy' direction in the first
    column and the 'clean' direction in the final column.

    This algorithm masks missing data so that they are not taken into account
    during the PCA. Missing values are not infilled or estimated, so NaN
    values in the input dataframe are given as NaN values in the output.

    Smallest eigenvalue 'quiet' direction

    Largest eiegenvalue 'noisy' direction

    Args:
        dates (datetime.datetime): dates of the time series measurements.
        obs_data (pandas.DataFrame): dataframe containing columns for
            monthly/annual means of the X, Y and Z components of the secular
            variation at the observatories of interest.
        model_data (pandas.DataFrame): dataframe containing columns for field
            model prediction of the X, Y and Z components of the secular
            variation at the same observatories as in obs_data.
        residuals (pandas.DataFrame): dataframe containing the SV residuals
            (difference between the observed data and model prediction).
        proxy_number (int): the number of 'noisy' directions used to create
            the proxy for the external signal removal. Default value is 1 (only
            the residual in the direction of the largest eigenvalue is used).
            Using n directions means that proxy is the sum of the SV residuals
            in the n noisiest eigendirections.

    Returns:
        (tuple): tuple containing:

        - denoised_sv (*pandas.DataFrame*):
            dataframe with datetime objects in the
            first column and columns for the denoised X, Y and Z SV components
            at each of the observatories for which data were provided.
        - proxy (*array*):
            the signal that was used as a proxy for unmodelled
            external magnetic field in the denoising stage.
        - eig_values (*array*):
            the eigenvalues of the obs_data matrix.
        - eig_vectors (*array*):
            the eigenvectors associated with the n largest
            eigenvalues of the data matrix. For example, if the residuals
            in the two 'noisiest' directions are used as the proxy for external
            signal, then these two eigenvectors are returned.
        - projected_residuals (*array*):
            SV residuals rotated into the eigendirections.
        - corrected_residuals (*array*):
            SV residuals after the denoising process.
    """
    # Create a masked version of the residuals array so that we can perform the
    # PCA ignoring all nan values
    masked_residuals = np.ma.array(residuals, mask=np.isnan(residuals))

    # Calculate the covariance matrix of the masked residuals array
    covariance_matrix = np.ma.cov(masked_residuals, rowvar=False,
                                  allow_masked=True)
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eig_values, eig_vectors = np.linalg.eig(covariance_matrix)
    # Sort the eigenvalues in decreasing order
    idx = np.argsort(eig_values)[::-1]
    eig_values = eig_values[idx]
    # Sort the eigenvectors according to the same index
    eig_vectors = eig_vectors[:, idx]

    # Project the residuals onto the eigenvectors
    projected_residuals = np.ma.dot(masked_residuals, eig_vectors)

    # Use the method of Wardinski & Holme (2011) to remove unmodelled external
    # signal in the SV residuals. The variable 'proxy' contains the noisy
    # component residual for all observatories combined

    corrected_residuals = []

    if proxy_number == 1:
        noisy_direction = eig_vectors[:, 0]
        proxy = projected_residuals[:, 0]
        for idx in range(len(proxy)):
            corrected_residuals.append(
                masked_residuals.data[idx, :] - proxy[idx] * noisy_direction)
    elif proxy_number > 1:
        noisy_direction = eig_vectors[:, 0:proxy_number]
        proxy = np.sum(projected_residuals[:, 0:proxy_number], axis=1)
        for idx in range(len(projected_residuals[:, 0])):
            corrected = masked_residuals.data[idx, :]
            for direction in range(proxy_number):
                corrected = corrected - projected_residuals[idx, direction] \
                    * noisy_direction[:, direction]
            corrected_residuals.append(corrected)

    corrected_residuals = pd.DataFrame(corrected_residuals,
                                       columns=obs_data.columns)
    denoised_sv = pd.DataFrame(
        corrected_residuals.values + model_data.values,
        columns=obs_data.columns)

    denoised_sv.insert(0, 'date', dates)

    return denoised_sv, proxy, eig_values, eig_vectors, projected_residuals,\
        corrected_residuals.astype('float')


def detect_outliers(*, dates, signal, obs_name, window_length, threshold,
                    signal_type='SV', plot_fig=False, save_fig=False,
                    write_path=None, fig_size=(8, 6), font_size=12,
                    label_size=16):
    """Detect outliers in a time series and remove them.

    Use the median absolute deviation from the median (MAD) to identify
    outliers. The time series are long and highly variable so it is not
    appropriate to use single values of median to represent the whole series.
    The function uses a running median to better characterise the series
    (the window length and a threshold value stating many MADs from the median
    a point must be before it is classed as an outlier are user-specified).

    Args:
        dates (datetime.datetime): dates of the time series measurements.
        signal (array): array (or column from a pandas.DataFrame) containing
            the time series of interest.
        obs_name (str): states the component of interest and the three digit
           IAGA observatory name.
        window_length (int): number of months over which to take the running
            median.
        threshold (float): the minimum number of median absolute deviations a
            point must be away from the median in order to be considered an
            outlier.
        signal_type (str): specify whether magnetic field ('MF') or secular
            variation ('SV') is plotted. Defaults to SV.
        plot_fig (bool): option to plot figure of the time series and
            identified outliers. Defaults to False.
        save_fig (bool): option to save figure if plotted. Defaults to False.
        write_path (str): output path for figure if saved.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.

    Returns:
        signal (array):
            the input signal with identified outliers removed (set to NaN).
    """
    signal_temp = pd.DataFrame(data=signal.copy())
    # Account for missing values when using rolling_median and rolling_std.
    # ffill (bfill) propagates the closest value forwards (backwards) through
    # nan values. E.g. [np.nan, np.nan, 1, 9, 7, np.nan, np.nan] returns as
    # [1, 1, 1, 9, 7, 7, 7]. The limit of half the window length is used so the
    # first ffill cannot overwrite the beginning of the next valid interval
    # (bfill values are used there instead).
    signal_temp = signal_temp.ffill(limit=int(window_length / 2 + 1)).bfill()
    # calculate the running median and median absolute standard deviation
    running_median = signal_temp.rolling(window=window_length,
                                         center=True).median().bfill().ffill()
    diff = (signal_temp - running_median).abs()
    med_abs_deviation = diff.rolling(window=window_length,
                                     center=True).median().bfill().ffill()
    # Normalise the median abolute deviation
    modified_z_score = diff / med_abs_deviation
    # Identify outliers
    outliers = signal_temp[modified_z_score > threshold]
    # Plot the outliers and original time series if required
    if plot_fig is True:
        plots.plot_outliers(dates=dates, obs_name=obs_name, signal=signal,
                            outliers=outliers, save_fig=save_fig,
                            write_path=write_path, fig_size=fig_size,
                            font_size=font_size, label_size=label_size,
                            signal_type=signal_type)
    # Set the outliers to NaN
    idx = np.where(modified_z_score > threshold)[0]
    signal.iloc[idx] = np.nan
    return signal.astype('float')

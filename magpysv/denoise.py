# -*- coding: utf-8 -*-
#    Copyright (C) 2016  Grace Cox (University of Liverpool)
#
#    Released under the MIT license, a copy of which is located at the root of
#    this project.
"""Module containing functions to remove external signal from geomagnetic data.

Part of the MagPySV package for geomagnetic data analysis. This module provides
various functions to denoise geomagnetic data by performing principal component
analysis and identifying and removing outliers.
"""


import pandas as pd
import magpysv.svplots as svplots
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
    method detailed in Wardinski & Holme (2011). The SV residuals of the noisy
    component for all observatories combined are used as a proxy for the
    unmodelled external signal. The denoised data are then rotated back into
    geographic coordinates. The pca algorithm outputs the singular values
    (these are equal to the square root of the eigenvalues of the covariance
    matrix) sorted from largest to smallest, so the corresponding eigenvector
    matrix has the 'noisy' direction in the first column and the 'clean'
    direction in the final column.

    Note that the SVD algorithm cannot be used if any data are missing, which
    is why imputation is needed with this method. The function
    denoise.eigenvalue_analysis permits missing values and does not
    infill them.

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
    noisy_direction = eig_vectors[0, :]
    proxy = 0

    if proxy_number == 1:
        proxy = projected_residuals[:, 0]
    elif proxy_number > 1:
        for direction in range(proxy_number):
            proxy = proxy + projected_residuals[:, direction]

    corrected_residuals = []

    for idx in range(len(proxy)):
        corrected_residuals.append(
            imputed_residuals[idx, :] - proxy[idx] * noisy_direction)

    corrected_residuals = pd.DataFrame(corrected_residuals,
                                       columns=obs_data.columns)
    denoised_sv = pd.DataFrame(
        corrected_residuals.values + model_data.values,
        columns=obs_data.columns)
    denoised_sv.insert(0, 'date', dates)

    return denoised_sv, proxy, eig_values, eig_vectors.data[:, 0:proxy_number]


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
    (2011, GJI). The SV residuals of the noisy component for all observatories
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
    noisy_direction = eig_vectors[:, 0]
    proxy = 0

    if proxy_number == 1:
        proxy = projected_residuals[:, 0]
    elif proxy_number > 1:
        for direction in range(proxy_number):
            proxy = proxy + projected_residuals[:, direction]

    corrected_residuals = []

    for idx in range(len(proxy)):
        corrected_residuals.append(
            masked_residuals.data[idx, :] - proxy[idx] * noisy_direction)

    corrected_residuals = pd.DataFrame(corrected_residuals,
                                       columns=obs_data.columns)
    denoised_sv = pd.DataFrame(
        corrected_residuals.values + model_data.values,
        columns=obs_data.columns)
    denoised_sv.insert(0, 'date', dates)

    return denoised_sv, proxy, eig_values, eig_vectors.data[:, 0:proxy_number]


def detect_outliers(*, dates, signal, obs_name, window_length, threshold,
                    plot_fig=False):
    """Detect outliers in a time series and remove them.

    Use the following formula to detect outliers:

    (signal - median) > threshold * standard deviation

    The time series are long and highly variable so it is not appropriate to
    use single values of median and standard deviation (std) to represent the
    whole series. The function uses a running median and std to better
    characterise the series (the window length and threshold value are
    specified by the user).

    Args:
        dates (datetime.datetime): dates of the time series measurements.
        signal (array): array (or column from a pandas.DataFrame) containing
            the time series of interest.
        obs_name (str): states the component of interest and the three digit
           IAGA observatory name.
        window_length (int): number of months over which to take the running
            median and running standard deviation.
        threshold (float): the threshold value used in the above criterion.
            Typical values would be between two and three (so that the
            difference between the data and median is twice (or three times)
            the standard deviation).

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
    signal_temp = signal_temp.ffill(limit=window_length / 2 + 1).bfill()
    # calculate the running median and standard deviation
    running_median = signal_temp.rolling(window=window_length,
                                         center=True).median().bfill().ffill()
    running_std = signal_temp.rolling(window=window_length,
                                      center=True).std().bfill().ffill()
    # Identify outliers as (signal - median) > threshold * std
    threshold_value = (signal_temp - running_median).abs()
    difference = threshold_value - threshold * running_std.abs()
    outliers = signal_temp[difference > 0]

    # Plot the outliers and original time series if required
    if plot_fig is True:
        svplots.plot_outliers(dates=dates, obs_name=obs_name, signal=signal,
                              outliers=outliers)
    # Set the outliers to nan
    signal[difference[obs_name] > 0] = np.nan

    return signal

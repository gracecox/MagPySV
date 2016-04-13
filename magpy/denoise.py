# -*- coding: utf-8 -*-
"""Module containing functions to remove external signal from geomagnetic data.

Part of the MagPy package for geomagnetic data analysis. This module provides
various functions to denoise geomagnetic data by performing principal component
analysis.

Copyright (C) 2016  Grace Cox

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import Imputer


def eigenvalue_analysis(*, dates, obs_data, model_data, residuals,
                        proxy_number=1):
    """Perform principal component analysis (PCA) on secular variation
    residuals (the difference between the observed SV and that predicted by a
    geomagnetic field model) calculated from annual differences of monthly
    means at several observatories. Uses the imputer from sklearn.preprocessing
    to fill in missing data points and calculates the eigenvalues/vectors of
    the (3nx3n) covariance matrix for n observatories. The residuals are
    rotated into the eigendirections and denoised using the method detailed in
    Wardinski & Holme (2011). The SV residuals of the noisy component for all
    observatories combined are used as a proxy for the unmodelled external
    signal.The denoised data are then rotated back into geographic coordinates.
    Principal component analysis - find the eigenvalues and eigenvectors of
    the covariance matrix of the residuals. Project the SV residuals into the
    eigenvector directions. The pca algorithm outputs the eigenvalues sorted
    from largest to smallest, so the corresponding eigenvector matrix has the
    'noisy' direction in the first column and the 'clean' direction in the
    final column.

    Smallest eigenvalue: 'quiet' direction

    Largest eiegenvalue: 'noisy' direction

    Args:
        **dates (datetime series): dates of the time series measurements.
        **obs_data (dataframe): dataframe containing columns for monthly/annual
            means of the X, Y and Z components of the secular variation at the
            observatories of interest.
        **model_data (dataframe): dataframe containing columns for field model
            prediction of the X, Y and Z components of the secular variation at
            the same observatories as in obs_data.
        **residuals (dataframe): dataframe containing the residuals of the SV
            (difference between the observed data and model prediction).
        **proxy_number (int): the number of 'noisy' used to create the proxy
            for theexternal signal removal. Default value is 1 (only the
            residual in the direction of the largest eigenvalue is used). Using
            n directions means that proxy is the sum of the SV residuals in the
            n noisiest eigendirections.
    Returns:
        denoised_sv (dataframe): dataframe with datetime objects in the first
        column and columns for the denoised X, Y and Z SV components at each of
        the observatories for which data were provided.
    """

    # Fill in missing SV values (indicated as NaN in the data files)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
    imputed_residuals = imp.fit_transform(residuals)

    pca = sklearnPCA()
    projected_residuals = pca.fit_transform(imputed_residuals)
    eig_values = pca.explained_variance_
    eig_vectors = pca.components_

    # Use the method of Wardinski & Holme (2011) to remove unmodelled external
    # signal in the SV residuals. The variable 'proxy' contains the noisy
    # component residual for all observatories combined
    noisy_direction = eig_vectors[0, :]
    proxy = projected_residuals[:, 0]

    if proxy_number > 1:
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

    return denoised_sv

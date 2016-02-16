# -*- coding: utf-8 -*-
"""
Perform principal component analysis (PCA) on secular variation 
residuals (the difference between the observed SV and that predicted by a 
geomagnetic field model) calculated from annual differences of monthly 
means at several observatories. Uses the imputer from sklearn.preprocessing to 
fill in missing data points and calculates the eigenvalues/vectors of the
(3nx3n) covariance matrix for n observatories. The residuals are rotated
into the eigendirections and denoised using the method detailed in
Wardinski & Holme (2011). The SV residuals of the noisy component for all 
observatories combined are used as a proxy for the unmodelled external signal. 
The denoised data are then rotated back into geographic coordinates

@author: Grace
"""

import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import Imputer

def eigenvalue_analysis(dates, obs_data, model_data, residuals,proxy_number):
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
    
    return denoised_sv
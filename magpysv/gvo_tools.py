# -*- coding: utf-8 -*-
#    Copyright (C) 2020  Grace Cox and Will Brown (British Geological Survey)
#
#    Released under the MIT license, a copy of which is located at the root of
#    this project.
"""Module containing functions to apply the Principal Component Analysis (PCA)
denoising method to Geomagnetic Virtual Observatories (GVOs). These functions
were used in the European Space Agency's Swarm DISC GVO project and are called
in the PCA part of the operational script (maintained at the British Geological
Survey) that produces Swarm GVOs on a regular basis.

Details of the European Space Agency's Swarm DISC GVO project are at:
https://www.space.dtu.dk/english/research/projects/project-descriptions/geomagnetic-virtual-observatories

Note that some of the input/output functions use data file formats that were
used internally within the project.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import scipy as sp
import glob
import os
import aacgmv2
import cartopy
from chaosmagpy import load_CHAOS_shcfile
from chaosmagpy.data_utils import mjd_to_dyear, mjd2000
from magpysv.tools import calculate_sv


def load_vo_txt_raw(*, fname, sampling='1M'):
    """Load internal Swarm DISC GVO files and return raw contents.

    Files contain co-latitude in degrees (theta), longitude in degrees (phi)
    year, month, time (MJD2000), radius in km (r), the magnetic field
    components (Br, Btheta, Bphi), where Bphi is equivalent to the Y component,
    errors for the magnetic field in nT (sigma_r, sigma_theta,
    sigma_phi) and the number of data used for produce the GVO (N_data)

    Args:
        fname (str): filename to GVO text file
        sampling (str): sampling rate of the GVOs. Should be either '1M' for
            monthly or '4M' for four-monthly. Defaults to '1M'

    Returns:
        df (pandas.Dataframe): dataframe containing raw contents of the file
    """
    # Set the day of month for time series depending on the MF sampling rate
    if sampling == '1M':
        day = 15
    elif sampling == '4M':
        day = 1

    # Positions given in degrees - co-latitude (0 to 180), longitude (
    df = pd.read_csv(fname, sep="\s+", header=14,
                     names=["theta", "phi", "Year", "Month", "Time", "r",
                            "Br", "Btheta", "Y", "sigma_r", "sigma_theta",
                            "sigma_phi", "N_data"], usecols=range(13))

    df["mjd2000"] = mjd2000(df["Year"], df["Month"], day)
    df["dyear"] = mjd_to_dyear(df["mjd2000"], leap_year=True)
    df["X"] = -df["Btheta"]  # -theta component
    df["Z"] = -df["Br"]  # -radial component
    df.drop(columns=["Btheta", "Br"], inplace=True)
    # To 00:00 on 1st or 15th each month
    # Multiplication by 10000 and 100 are needed to convert to datetime
    # (see documentation for pandas.datetime)
    df["date"] = pd.to_datetime(df["Year"]*10000+df["Month"]*100+day,
                                format="%Y%m%d")

    return df


def load_vo_txt(*, fname, sampling='1M', sv_spacing):
    """Load interim GVOs output in the Swarm DISC project (internal files).

    Load the contents of internal Swarm DISC GVO files and return time series
    of magnetic field, secular variation, uncertainties (sigmas) and positions

    Args:
        fname (str): filename to GVO text file
        sampling (str): sampling rate of the GVOs. Should be either '1M' for
            monthly or '4M' for four-monthly. Defaults to '1M'
        sv_spacing (int): specifies how to calculate SV for monthly ('1M') MF
            data. Use 12 for annual differences of monthly means or 1 for
            monthly first differences

    Returns:
        (tuple): tuple containing:

        - df_mf_pos (pandas.Dataframe): dataframe containing the MF timeseries\
         at each position
        - df_sv_pos (pandas.Dataframe): dateframe containing the SV timeseries\
         at each position
        - positions (list): list of GVO positions [r, theta, phi] in km,\
         degrees
        - dates (list): list of GVO timeseries dates in decimal years
        - df_sigma (pandas.Dataframe): dataframe containing data uncertainties\
         and data numbers ['sigma_r','sigma_theta','sigma_phi','N_data'] in nT
    """
    # Load GVO MF and sigmas from txt file
    df_mf = load_vo_txt_raw(fname=fname, sampling=sampling)
    df_sigma = pd.DataFrame({'sigma_r': df_mf['sigma_r'],
                             'sigma_theta': df_mf['sigma_theta'],
                             'sigma_phi': df_mf['sigma_phi'],
                             'N_data': df_mf['N_data']})

    # Rearrange to dataframe of timeseries at each GVO location
    dates = df_mf["date"].unique()  # timeseries for all GVO
    # dates_mjd2000 = df_mf["mjd2000"].unique()

    # Array of GVO positions [t, p, r]
    positions = df_mf.drop_duplicates(subset=["theta", "phi"])
    positions = positions[["theta", "phi"]].to_numpy()
    positions = np.column_stack((positions,
                                 np.repeat(df_mf["r"].dropna().unique(),
                                           len(positions))))  # txt file

    df_mf_pos = pd.DataFrame({"date": dates})
    for pos in range(len(positions)):
        df_mf_pos["X"] = df_mf["X"].loc[(df_mf["theta"] == positions[pos, 0])
                                        & (df_mf["phi"]
                                           == positions[pos, 1])].values
        df_mf_pos["Y"] = df_mf["Y"].loc[(df_mf["theta"] == positions[pos, 0])
                                        & (df_mf["phi"]
                                           == positions[pos, 1])].values
        df_mf_pos["Z"] = df_mf["Z"].loc[(df_mf["theta"] == positions[pos, 0])
                                        & (df_mf["phi"]
                                           == positions[pos, 1])].values

        # Get SV, then rename SV columns
        if sampling == '1M':
            df_sv = calculate_sv(df_mf_pos, mean_spacing=sv_spacing)
        elif sampling == '4M':
            # Calculate SV as annual differences of four-monthly GVO MF and set
            # the SV date to be between the two MF dates
            df_sv = pd.DataFrame()
            df_sv['date'] = df_mf_pos['date'] - \
                pd.tseries.offsets.DateOffset(months=6)
            df_sv['dx'] = df_mf_pos['X'].diff(periods=3)
            df_sv['dy'] = df_mf_pos['Y'].diff(periods=3)
            df_sv['dz'] = df_mf_pos['Z'].diff(periods=3)
            df_sv.drop(df_sv.head(3).index, inplace=True)
        df_sv.reset_index(drop=True, inplace=True)
        if pos == 0:  # keep date and data columns in first instance
            df_sv_pos = df_sv
            df_sv_pos.rename(columns={"dx": "dx_"+str(pos).zfill(3),
                                      "dy": "dy_"+str(pos).zfill(3),
                                      "dz": "dz_"+str(pos).zfill(3)},
                             inplace=True)
        else:  # only keep data columns otherwise
            df_sv_pos[["dx_"+str(pos).zfill(3),
                       "dy_"+str(pos).zfill(3),
                       "dz_"+str(pos).zfill(3)]] = df_sv[["dx", "dy", "dz"]]

        # Rename MF columns starting from GVO 000
        df_mf_pos.rename(columns={"X": "X_"+str(pos).zfill(3),
                                  "Y": "Y_"+str(pos).zfill(3),
                                  "Z": "Z_"+str(pos).zfill(3)}, inplace=True)

    return df_mf_pos, df_sv_pos, positions, dates, df_sigma


def load_validation_vo_txt(*, fname, sv_spacing):
    """Read a text file containing monthly MF and SV data at a validation GVO.

    Validation GVO are located 490km above a ground observatory (these files
    were internal to the Swarm DISC GVO project).

    Args:
        fname (str): filename to validation GVO text file
        sv_spacing (int): specifies how to calculate SV for monthly ('1M') MF
            data. Use 12 for annual differences of monthly means or 1 for
            monthly first differences

    Returns:
        (tuple): tuple containing:

        - df_mf (pandas.Dataframe): dataframe containing MF ['date','X','Y',\
         'Z'] in nT
        - df_sv (pandas.Dataframe): dataframe containing ['date','dx','dy',
         dz'] in nT/year
        - position (list): array of position coordinates [theta,phi,r] in\
         degrees, km
        - obs (str): observatory name
        - df_sigma (pandas.Dataframe): dataframe containing data uncertainties\
         and data numbers ['sigma_r','sigma_theta','sigma_phi','N_data'] in nT
    """
    df_mf = load_vo_txt_raw(fname=fname, sampling='1M')
    # Pull the corresponding ground station name out of the file name
    obs = fname[-19:-15]
    df_sigma = pd.DataFrame({'sigma_r': df_mf['sigma_r'],
                             'sigma_theta': df_mf['sigma_theta'],
                             'sigma_phi': df_mf['sigma_phi'],
                             'N_data': df_mf['N_data']})
    radius = (6.3712+0.49)*1e3  # For GVOs at 490km above Earth's surface
    position = [df_mf.theta[0], df_mf.phi[0], radius]
    df_mf = df_mf[['date', 'X', 'Y', 'Z']]
    # Calculate SV
    df_sv = calculate_sv(df_mf, mean_spacing=sv_spacing)
    # Rename MF and SV columns to include GVO number
    df_mf.rename(columns={"X": "X_"+obs, "Y": "Y_"+obs, "Z": "Z_"+obs},
                 inplace=True)
    df_sv.rename(columns={"dx": "dx_"+obs, "dy": "dy_"+obs, "dz": "dz_"+obs},
                 inplace=True)

    return df_mf, df_sv, position, obs, df_sigma


def read_all_validation_txt(*, file_dir):
    """Load and concatenate all validation GVO dataframes.

    Produces dataframe of timeseries to match regular GVO dataframes. The
    validation GVOs are located at satellite altitude above a ground magnetic
    observatory and were used to validate the Swarm DISC GVO project. These
    internal data files were provided as one GVO per file, unlike the regular
    GVOs which had a single file for the entire GVO grid

    Args:
        file_dir (str): Directory path containing validation GVO files

    Returns:
        (tuple): tuple containing:

        - mf (pandas.Dataframe):
         dataframe containing MF data
        - sv (pandas.Dataframe):
         dataframe containing SV data
        - positions (list):
         list of all validation positions [r,theta,phi]
        - names (list):
         list of all associated GO position IAGA codes
        - sigmas (pandas.dataframe):
         dataframe containing data uncertainties and numbers
    """
    # Initialise the validation dataframes using first validation GVO file
    mf, sv, positions, names, sigma = load_validation_vo_txt(
        fname=sorted(glob.glob(file_dir))[0])
    positions = [positions]
    names = [names]
    # Loop over the remaining files and append to the first
    for fname in sorted(glob.glob(file_dir))[1:]:
        mf_single, sv_single, position, name, sigma_single = \
            load_validation_vo_txt(fname=fname)
        mf = pd.concat([mf, mf_single.drop(columns=['date'])], axis=1)
        sv = pd.concat([sv, sv_single.drop(columns=['date'])], axis=1)
        sigma = pd.concat([sigma, sigma_single], axis=1)
        positions.extend([position])
        names.append(name)
    positions = np.array(positions)
    return mf, sv, positions, names, sigma


def select_validation_sv_by_index(*, vo_sv, model_sv, idx, validation_names):
    """Extract validation GVO SV and core model SV for specified locations.

    Validation GVOs are at 490km altitude above a ground magnetic observatory
    (used for validating the Swarm DISC GVO project). The validation GVO name
    is set as the IAGA ground station name.

    Model SV is converted from [Br,Btheta,Bphi] input to [X,Y,Z] output.

    Args:
        vo_sv (pandas.Dataframe): GVO SV timeseries for all validation GVOs
        model_sv (tuple): core model SV timeseries for all validation GVOs
            vo_sv
        idx (list): indices required in list of all validation GVOs (i.e.
            the GVOs located in a particular magnetic region)
        validation_names (list): list of all validation GVO names

    Returns:
        (tuple): tuple containing:

        - vo_sv_selected (pandas.Dataframe):
         columns of vo_sv matching given validation names
        - model_sv_selected (pandas.Dataframe):
         columns of model_sv matching given idx
    """
    # Extract relevant input for validation GVOs and model SV prediction
    vo_sv_selected = pd.DataFrame({'date': vo_sv.date.values}).reset_index(
        drop=True)
    model_sv_selected = pd.DataFrame({'date': vo_sv.date.values})
    col_names = ['date']
    stacked = np.array([])
    for position in sorted(idx):
        # Get the name of the validation GVO matching the current index and
        # extract data
        col_names.extend(vo_sv.filter(
            regex=validation_names[position]).columns)
        vo_sv_selected = pd.concat([vo_sv_selected, vo_sv.filter(
            regex=validation_names[position]).reset_index(drop=True)], axis=1,
            ignore_index=True)
        # Extract model SV predictions for the same validation GVO
        stacked = np.vstack([stacked, -model_sv[1][position]]) \
            if stacked.size else -model_sv[1][position]
        stacked = np.vstack([stacked, model_sv[2][position]])
        stacked = np.vstack([stacked, -model_sv[0][position]])
    # Rename the columns
    vo_sv_selected.columns = col_names
    model_sv_selected = pd.concat([model_sv_selected,
                                   pd.DataFrame(stacked.transpose())], axis=1)
    model_sv_selected.columns = vo_sv_selected.columns
    return vo_sv_selected, model_sv_selected


def calculate_magnetic_latitudes(*, dates, positions):
    """Calculate mean magnetic latitude for GVO series.

    Note that some low latitudes are not defined in the AAGCM2 model, which
    produces a runtime warning.
    Calculates magnetic latitude for each time sample and returns the mean
    magnetic latitude over the series.

    Args:
        dates (list): list of datetimes for GVO timeseries
        positions (list): list of GVO positions for which to return geomagnetic
            latitudes [theta,phi,r] in degrees, km

    Returns:
        mlat_means (list):
            list of mean magnetic latitudes, one for each given position
    """
    mlat = np.full([len(positions), len(dates)], np.nan)
    for idx_date, date in enumerate(pd.to_datetime(dates)):
        for idx_pos, pos in enumerate(positions):
            mlat[idx_pos, idx_date], _, _ = \
                aacgmv2.get_aacgm_coord(90-pos[0], pos[1], pos[2]-6371.2,
                                        date, method="GEOCENTRIC")
    # AACGM not defined at various locations around geomag equator, returns NaN
    # Fill the NaN with zero, assuming we want to select these locations as low
    # magnetic latitude
    mlat[np.isnan(mlat)] = 0.0
    # Calculate the mean magnetic latitude over the time series
    mlat_means = np.nanmean(mlat, 1)
    return mlat_means


def calculate_model_sv(*, model_file, vo_sv, positions):
    """Calculate reference field model SV at given positions.

    Model max spherical harmonic degree set to 13.

    Args:
        model_file (str): path to model .shc format file with coefficients in
            nT and time in decimal years
        vo_sv (pandas.Dataframe): dataframe of GVO SV series
        positions (list): list of GVO positions [theta,phi,r] in degrees, km

    Returns:
        dB_rtp_core (tuple):
            model core field SV at times of vo_sv in locations given in positions,
            as Br, Btheta, Bphi in nT/yr

    """
    # Calculate model SV predictions at each GVO
    model = load_CHAOS_shcfile(model_file)

    # Datetime to MJD2000 for SV times
    date_mjd2000 = vo_sv["date"] - dt.datetime(2000, 1, 1)
    date_mjd2000 = date_mjd2000.dt.total_seconds() / (24*3600)

    theta = positions[:, 0]
    phi = positions[:, 1]
    radius = positions[:, 2]

    # Reshape to use NumPy broadcasting
    time = np.reshape(date_mjd2000.to_numpy(), (1, -1))  # 1 x Nt
    radius = np.reshape(radius, (-1, 1))  # Np x 1
    theta = np.reshape(theta, (-1, 1))  # Np x 1
    phi = np.reshape(phi, (-1, 1))  # Np x 1

    # Compute SV field components of shape 3xNpxNt
    # Despite the warning, chaosmagpy is correct (i.e. non-NaN/zero) at the
    # geographic poles, unlike the Matlab version of the CHAOS code
    dB_rtp_core = model.synth_values_tdep(time, radius, theta, phi,
                                          nmax=13, deriv=1)
    return dB_rtp_core


def correlation_with_indices(*, index_filename, pc_df):
    """Correlate principal component with SV of magnetic index read from file.

    SV of indices AE, Dst, PC, Em must be precalculated and written to file.
    Index files are only read here, not calculated. The SV for the magnetic
    indices should be the same temporal resolution as for the principal
    component (e.g. annual differences or first differences for monthly data)

    Args:
        index_filename (str): path to index SV file
        pc_df (pandas.Dataframe): principal component to correlate to index SV

    Returns:
        (tuple): tuple containing:

        - index_cols (pandas.Dataframe): SV of geomagnetic index merged to\
        pc_df times
        - coeffs (numpy.array): correlation coefficients between pc_df and \
        each column of index_cols
    """
    index_df = pd.read_csv(index_filename,
                           parse_dates=['date'], index_col=0)
    # Keep only data common to both time series
    merged = pd.merge(pc_df.dropna(), index_df.dropna(), on='date',
                      how='inner')

    index_cols = merged.filter(regex='AE|AO|AU|AL|Dst|PCN|PCS|Em',
                               axis=1).columns
    coeffs = []
    for index in index_cols:
        # Calculate correlation coefficient between principal component and
        # index
        coeff = np.corrcoef(merged.proxy, merged[index])
        coeffs.append(coeff.data[0, 1])
    return index_cols, coeffs


def calculate_residuals(*, vo_data, model_data):
    """Calculate SV residuals between GVO values and core model.

    Args:
        vo_data (pandas.Dataframe): GVO SV timeseries
        model_data (pandas.Dataframe): core field model SV timeseries matching
            vo_data

    Returns:
        residuals (pandas.Dataframe):
            residual SV between vo_data and model_data
    """
    # Drop the dates and then add back after residuals calculation
    dates = vo_data['date']
    vo_data.drop(vo_data.columns[[0]], axis=1, inplace=True)
    model_data.drop(model_data.columns[[0]], axis=1, inplace=True)
    # Calculate residuals as data minus model values
    residuals = pd.DataFrame(
            vo_data.values - model_data.values,
            columns=vo_data.columns)
    model_data.insert(0, 'date', dates)
    return residuals


def plot_eigenvalues_percentage(*, values, fig_size=(8, 6), font_size=12,
                                label_size=16, save_fig=False,
                                write_path=None):
    """Plot % variance explained by eigenvalues of residuals covariance matrix.

    Produces a plot of the eigenvalues as percentage variance explained by each
    principal component of SV residual covariance matrix. The largest
    eigenvalues represent the eigendirections (principal components) with the
    largest contributions to the SV residuals. Also plots the cumulative
    percentage variance.

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
    # Calculate the percentage variance from the eigenvalues
    plt.plot(100*values/np.sum(np.abs(values)), 'o-', linewidth=2,
             markersize=10, label=r'Variance $\lambda_i$')
    plt.plot(np.cumsum(100*values/np.sum(np.abs(values))), '^-', linewidth=2,
             markersize=10, label='Cum. sum variance')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.ylabel(r'Percentage variance explained by $\lambda_i$',
               fontsize=label_size)
    plt.xlabel(r'$i$', fontsize=label_size)
    plt.gca().legend(fontsize=font_size, frameon=False)
    plt.grid(True)

    if save_fig is True:
        # Create the output directory if it does not exist
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        fpath = os.path.join(write_path, 'eigenvalues.png')
        plt.savefig(fpath, bbox_inches='tight')
        plt.close()


def select_vo_sv_by_index(*, vo_sv, model_sv, idx):
    """Select GVO data and core model predictions SV by location index.

    Model SV is converted from [Br,Btheta,Bphi] input to [X,Y,Z] output.

    Args:
        vo_sv (pandas.Dataframe): GVO SV timeseries
        model_sv (tuple): core model SV timeseries at matching GVO positions to
            vo_sv
        idx (list): GVO position numbers to select from vo_sv

    Returns:
        (tuple): tuple containing:

        - vo_sv_selected (pandas.Dataframe):
         columns of vo_sv matching idx
        - model_sv_selected (pandas.Dataframe):
         columns of model_sv matching idx
    """
    # Initialise variables
    vo_sv_selected = pd.DataFrame({'date': vo_sv.date})
    model_sv_selected = pd.DataFrame({'date': vo_sv.date})
    stacked = np.array([])

    for position in sorted(idx):
        # Extract relevant GVO data
        vo_sv_selected = pd.concat([vo_sv_selected, vo_sv.filter(
            regex=str(position).zfill(3))], axis=1)
        # Extract relevant model data, converting to x, y and z components
        stacked = np.vstack([stacked, -model_sv[1][position]])\
            if stacked.size else -model_sv[1][position]
        stacked = np.vstack([stacked, model_sv[2][position]])
        stacked = np.vstack([stacked, -model_sv[0][position]])
    model_sv_selected = pd.concat([model_sv_selected,
                                   pd.DataFrame(stacked.transpose())], axis=1)
    model_sv_selected.columns = vo_sv_selected.columns
    return vo_sv_selected, model_sv_selected


def plot_comparison(*, dates, vo, vo_denoised, position, model_data,
                    fig_size=(15, 5), font_size=12, label_size=14,
                    save_fig=False, write_path=None, deriv=1):
    """Plot comparison of pre-and post-PCA denoising SV or MF at a single GVO.

    Produces a plot of the X, Y and Z components of the GVO data (SV or MF),
    the PCA denoised SV or MF and the field model prediction for a single GVO.

    Args:
        dates (datetime.datetime): dates of time series measurements
        vo (time series): X, Y and Z components of uncorrected SV or MF at a
            single VO location
        vo_denoised (time series): X, Y and Z components of the denoised SV or
            MF at the same VO location
        derive (int): specify whether the given data are MF (deriv=0) or SV
            (deriv=1). Defaults to 1
        model_data (time series): X, Y and Z components of the SV or MF
            predicted by a field model for the same location as the data.
        position (str): GVO location name
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    """
    # Choose labels for plots (either for MF or SV)
    if deriv == 0:
        components = ['X_', 'Y_', 'Z_']
        labels = [r'X (nT)', r'Y (nT)', r'Z (nT)']
    elif deriv == 1:
        components = ['dx_', 'dy_', 'dz_']
        labels = [r'dX/dt (nT/yr)', r'dY/dt (nT/yr)', r'dZ/dt (nT/yr)']
    colors = ['red', 'green', 'blue']
    fig, ax = plt.subplots(1, 3, sharex=True, figsize=fig_size)

    for i in range(0, len(components)):
        plt.subplot(1, 3, i+1)
        plt.plot(dates, vo.filter(regex=components[i]+str(position).zfill(3)),
                 color='grey', linestyle='--', linewidth=2)
        plt.plot(dates, vo_denoised.filter(
            regex=components[i]+str(position).zfill(3)), color=colors[i],
            linewidth=2)
        plt.plot(dates, model_data.filter(
            regex=components[i]+str(position).zfill(3)), color='black',
            linewidth=2)
        plt.ylabel(labels[i])
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
    fig.text(0.5, 0.01, 'Date', ha='center', fontsize=label_size)
    fig.text(0.05, 0.5, 'GVO ' + str(position), fontsize=label_size,
             rotation='vertical')
    if save_fig is True:
        # Create the output directory if it does not exist
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        fpath = os.path.join(write_path, 'GVO_'+str(position).zfill(3) +\
                             '.png')
        plt.savefig(fpath, bbox_inches='tight')
        plt.close()


def eigenvalue_analysis(*, dates, obs_data, model_data, residuals,
                        proxy_number=1):
    """Remove external signal from SV data using principal Component Analysis.

    Perform principal component analysis (PCA) on secular variation
    residuals (the difference between the observed SV and that predicted by a
    geomagnetic field model) calculated from annual differences of monthly
    means at several observatories. Uses masked arrays to discount missing data
    points and calculates the Principal Components (PCs, also known as the
    eigenvalues/vectors) of the (3nx3n) covariance matrix for n observatories.
    The residuals are rotated into the eigendirections and denoised using the
    method detailed in Cox et al (2018, Geochemistry, Geophysics, Geosystems
    (https://doi.org/10.1029/2018GC007714). The SV residuals projected into the
    dominant PCs component are used as a proxy for the unmodelled contaminating
    signals such as external magnetic fields and, for satellite data, local
    time sampling biases arising from orbital dynamics. The denoised data are
    then rotated back into geographic coordinates. The PCA algorithm outputs
    the eigenvalues sorted from largest to smallest (absolute values), so the
    corresponding eigenvector matrix has the dominant PC in the first column
    (accounts for the most variance in the residuals).

    This algorithm masks missing data so that they are not taken into account
    during the PCA. Missing values are not infilled or estimated, so NaN
    values in the input dataframe are given as NaN values in the output.

    Args:
        dates (datetime.datetime): dates of the time series measurements.
        obs_data (pandas.DataFrame): dataframe containing columns for
            monthly/annual means of the X, Y and Z components of the secular
            variation at the magnetic observatories of interest.
        model_data (pandas.DataFrame): dataframe containing columns for field
            model prediction of the X, Y and Z components of the secular
            variation at the same observatories as in obs_data.
        residuals (pandas.DataFrame): dataframe containing the SV residuals
            (difference between the observed data and model prediction).
        proxy_number (int): the number of Principal Components used to create
            the proxy for the external signal removal. Default value is 1 (only
            the residual in the direction of the largest eigenvalue (i.e. the
            dominant PC "PC0") is used). Using m directions means that proxy
            is the sum of the SV residuals in the m largest PCs.

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
         in the two largest PCs are used as the proxy for external
         signal, then these two eigenvectors are returned.
        - projected_residuals (*array*):
         SV residuals rotated into the PCs (eigendirections).
        - corrected_residuals (*array*):
         SV residuals after the denoising process.
        - covariance_matrix (*array*):
         the residuals covariance matrix
    """
    # Create a masked version of the residuals array so that we can perform the
    # PCA ignoring all nan values
    masked_residuals = np.ma.array(residuals, mask=np.isnan(residuals))

    # Calculate the covariance matrix of the masked residuals array
    covariance_matrix = np.ma.cov(masked_residuals, rowvar=False,
                                  allow_masked=True)
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eig_values, eig_vectors = np.linalg.eigh(covariance_matrix)
    # Sort the eigenvalues in decreasing order
    idx = np.argsort(np.abs(eig_values))[::-1]
    eig_values = eig_values[idx]
    # Sort the eigenvectors according to the same index
    eig_vectors = eig_vectors[:, idx]

    # Project the residuals onto the eigenvectors
    projected_residuals = np.ma.dot(masked_residuals, eig_vectors)

    # Use the method of Cox et al (2018) to remove unmodelled external
    # signal in the SV residuals. The variable 'proxy' contains the sum
    # of the SV residuals projected into the number of dominant PCs
    # specified by proxy_number

    corrected_residuals = []

    if proxy_number == 1:
        noisy_direction = eig_vectors[:, 0]
        proxy = projected_residuals[:, 0]
        for idx in range(len(proxy)):
            corrected_residuals.append(
                masked_residuals.data[idx, :] - proxy[idx] * noisy_direction)
    # Apply the denoising algorithm to each of the PCs specified
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
    # Re-form the SV from the denoised residuals
    denoised_sv = pd.DataFrame(
        corrected_residuals.values + model_data.values,
        columns=obs_data.columns)

    denoised_sv.insert(0, 'date', dates)

    return denoised_sv, proxy, np.abs(eig_values), eig_vectors,\
        projected_residuals, corrected_residuals.astype('float'),\
        covariance_matrix


def format_and_write(*, fname, header, data, df_sigma, positions):
    """Write denoised GVO data to output files.

    The files are written out in the same format as the internal files used in
    the Swarm DISC GVO project. See `load_vo_txt_raw` for a description

    Args:
        fname (str): name of output file
        header (str): header text to write to file
        data (pandas.Dataframe): denoised MF or SV GVO data to write
        df_sigma (pandas.Dataframe): associated uncertainties for data
        positions (list): data positions [theta,phi,r]
    """
    # Reorder to all positions at each timesample a la .txt file format
    row_list = []
    for idx_date, date_val in enumerate(data['date']):
        for idx_pos, pos in enumerate(positions):
            dict_row = {}
            if not df_sigma.empty:  # assume MF has uncertainties
                dict_row.update(
                    {'theta': pos[0],
                     'phi': pos[1],
                     'Year': date_val.year,
                     'Month': date_val.month,
                     'Time': mjd2000(date_val),
                     'r': pos[2],
                     # -Z = B_r
                     'B_r': -data.iloc[idx_date].filter(
                         regex='Z_' + str(idx_pos).zfill(3)).values[0],
                     # -X = B_theta
                     'B_theta': -data.iloc[idx_date].filter(
                         regex='X_' + str(idx_pos).zfill(3)).values[0],
                     # Y = B_phi
                     'B_phi': data.iloc[idx_date].filter(
                         regex='Y_' + str(idx_pos).zfill(3)).values[0]
                     })
            else:  # assume SV has empty df instead of uncertainties
                dict_row.update(
                    {'theta': pos[0],
                     'phi': pos[1],
                     'Year': date_val.year,
                     'Month': date_val.month,
                     'Time': mjd2000(date_val),
                     'r': pos[2],
                     # -Z = B_r
                     'B_r': -data.iloc[idx_date].filter(
                         regex='dz_' + str(idx_pos).zfill(3)).values[0],
                     # -X = B_theta
                     'B_theta': -data.iloc[idx_date].filter(
                         regex='dx_' + str(idx_pos).zfill(3)).values[0],
                     # Y = B_phi
                     'B_phi': data.iloc[idx_date].filter(
                         regex='dy_' + str(idx_pos).zfill(3)).values[0]
                     })
            row_list.append(dict_row)
    data_to_write = pd.DataFrame(row_list)
    if not df_sigma.empty:
        data_to_write = pd.concat([data_to_write.reset_index(drop=True),
                                   df_sigma.reset_index(drop=True)], axis=1)

    with open(fname, 'w') as file:
        file.write(header)
        # data_to_write.to_csv(fname, na_rep='NaN',
        #                      index=False, header=False,
        #                      mode='a', sep='\t')
        for idx, row in data_to_write.iterrows():
            if not df_sigma.empty:
                file.write((
                    '%9.5f%13.5f%9d%5d%14.4f%12.2f%15.5f%15.5f%15.5f%10.4f%10.4f%10.4f%12.0f\n')
                    % (row['theta'], row['phi'], row['Year'], row['Month'],
                       row['Time'], row['r'], row['B_r'], row['B_theta'],
                       row['B_phi'], row['sigma_r'], row['sigma_theta'],
                       row['sigma_phi'], row['N_data']))
            else:
                file.write((
                    '%9.5f%13.5f%9d%5d%14.4f%12.2f%15.5f%15.5f%15.5f\n')
                    % (row['theta'], row['phi'], row['Year'], row['Month'],
                       row['Time'], row['r'], row['B_r'], row['B_theta'],
                       row['B_phi']))


def create_headers(*, magnetic_regions, model_name):
    """Create output file header strings (format internal to GVO DISC project)
    Args:
        magnetic_regions (dict): dict containing denoising region latitude
            limits, region name, and number of PC removed
        model_name (str): name of model used to detrend SV

    Returns:
        (tuple): tuple containing:

        - header_mf (str):
         MF GVO file header text
        - header_sv (str):
         SV GVO file header text

    """
    date_str = dt.datetime.now().strftime("%d-%b-%Y")

    header_mf = (f"% Geomagnetic Virtual Observatory Model, file created on: {date_str}\n"
              "% PID_OBA_SUB\n"
              "% Grid solution: EQ\n"
              "% Swarm data used\n"
              "% Data time used: all\n"
              "% Include external field correction: yes\n"
              "% Crustal field corrections used\n"
              "% Potential spatial degree: cubic\n"
              "% Search radius: 700\n"
              "% Target point altitude: 490\n"
              "% Inversion limit: 30\n"
              "% \n"
              "% PCA:\n"
              f"% SV detrended using {model_name}\n"
              "% QD lat min | QD lat max | # PC removed\n"
              f"% {magnetic_regions['1']['min_mag_lat']}  {magnetic_regions['1']['max_mag_lat']} {magnetic_regions['1']['proxy_number']}\n"
              f"% {magnetic_regions['2']['min_mag_lat']}  {magnetic_regions['2']['max_mag_lat']} {magnetic_regions['2']['proxy_number']}\n"
              f"% {magnetic_regions['3']['min_mag_lat']}  {magnetic_regions['3']['max_mag_lat']} {magnetic_regions['3']['proxy_number']}\n"
              f"% {magnetic_regions['4']['min_mag_lat']}  {magnetic_regions['4']['max_mag_lat']} {magnetic_regions['4']['proxy_number']}\n"
              f"% {magnetic_regions['5']['min_mag_lat']}  {magnetic_regions['5']['max_mag_lat']} {magnetic_regions['5']['proxy_number']}\n"
              "% \n"
              "% theta    |    phi    |  Year  Month |   Time       |     r    |    B_r         B_theta       B_phi         | sigma_r  sigma_theta  sigma_phi  | N_{data}  |\n"
              "% [deg]    |   [deg]   |              |  [mjd2000]   |    [km]  |          Predicted field - [nT]            |       Estimated error [nT]       |  # data   |\n"
              "% \n")

    header_sv = (f"% Geomagnetic Virtual Observatory Model, file created on: {date_str}\n"
              "% PID_OBA_SUB\n"
              "% Grid solution: EQ\n"
              "% Swarm data used\n"
              "% Data time used: all\n"
              "% Include external field correction: yes\n"
              "% Crustal field corrections used\n"
              "% Potential spatial degree: cubic\n"
              "% Search radius: 700\n"
              "% Target point altitude: 490\n"
              "% Inversion limit: 30\n"
              "% \n"
              "% PCA:\n"
              f"% SV detrended using {model_name}\n"
              "% QD lat min | QD lat max | # PC removed\n"
              f"% {magnetic_regions['1']['min_mag_lat']}  {magnetic_regions['1']['max_mag_lat']} {magnetic_regions['1']['proxy_number']}\n"
              f"% {magnetic_regions['2']['min_mag_lat']}  {magnetic_regions['2']['max_mag_lat']} {magnetic_regions['2']['proxy_number']}\n"
              f"% {magnetic_regions['3']['min_mag_lat']}  {magnetic_regions['3']['max_mag_lat']} {magnetic_regions['3']['proxy_number']}\n"
              f"% {magnetic_regions['4']['min_mag_lat']}  {magnetic_regions['4']['max_mag_lat']} {magnetic_regions['4']['proxy_number']}\n"
              f"% {magnetic_regions['5']['min_mag_lat']}  {magnetic_regions['5']['max_mag_lat']} {magnetic_regions['5']['proxy_number']}\n"
              "% \n"
              "% theta    |    phi    |  Year  Month |   Time       |     r    |    dB_r         dB_theta       dB_phi      | sigma_r  sigma_theta  sigma_phi  | N_{data}  |\n"
              "% [deg]    |   [deg]   |              |  [mjd2000]   |    [km]  |          Predicted field - [nT/yr]         |       Estimated error [nT/yr]    |  # data   |\n"
              "% \n")
    return header_mf, header_sv


def plot_residuals_dft_all(*, projected_residuals, dates, sampling,
                           fig_size=(10, 8),
                           font_size=12, label_size=16,
                           save_fig=False, write_path=None):
    """Compare the DFTs of the projected residuals with each other.

    Calculates the Discrete Fourier Transform (DFT) of the projected residuals
    in each Principal Component given and plots it alongside the projected
    residuals themselves. Produces a separate figure per PC. The length of the
    time series are padded with zeroes up to the next power of two.

    Args:
        dates (datetime.datetime): dates of time series measurements.
        projected_residuals (time series): difference between modelled and
            SV rotated into the PCs obtained during denoising
            (PC). The proxy for unmodelled contaminating
            signal is the residual projected in the dominant PC(s).
        sampling (int): sampling rate of GVOs in months. Typically 1 or 4
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        plot_legend (bool): option to include a legend on the plot. Defaults to
            True.
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    """
    sampling_period = sampling / 12.0   # Sampling time in years

    sample_length = int(pow(2, np.ceil(np.log2(projected_residuals.shape[0]))))

    # Iterate over the eigendirections (PCs) and produce a figure for each
    for direction in range(projected_residuals.shape[1]):
        residual_dft = sp.fft(projected_residuals[:, direction], sample_length)
        freq = np.linspace(0.0, 1.0 / (2.0 * sampling_period),
                           num=(sample_length // 2))

        residual_power = (2.0 / sample_length) * np.abs(
            residual_dft[:sample_length // 2])
        plt.subplots(nrows=1, ncols=2, figsize=fig_size)[1]
        plt.subplot(2, 1, 1)
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
        plt.ylabel('DFT PC%d' % direction, fontsize=label_size)

        if save_fig is True:
            # Create the output directory if it does not exist
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            fpath = os.path.join(write_path,
                                 'dft_eigendirection%03d.png' % direction)
            plt.savefig(fpath, bbox_inches='tight')
            plt.close()


def plot_eigenvectors(*, obs_names, eigenvecs, fig_size=(8, 6), font_size=12,
                      label_size=16, save_fig=False, write_path=None):
    """Plot eigenvectors (PCs) of the covariance matrix of SV residuals.

    Produces a plot of the eigenvectors (PCs) corresponding to the n largest
    eigenvalues of the covariance matrix obtained during PCA of SV residuals,
    where n is the number of PCs used as a proxy for unmodelled
    contaminating signals. The n eigenvectors corresponding to the n largest
    eigenvalues represent the directions with the largest contribution
    to the residuals (the "dominant PCs")

    Args:
        obs_names (list): list of observatory names given as three digit IAGA
            codes or GVO numbers.
        eigenvecs (array): the eigenvalues obtained from the principal
        component analysis of the SV residuals.
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        font_size (int): font size for axes. Defaults to 12 pt.
        label_size (int): font size for axis labels. Defaults to 16 pt.
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    """
    # Loop over directions and plot each eigenvector on a separate figure
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
        plt.legend(['$r$ direction', r'$\theta$ direction',
                    '$\phi$ direction'], loc='upper right', frameon=False,
                   fontsize=label_size)
        plt.xlabel('Location', fontsize=label_size)

        if save_fig is True:
            # Create the output directory if it does not exist
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            fpath = os.path.join(write_path,
                                 'eigendirection%03d.png' % direction)
            plt.savefig(fpath, bbox_inches='tight')
            plt.close()


def plot_eigenvector_map(*, eigenvecs, positions, idx, marker_size=80,
                         linewidth=0.2, fig_size=(10, 10),
                         save_fig=False, write_path=None):
    """Plot map of eigenvector (PC) of SV residuals covariance matrix.

    Produces a map of the contribution to a single eigenvector (PC)
    of the covariance matrix obtained during PCA of SV residuals.

    The eigenvector is split into (X, Y, Z) contributions from each GVO
    considered in the PCA, and these are mapped onto a Maxwell colour triangle
    so that these individual (X, Y, Z) contributions to the eigenvector are
    represented by an (R, G, B) triplet. The contribution at each GVO is
    plotted as a colour coded circle on a map. E.g. a purely X vector is red, a
    purely Y vector is green and a purely Z vector is blue. Combinations of
    these colours represent the relative contributions of X, Y and Z.

    Args:
        eigenvecs (array): the eigenvalues obtained from the principal
        component analysis of the SV residuals.
        positions (list): all GVO positions [theta,phi,r] in degrees, km
        idx (list): indices of the desired GVO locations in the positions list
        marker_size (float: marker size on map. Defaults to 80
        linewidth (float): width of marker border. Defaults to 0.2
        fig_size (array): figure size in inches. Defaults to 8 inches by 6
            inches.
        save_fig (bool): option to save figure. Defaults to False.
        write_path (str): output path for figure if saved.
    """
    # Loop over directions and plot each eigenvector on a separate figure

    # Extract x, y and z components
    for direction in range(eigenvecs.shape[1]):
        x_values = np.abs(eigenvecs[::3, direction])
        y_values = np.abs(eigenvecs[1::3, direction])
        z_values = np.abs(eigenvecs[2::3, direction])
        # Map setup
        plt.figure(figsize=fig_size)
        ax = plt.axes(projection=cartopy.crs.PlateCarree())
        ax.set_global()
        ax.add_feature(cartopy.feature.LAND, color='lightgray')

        # Loop over each GVO location used in PCA and plot its contribution
        # to the current eigenvector (PC)
        for i, location in enumerate(idx):
            # Get locations for the selected GVO indices
            lon = positions[location, 1]
            lat = 90 - positions[location, 0]
            # Plot the colour coded symbol
            total = x_values[i] + y_values[i] + z_values[i]
            plt.scatter(lon, lat, marker='o', color=[x_values[i]/total,
                        y_values[i]/total, z_values[i]/total],
                        s=marker_size, linewidth=linewidth, edgecolor='k')

        if save_fig is True:
            # Create the output directory if it does not exist
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            fpath = os.path.join(write_path, 'map_PC%03d.png' % direction)
            plt.savefig(fpath, bbox_inches='tight')
            plt.close()


def denoised_sv_to_mf(*, vo_mf, denoised_sv):
    """Integrate PCA denoised annual differences SV to get denoised monthly MF.

    Re-integrate monthly MF from cleaned SV (orginally calculated as annual
    differences of monthly field values), using initial year of raw MF samples
    to level denoised MF. Denoised MF will start 1 year later than raw MF as a
    result.

    Args:
        vo_mf (pandas.Dataframe): dataframe containing raw GVO monthly MF
        denoised_sv (pandas.Dataframe): dataframe containing denoised GVO SV

    Returns:
        denoised_mf (pandas.Dataframe):
            dataframe containing re-integrated MF rom denoised SV
    """
    denoised_mf = pd.DataFrame(np.nan, index=vo_mf.index,
                               columns=vo_mf.columns)
    denoised_mf['date'] = vo_mf['date']
    # Get the first and last valid (non-NaN) indices of the time series
    # Index 1 means X_000 is used - chosen to avoid the date column
    first_valid = int(vo_mf.apply(pd.Series.first_valid_index)[1])
    last_valid = int(vo_mf.apply(pd.Series.last_valid_index)[1])
    #
    for (col_mf, col_sv) in zip(denoised_mf.columns[1:].sort_values(),
                                denoised_sv.columns[1:].sort_values()):
        # first non-NaN MF/SV row to last non-NaN SV row
        # Annual differences of monthly means formulation
        for idx_date in range(first_valid, last_valid+1):
            # First year is uncorrected MF, then denoised SV added to preceding
            # MF
            if idx_date == first_valid:
                denoised_mf[col_mf][idx_date:idx_date+12] = \
                    vo_mf[col_mf][idx_date:idx_date+12]
            elif idx_date >= first_valid + 12:
                denoised_mf[col_mf][idx_date] = \
                    denoised_mf[col_mf][idx_date-12] + \
                    denoised_sv[col_sv][idx_date-12]

    # Cut original MF values from denoised MF df
    denoised_mf.iloc[first_valid:first_valid+12, 1:] = np.nan

    return denoised_mf

"""
Module: laser_experiment.py
Description:
    This module provides functions to analyze the optimal laser separation parameters.
    In particular, the function `laser_separation_experiment` uses spectral interpolation,
    cross-validation with PLSRegression, and plots the RMSEP (root mean square error of prediction)
    against the laser shift (lambda_2 - lambda_1) to assess performance.
    
    Use this module to determine the best laser shift values for your simulation.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt

def laser_separation_experiment(w_cm, conv_2d_FSR0, gen_y, wavenumber_to_wavelength,
                                lambda_1_values=None, lambda_2_values=None,
                                png_path=None):
    """
    Perform the laser separation experiment analysis.
    
    Parameters
    ----------
    w_cm : array-like
        The wavenumber array.
    conv_2d_FSR0 : 2D numpy array
        The spectral data array.
    gen_y : array-like
        The response values.
    wavenumber_to_wavelength : function
        Function that converts wavenumbers to wavelengths given a reference lambda.
        Example call: wavelengths = wavenumber_to_wavelength(w_cm_subset, lambda_ref)
    lambda_1_values : list or array, optional
        List of lambda_1 values. Defaults to:
            [784.8, 784.6, 784.4, 784.2, 784, 783, 782, 781, 780, 779, 778]
    lambda_2_values : list or array, optional
        List of lambda_2 values. Defaults to: [785]
    png_path : str, optional
        File path to save the plot. If None, the plot is shown.
    
    Returns
    -------
    results : dict
        A dictionary where each key is a tuple (lambda_1, lambda_2) and each value is a dict
        with keys: 'r2', 'mse', 'mae', 'rmsep'.
    """
    if lambda_1_values is None:
        lambda_1_values = [784.8, 784.6, 784.4, 784.2, 784, 783, 782, 781, 780, 779, 778]
    if lambda_2_values is None:
        lambda_2_values = [785]
    
    # Select a subset of the wavenumber array (starting from index 34)
    w_cm_990 = w_cm[34:]
    gen_spec = conv_2d_FSR0     
    y_aa_ave = gen_y   
    results = {}
    
    for lambda_1 in lambda_1_values:
        for lambda_2 in lambda_2_values:
            # Convert wavenumbers to wavelengths for both lasers
            wavelengths_1 = wavenumber_to_wavelength(w_cm_990, lambda_1) 
            wavelengths_2 = wavenumber_to_wavelength(w_cm_990, lambda_2)
    
            # Initialize arrays to store spectra
            specs_1 = np.zeros(gen_spec.shape)
            specs_2 = np.zeros(gen_spec.shape)
            start = 0
            specs_1_cut = np.zeros((gen_spec.shape[0], gen_spec.shape[1] - start))
            specs_2_cut = np.zeros((gen_spec.shape[0], gen_spec.shape[1] - start))
    
            # Interpolate each spectrum
            for spectrum in range(gen_spec.shape[0]):
                f_2 = CubicSpline(wavelengths_2, gen_spec[spectrum, :], bc_type='natural')
                specs_1[spectrum, :] = gen_spec[spectrum, :]
                specs_2[spectrum, :] = f_2(wavelengths_1)
                specs_1_cut[spectrum, :] = specs_1[spectrum, :][start:]
                specs_2_cut[spectrum, :] = specs_2[spectrum, :][start:]
    
            # Calculate SERDS difference
            serds_cut = specs_1_cut - specs_2_cut
    
            # Prepare data for regression
            X_regression = np.array(serds_cut)
            y_m4 = y_aa_ave
    
            pls = PLSRegression(n_components=6)
            y_pred_cv_plsr = cross_val_predict(pls, X_regression, y_m4, cv=10)
    
            r2 = r2_score(y_m4, y_pred_cv_plsr)
            mse = mean_squared_error(y_m4, y_pred_cv_plsr)
            mae = mean_absolute_error(y_m4, y_pred_cv_plsr)
            rmsep = np.sqrt(mse)
    
            key = (lambda_1, lambda_2)
            results[key] = {'r2': r2, 'mse': mse, 'mae': mae, 'rmsep': rmsep}
    
    # Plot RMSEP vs laser shift
    rmsep_values = []
    laser_shifts = []
    for key, result in results.items():
        lambda_1, lambda_2 = key
        laser_shifts.append(lambda_2 - lambda_1)
        rmsep_values.append(result['rmsep'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(laser_shifts, rmsep_values, marker='o', linestyle='-', color='purple', label='RMSEP')
    plt.xlabel("Laser Shift (nm)", fontsize=16, fontname='Arial')
    plt.ylabel("Averaged RMSEP", fontsize=16, fontname='Arial')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    if png_path:
        plt.savefig(png_path, dpi=400, format='png')
        plt.close()
    else:
        plt.show()
    
    return results

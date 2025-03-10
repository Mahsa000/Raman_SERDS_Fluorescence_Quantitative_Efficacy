"""
Module: pre_processing.py
Description:
    This module provides functions for preprocessing Raman/SERDS spectra using two different
    background removal techniques: an iterative Discrete Wavelet Transform (DWT) filter and 
    an Asymmetric Least Squares (ALS) method.

    The DWT functions remove background components by zeroing selected wavelet coefficients,
    while the ALS functions estimate and subtract the background using an iterative smoothing
    algorithm. These functions operate on spectra stored in a results dictionary keyed by fluorescence levels.
"""

import numpy as np
import pywt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.stats import linregress

def norm_mean(x): 
    return (x - np.mean(x)) / np.std(x)

def dwt_multilevel_filter(spectrum, wavelet, scale, apprx_rm, low_cut, high_cut):
    """
    Applies a multilevel discrete wavelet transform (DWT) filter to remove background components from a spectrum.

    The function decomposes the input spectrum into wavelet coefficients, then zeros out selected coefficients.
    In particular, if `apprx_rm` is True, the approximation coefficients (level 0) are set to zero.
    Additionally, detail coefficients for levels up to `low_cut` and from level `scale - high_cut` onward are zeroed.

    Parameters
    ----------
    spectrum : np.ndarray
        1D array representing the spectrum.
    wavelet : str or pywt.Wavelet
        Wavelet name or object used for decomposition.
    scale : int
        Decomposition level (i.e., number of levels in the DWT).
    apprx_rm : bool
        Flag indicating whether to remove the approximation coefficients.
    low_cut : int
        Number of lower detail levels to remove.
    high_cut : int
        Number of higher detail levels to remove (counting from the highest level).

    Returns
    -------
    reconstructed : np.ndarray
        The reconstructed spectrum after filtering.
    """
    coeffs = pywt.wavedec(spectrum, wavelet, level=scale)
    for c_ix in range(len(coeffs)):
        if (c_ix == 0 and apprx_rm) or (c_ix > 0 and c_ix <= low_cut) or c_ix > scale - high_cut:
            coeffs[c_ix] = np.zeros_like(coeffs[c_ix])
    return pywt.waverec(coeffs, wavelet)


def dwt_iterative_bg_rm(spectrum, wavelet, scale, iterations):
    """
    Performs iterative background removal on a spectrum using a DWT-based filter.

    This function repeatedly applies the DWT multilevel filter to compute a background approximation,
    and then subtracts it from the original spectrum.

    Parameters
    ----------
    spectrum : np.ndarray
        1D array representing the original spectrum.
    wavelet : str or pywt.Wavelet
        Wavelet used for decomposition.
    scale : int
        Decomposition level.
    iterations : int
        Number of iterations to perform.

    Returns
    -------
    spectrum_bg_removed : np.ndarray
        The spectrum with the background removed.
    bg_approx : np.ndarray
        The final estimated background.
    """
    bg_approx = spectrum.copy()
    for _ in range(iterations):
        apprx_ix = dwt_multilevel_filter(bg_approx, wavelet, scale, apprx_rm=False, low_cut=0, high_cut=scale)
        if bg_approx.shape[0] % 2 == 0:
            bg_approx = np.minimum(bg_approx, apprx_ix)
        else:
            bg_approx = np.minimum(bg_approx, apprx_ix[:-1])
    spectrum_bg_removed = spectrum - bg_approx
    return spectrum_bg_removed, bg_approx


def preprocess_DWT(results_dict, initial_F_levels, scales, dwt_iterative_bg_rm_func):
    """
    Preprocesses Raman/SERDS spectra using DWT-based iterative background removal.

    For each fluorescence level in `initial_F_levels`, this function applies the iterative DWT background
    removal to both laser channels in the provided results dictionary. The SERDS difference is computed
    as the difference between the two background-removed channels.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing conventional spectra for each fluorescence level. Each entry should contain:
            - "specs_1_bg_noise_mean": np.ndarray (n_spectra x n_points)
            - "specs_2_bg_noise_mean": np.ndarray (n_spectra x n_points)
    initial_F_levels : list
        List of fluorescence levels (keys in results_dict) to process.
    scales : list
        List of scale parameters to use cyclically for DWT filtering.
    dwt_iterative_bg_rm_func : function
        Function implementing iterative DWT background removal (e.g., dwt_iterative_bg_rm).

    Returns
    -------
    concatenated_data_dict : dict
        Dictionary keyed by fluorescence level. Each value is a dictionary containing:
            - "specs_1_bg_noise_mean": background-removed spectra for laser 1
            - "specs_2_bg_noise_mean": background-removed spectra for laser 2
            - "serds_bg_noise_mean": difference between the two channels
    """
    concatenated_data_dict = {}
    wavelet = 'sym5'
    iterations = 50
    for F in initial_F_levels:
        target_results = results_dict.get(F)
        if target_results is not None:
            X_conv1 = target_results.get("specs_1_bg_noise_mean")
            X_conv2 = target_results.get("specs_2_bg_noise_mean")
            n_spectra, n_points = X_conv1.shape
            specs1_bg_removed = np.zeros((n_spectra, n_points))
            specs2_bg_removed = np.zeros((n_spectra, n_points))
            for i in range(n_spectra):
                scale = scales[i % len(scales)]
                specs1_bg_removed[i, :], _ = dwt_iterative_bg_rm_func(X_conv1[i, :], wavelet, scale, iterations)
                specs2_bg_removed[i, :], _ = dwt_iterative_bg_rm_func(X_conv2[i, :], wavelet, scale, iterations)
            after_diff = specs1_bg_removed - specs2_bg_removed
            concatenated_data_dict[F] = {
                "specs_1_bg_noise_mean": specs1_bg_removed,
                "specs_2_bg_noise_mean": specs2_bg_removed,
                "serds_bg_noise_mean": after_diff
            }
        else:
            concatenated_data_dict[F] = None
    return concatenated_data_dict


def asymmetric_least_squares(y, lam, p, niter=10):
    """
    Performs asymmetric least squares smoothing for background estimation.

    The algorithm iteratively computes a background estimate by adjusting weights based on
    the differences between the signal and the current background estimate.

    Parameters
    ----------
    y : np.ndarray
        1D array representing the input spectrum.
    lam : float
        Smoothing parameter (larger values yield smoother estimates).
    p : float
        Asymmetry parameter (between 0 and 1) controlling the weighting.
    niter : int, optional
        Number of iterations (default is 10).

    Returns
    -------
    z : np.ndarray
        The estimated background.
    """
    L = len(y)
    D = spdiags([-1 * np.ones(L), 2 * np.ones(L), -1 * np.ones(L)], [-1, 0, 1], L, L).tocsc()
    D = lam * (D.T @ D)
    w = np.ones(L)
    for _ in range(niter):
        W = spdiags(w, 0, L, L)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def preprocess_ALS(results_dict, initial_F_levels, pad_length, als_func):
    """
    Preprocesses Raman/SERDS spectra using Asymmetric Least Squares (ALS) background removal.

    For each fluorescence level in `initial_F_levels`, the function pads the spectra, applies ALS to
    estimate the background, and subtracts the estimated background from the original spectra.
    The SERDS difference is computed as the difference between the two laser channels.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing conventional spectra for each fluorescence level. Each entry should contain:
            - "specs_1_bg_noise_mean": np.ndarray (n_spectra x n_points)
            - "specs_2_bg_noise_mean": np.ndarray (n_spectra x n_points)
    initial_F_levels : list
        List of fluorescence levels (keys in results_dict) to process.
    pad_length : int
        Number of points to pad on each side of the spectrum before applying ALS.
    als_func : function
        Function implementing the ALS algorithm (e.g., asymmetric_least_squares).

    Returns
    -------
    concatenated_data_dict : dict
        Dictionary keyed by fluorescence level. Each value is a dictionary containing:
            - "specs_1_bg_noise_mean": background-removed spectra for laser 1
            - "specs_2_bg_noise_mean": background-removed spectra for laser 2
            - "serds_bg_noise_mean": difference between the two channels
    """
    concatenated_data_dict = {}
    for F in initial_F_levels:
        target_results = results_dict.get(F)
        if target_results is not None:
            X_conv1 = target_results.get("specs_1_bg_noise_mean")
            X_conv2 = target_results.get("specs_2_bg_noise_mean")
            n_spectra, _ = X_conv1.shape
            X_conv_spec1 = []
            X_conv_spec2 = []
            for i in range(n_spectra):
                padded1 = np.pad(X_conv1[i], pad_width=pad_length, mode='edge')
                padded2 = np.pad(X_conv2[i], pad_width=pad_length, mode='edge')
                als_removed1 = als_func(padded1, 1e5, 0.1, niter=10)[pad_length:-pad_length]
                als_removed2 = als_func(padded2, 1e5, 0.1, niter=10)[pad_length:-pad_length]
                X_conv_spec1.append(als_removed1)
                X_conv_spec2.append(als_removed2)
            X_conv_spec1 = np.array(X_conv_spec1)
            X_conv_spec2 = np.array(X_conv_spec2)
            after_removed1 = X_conv1 - X_conv_spec1
            after_removed2 = X_conv2 - X_conv_spec2
            after_diff = after_removed1 - after_removed2
            concatenated_data_dict[F] = {
                "specs_1_bg_noise_mean": after_removed1,
                "specs_2_bg_noise_mean": after_removed2,
                "serds_bg_noise_mean": after_diff
            }
        else:
            concatenated_data_dict[F] = None
    return concatenated_data_dict



def advanced_extrapolate(arr, pad_width, num_points=10):
    """
    Pads the input 1D array via linear extrapolation.
    
    This function fits a linear regression to the first and last few points of the array,
    then extrapolates to generate pad_width extra points on each side.
    
    Parameters
    ----------
    arr : np.ndarray
        1D array representing the spectrum.
    pad_width : int
        Number of points to pad on each side.
    num_points : int, optional
        Number of points to use for regression at each end (default is 10, capped at len(arr)//2).
    
    Returns
    -------
    padded_arr : np.ndarray
        Array with left and right padding generated via linear extrapolation.
    """
    num_points = min(num_points, len(arr) // 2)

    # Left end regression
    x_left = np.arange(num_points)
    y_left = arr[:num_points]
    slope_left, intercept_left, _, _, _ = linregress(x_left, y_left)

    # Right end regression
    x_right = np.arange(len(arr) - num_points, len(arr))
    y_right = arr[-num_points:]
    slope_right, intercept_right, _, _, _ = linregress(x_right, y_right)

    # Generate padding
    left_pad = intercept_left + slope_left * (np.arange(-pad_width, 0))
    right_pad = intercept_right + slope_right * (np.arange(len(arr), len(arr) + pad_width))
    
    return np.concatenate((left_pad, arr, right_pad))


def preprocess_als_extrapolate(results_dict, initial_F_levels, pad_length, asymmetric_least_squares):
    """
    Applies ALS-based background removal using advanced extrapolation for padding.
    
    For each F level in `initial_F_levels`, this function:
      - Retrieves the noise-added conventional spectra ("specs_1_bg_noise_mean" and "specs_2_bg_noise_mean")
        from results_dict.
      - For each spectrum (sample), pads the spectrum using advanced_extrapolate (with a pad of length pad_length),
        applies the ALS background correction with parameters (1e5, 0.1, niter=10), and then removes the padded regions.
      - Computes the background-removed spectra by subtracting the ALS fit from the original spectrum.
      - Computes the SERDS difference as the difference between the two background-removed spectra.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary keyed by F level containing noise-added spectra.
    initial_F_levels : list
        List of F levels to process.
    pad_length : int
        Length of the padding (on both sides) used for extrapolation.
    asymmetric_least_squares : function
        Function that performs ALS background removal.
    
    Returns
    -------
    concatenated_data_dict : dict
        Dictionary keyed by F level. For each F level, the value is a dictionary with:
            "specs_1_bg_noise_mean": background-removed spectrum for laser 1,
            "specs_2_bg_noise_mean": background-removed spectrum for laser 2,
            "serds_bg_noise_mean": SERDS difference (the difference between the two).
        If an F level is not found in results_dict, its value will be None.
    """
    concatenated_data_dict = {}

    for F in initial_F_levels:
        target_results = results_dict.get(F)
        if target_results is not None:
            # Retrieve noise-added spectra
            X_conv1 = target_results.get("specs_1_bg_noise_mean")
            X_conv2 = target_results.get("specs_2_bg_noise_mean")
            n_spectra, _ = X_conv1.shape

            X_conv_spec1 = []
            X_conv_spec2 = []

            for i in range(n_spectra):
                # Pad the spectrum using advanced_extrapolate
                padded1 = advanced_extrapolate(X_conv1[i], pad_length)
                padded2 = advanced_extrapolate(X_conv2[i], pad_length)
                
                # Apply ALS background removal on the padded spectrum
                als_fit1 = asymmetric_least_squares(padded1, 1e5, 0.1, niter=10)
                als_fit2 = asymmetric_least_squares(padded2, 1e5, 0.1, niter=10)
                
                # Remove the padded regions from the ALS fit
                als_removed1 = als_fit1[pad_length:-pad_length]
                als_removed2 = als_fit2[pad_length:-pad_length]
                
                X_conv_spec1.append(als_removed1)
                X_conv_spec2.append(als_removed2)

            X_conv_spec1 = np.array(X_conv_spec1)
            X_conv_spec2 = np.array(X_conv_spec2)

            # Compute background-removed spectra as difference between original and ALS fit
            after_removed1 = X_conv1 - X_conv_spec1
            after_removed2 = X_conv2 - X_conv_spec2
            after_diff = after_removed1 - after_removed2

            concatenated_data_dict[F] = {
                "specs_1_bg_noise_mean": after_removed1,
                "specs_2_bg_noise_mean": after_removed2,
                "serds_bg_noise_mean": after_diff
            }
        else:
            concatenated_data_dict[F] = None

    return concatenated_data_dict
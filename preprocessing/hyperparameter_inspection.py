"""
Module: hyperparameter_inspection.py
Description:
    This module provides functions to visually inspect the performance of background 
    removal methods (ALS and DWT) so you can select optimal hyperparameters.
    
    Functions:
      - plot_als_bg_removal_edge: Plots ALS background removal using edge padding.
      - plot_als_bg_removal_extrapolate: Plots ALS background removal using advanced (extrapolated) padding.
      - plot_dwt_bg_removal_inspection: Plots examples of DWT background removal.
      
Dependencies:
    numpy, matplotlib, and the background removal functions (asymmetric_least_squares, 
    dwt_iterative_bg_rm) along with an advanced_extrapolate function.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_als_bg_removal_edge(results_dict, initial_F_levels, asymmetric_least_squares, pad_length=200,
                             title_fontsize=16, label_fontsize=16, legend_fontsize=16, axis_fontsize=16,
                             save_path=None):
    """
    Plots an example of ALS background removal using edge padding.
    
    For each fluorescence level in initial_F_levels, a random spectrum is selected from 
    results_dict (key "specs_1_bg_noise_mean"). The spectrum is padded using edge padding,
    ALS background removal is applied, and both the padded (with estimated background)
    and the background-removed spectrum are plotted.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary keyed by F level; each value should contain "specs_1_bg_noise_mean" as a 2D array.
    initial_F_levels : list
        List of fluorescence levels to inspect.
    asymmetric_least_squares : function
        Function to perform ALS background removal.
    pad_length : int, optional
        Padding length (default is 200).
    title_fontsize, label_fontsize, legend_fontsize, axis_fontsize : int, optional
        Font sizes for titles, axis labels, legend, and tick labels.
    save_path : str, optional
        If provided, the figure is saved to this path.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    axs : numpy.ndarray
        Array of axes used in the figure.
    """
    n_levels = len(initial_F_levels)
    fig, axs = plt.subplots(n_levels, 2, figsize=(15, 5 * n_levels))
    
    for index, fsr in enumerate(initial_F_levels):
        target = results_dict.get(fsr)
        if target is None or "specs_1_bg_noise_mean" not in target:
            continue
        n_samples = len(target["specs_1_bg_noise_mean"])
        spectrum_index = np.random.choice(range(n_samples))
        spectrum = target["specs_1_bg_noise_mean"][spectrum_index]
        
        # Use edge padding
        padded_spectrum = np.pad(spectrum, (pad_length, pad_length), mode='edge')
        # Apply ALS background removal
        als_background = asymmetric_least_squares(padded_spectrum, 1e5, 0.1, niter=10)
        bg_removed = spectrum - als_background[pad_length:-pad_length]
        
        # Left subplot: original padded spectrum, estimated background, and edge markers
        axs[index, 0].plot(padded_spectrum, label='Original Spectrum with Edge Padding', linewidth=1.5)
        axs[index, 0].axvline(pad_length, color='red', linestyle='--', linewidth=1.5, label='Start of Actual Spectrum')
        axs[index, 0].axvline(len(padded_spectrum) - pad_length, color='red', linestyle='--', linewidth=1.5, label='End of Actual Spectrum')
        axs[index, 0].plot(als_background, label='Estimated Background', linestyle='--')
        axs[index, 0].spines['top'].set_visible(False)
        axs[index, 0].spines['right'].set_visible(False)
        axs[index, 0].tick_params(axis='both', labelsize=axis_fontsize)
        axs[index, 0].set_title(f'ALS BG Removal (Edge Padding) - FSR {fsr}', fontsize=title_fontsize)
        axs[index, 0].legend(fontsize=legend_fontsize)
        
        # Right subplot: background-removed spectrum
        axs[index, 1].plot(bg_removed, label='Spectrum after BG Removal', linewidth=1.5)
        axs[index, 1].tick_params(axis='both', labelsize=axis_fontsize)
        axs[index, 1].set_title(f'BG Removed Spectrum - FSR {fsr}', fontsize=title_fontsize)
        axs[index, 1].legend(fontsize=legend_fontsize)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.show()
    return fig, axs

def plot_als_bg_removal_extrapolate(results_dict, initial_F_levels, asymmetric_least_squares, advanced_extrapolate,
                                    pad_length=200, title_fontsize=16, label_fontsize=16, legend_fontsize=16, axis_fontsize=16,
                                    save_path=None):
    """
    Plots an example of ALS background removal using advanced extrapolation for padding.
    
    For each fluorescence level in initial_F_levels, a random spectrum is selected from results_dict.
    The spectrum is padded using advanced_extrapolate, ALS background removal is applied, and both 
    the padded spectrum (with estimated background) and the background-removed spectrum are plotted.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary keyed by F level containing "specs_1_bg_noise_mean".
    initial_F_levels : list
        List of fluorescence levels to inspect.
    asymmetric_least_squares : function
        Function to perform ALS background removal.
    advanced_extrapolate : function
        Function to perform linear extrapolation padding.
    pad_length : int, optional
        Padding length (default 200).
    title_fontsize, label_fontsize, legend_fontsize, axis_fontsize : int, optional
        Font sizes for titles, labels, legend, and ticks.
    save_path : str, optional
        If provided, the figure is saved to this path.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    axs : numpy.ndarray
        Array of axes used in the figure.
    """
    n_levels = len(initial_F_levels)
    fig, axs = plt.subplots(n_levels, 2, figsize=(15, 5 * n_levels))
    
    for index, fsr in enumerate(initial_F_levels):
        target = results_dict.get(fsr)
        if target is None or "specs_1_bg_noise_mean" not in target:
            continue
        n_samples = len(target["specs_1_bg_noise_mean"])
        spectrum_index = np.random.choice(range(n_samples))
        spectrum = target["specs_1_bg_noise_mean"][spectrum_index]
        
        # Use advanced extrapolation for padding
        padded_spectrum = advanced_extrapolate(spectrum, pad_length)
        # Apply ALS background removal
        als_background = asymmetric_least_squares(padded_spectrum, 1e5, 0.1, niter=10)
        bg_removed = spectrum - als_background[pad_length:-pad_length]
        
        # Left subplot: original padded spectrum and estimated background with markers
        axs[index, 0].plot(padded_spectrum, label='Original Spectrum with Extrapolated Padding', linewidth=1.5)
        axs[index, 0].axvline(pad_length, color='red', linestyle='--', linewidth=1.5, label='Start of Actual Spectrum')
        axs[index, 0].axvline(len(padded_spectrum) - pad_length, color='red', linestyle='--', linewidth=1.5, label='End of Actual Spectrum')
        axs[index, 0].plot(als_background, label='Estimated Background', linestyle='--')
        axs[index, 0].spines['top'].set_visible(False)
        axs[index, 0].spines['right'].set_visible(False)
        axs[index, 0].tick_params(axis='both', labelsize=axis_fontsize)
        axs[index, 0].set_title(f'ALS BG Removal (Extrapolated Padding) - FSR {fsr}', fontsize=title_fontsize)
        axs[index, 0].legend(fontsize=legend_fontsize)
        
        # Right subplot: background-removed spectrum
        axs[index, 1].plot(bg_removed, label='Spectrum after BG Removal', linewidth=1.5)
        axs[index, 1].tick_params(axis='both', labelsize=axis_fontsize)
        axs[index, 1].set_title(f'BG Removed Spectrum - FSR {fsr}', fontsize=title_fontsize)
        axs[index, 1].legend(fontsize=legend_fontsize)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.show()
    return fig, axs

def plot_dwt_bg_removal_inspection(results_dict, initial_F_levels, scales, dwt_iterative_bg_rm, start, end, save_path=None):
    """
    Plots DWT background removal inspection for a subset of fluorescence levels.
    
    For each fluorescence level in initial_F_levels (from index start to end), a random spectrum 
    is selected from results_dict. DWT background removal is applied using the specified scale and 50 iterations.
    Two subplots are created: one showing the original spectrum with the estimated background, and 
    one showing the background-removed spectrum.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary keyed by F level containing "specs_1_bg_noise_mean".
    initial_F_levels : list
        List of fluorescence levels to inspect.
    scales : list
        List of scales for DWT corresponding to each fluorescence level.
    dwt_iterative_bg_rm : function
        Function that performs iterative DWT background removal.
    start : int
        Starting index for selecting fluorescence levels.
    end : int
        Ending index (non-inclusive) for selecting fluorescence levels.
    save_path : str, optional
        File path to save the generated figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    axs : numpy.ndarray
        Array of axes used in the figure.
    """
    n_plots = end - start
    fig, axs = plt.subplots(n_plots, 2, figsize=(15, 5 * n_plots))
    
    for idx, fsr in enumerate(initial_F_levels[start:end]):
        target = results_dict.get(fsr)
        if target is None or "specs_1_bg_noise_mean" not in target:
            continue
        n_samples = len(target["specs_1_bg_noise_mean"])
        spectrum_index = np.random.choice(range(n_samples))
        spectrum = target["specs_1_bg_noise_mean"][spectrum_index]
        
        # Apply DWT background removal
        bg_removed, bg_estimated = dwt_iterative_bg_rm(spectrum, 'sym5', scales[idx + start], 50)
        
        # Left subplot: original spectrum and estimated background
        axs[idx, 0].plot(spectrum, label='Original Spectrum')
        axs[idx, 0].plot(bg_estimated, label='Estimated Background', linestyle='--')
        axs[idx, 0].spines['top'].set_visible(False)
        axs[idx, 0].spines['right'].set_visible(False)
        axs[idx, 0].set_title(f'Original & Estimated BG - FSR {fsr}', fontsize=16)
        axs[idx, 0].legend(fontsize=16)
        axs[idx, 0].tick_params(axis='both', labelsize=16)
        
        # Right subplot: background removed spectrum
        axs[idx, 1].plot(bg_removed, label='Spectrum after BG Removal')
        axs[idx, 1].spines['top'].set_visible(False)
        axs[idx, 1].spines['right'].set_visible(False)
        axs[idx, 1].set_title(f'BG Removed Spectrum - FSR {fsr}', fontsize=16)
        axs[idx, 1].legend(fontsize=16)
        axs[idx, 1].tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.show()
    return fig, axs

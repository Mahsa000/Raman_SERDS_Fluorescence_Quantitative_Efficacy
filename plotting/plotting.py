"""
Module: plotting.py
Description:
    This module provides functions for visualizing Raman/SERDS data.
    
    Available functions:
      - plot_feature_importance:
            Plots background spectra and overlays feature importance markers for regular datasets.
      - plot_calibration:
            Creates calibration scatter plots (measured vs. predicted) for regular datasets.
      - plot_feature_importance_dataset5:
            Similar to plot_feature_importance but uses a title format for heterogeneous Dataset 5,
            showing both FSR values (e.g. "FSR1: X, FSR2: Y").
      - plot_calibration_dataset5:
            Creates calibration scatter plots for Dataset 5, with titles displaying both FSR1 and FSR2.
    
Imports:
    os, numpy, matplotlib.pyplot, and matplotlib.ticker.ScalarFormatter.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.integrate import trapz
from scipy.interpolate import interp1d

# Plot 1D Raman spectra
def plot_specs(specs, w=None, id_list=None, title=None, show_labels=False):
    if w is None:
        w = range(len(specs[0]))
    if id_list is None:
        id_list = [f'Sample {i+1}' for i in range(len(specs))]  # Default labels
        
    # Reshape specs to a 2D array if it is 1D
    if len(specs.shape) == 1:
        specs = specs.reshape(1, -1)
    
    fig = plt.figure()
    grid = plt.GridSpec(1, 1, wspace=0.08, hspace=0.15)
    ax = fig.add_subplot(grid[0, 0])
    for sample_num in range(len(specs)):
        plt.plot(w, specs[sample_num]/10000, label=id_list[sample_num], linewidth=0.9)

    if w[-1] > 1000:
        plt.xlabel('Raman shift ($cm^{-1}$)', fontsize=14)
        plt.ylabel('Intensity (a.u.) x 10$^{4}$', fontsize=14)
        plt.title(title, fontsize=16)
    else:
        plt.xlabel('Wavelength ($nm$)', fontsize=14)
        plt.ylabel('Intensity (a.u.) x 10$^{4}$', fontsize=14)
        plt.title(title, fontsize=16)
    
    if show_labels:
        plt.legend()

    plt.show()
    

def plot_synthetic_specs(gen_spec, pure_fluorescence, w, title=None, save_path=None):
    plt.rcParams["figure.figsize"] = (10, 8)
    name_size = 22
    tick_label_size = 18

    fig, ax = plt.subplots()

    # Plot 1000 spectra using a colormap
    cm = plt.get_cmap('viridis')
    for i, spec in enumerate(gen_spec):
        ax.plot(w, spec / 10000,  linewidth=0.5)
    ax.set_xlabel('Wavelength (nm)', fontsize=name_size, fontname='Arial', labelpad=8)
    ax.set_ylabel('Intensity (a.u.) x 10$^{4}$', fontsize=name_size, fontname='Arial', labelpad=8)
    if title:
        ax.set_title(title, fontsize=name_size, fontname='Arial')
    
    ax.tick_params(axis='both', which='both', labelsize=tick_label_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=400)

    plt.show()

def plot_FSR_definition(gen_spec, pure_fluorescence, w, title=None, save_path=None):
    """
    Plots the synthetic spectra together with pure fluorescence and marks the key points
    that define the FSR (fluorescence-to-Raman ratio).

    The function plots:
      - All synthetic spectra (gen_spec) scaled down.
      - The pure fluorescence (scaled) in a distinctive color (#FFA500).
      - A vertical line at the wavelength corresponding to the maximum of the pure fluorescence.
      - A vertical line (dotted) at the wavelength corresponding to the overall maximum of the synthetic mixture.
      
    Parameters
    ----------
    gen_spec : np.ndarray
        2D array of synthetic spectra.
    pure_fluorescence : np.ndarray
        1D (or 2D) array of pure fluorescence values. If 2D, it is assumed to be scaled appropriately.
    w : array-like
        1D array of wavelength values.
    title : str, optional
        Title for the plot.
    save_path : str, optional
        File path (including filename) where the figure will be saved.
    
    Returns
    -------
    None
    """
    # Adjust figure settings
    plt.rcParams["figure.figsize"] = (10, 8)
    name_size = 22
    tick_label_size = 18

    fig, ax = plt.subplots()

    # Plot synthetic spectra using a colormap (here, using default color)
    for i, spec in enumerate(gen_spec):
        ax.plot(w, spec / 10000, linewidth=0.5)

    # Plot pure fluorescence scaled slightly to generate FSR = 1 and in a distinctive color
    ax.plot(w, (pure_fluorescence / 1.4342505360698254) / 10000, color='#FFA500', label='Pure Fluorescence', linewidth=1.5)

    # Mark maximum point of pure fluorescence
    max_fluorescence_idx = np.argmax(pure_fluorescence)
    max_fluorescence_wavelength = w[max_fluorescence_idx]
    ax.axvline(max_fluorescence_wavelength, color='black', linestyle='--', linewidth=1.5, label='Max Fluorescence')

    # Mark overall maximum of the synthetic mixture
    max_values = np.max(gen_spec, axis=1)  # Maximum value from each synthetic spectrum
    overall_max_idx = np.argmax(max_values)
    max_spec_idx = np.argmax(gen_spec[overall_max_idx])
    max_spec_wavelength = w[max_spec_idx]
    ax.axvline(max_spec_wavelength, color='black', linestyle='dotted', linewidth=1.5, label='Max Synthetic Mixture')

    ax.set_xlabel('Wavelength (nm)', fontsize=name_size, fontname='Arial', labelpad=8)
    ax.set_ylabel('Intensity (a.u.) x 10$^{4}$', fontsize=name_size, fontname='Arial', labelpad=8)
    if title:
        ax.set_title(title, fontsize=name_size, fontname='Arial')

    ax.tick_params(axis='both', which='both', labelsize=tick_label_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.show()



def plot_shifted_fluorescence_shapes(pure_fluorescence, w_nm, save_path=None):
    """
    Plots the pure fluorescence and its shifted versions for a range of shift points.
    
    For each shift point (from 140 to 340 in steps of 20), the function:
      - Computes the shift offset relative to the peak of the pure fluorescence,
      - Generates a shifted x-axis and interpolates to obtain the shifted fluorescence,
      - Plots both the original and shifted fluorescence on a subplot.
    
    The function creates subplots (3 per row) with increased axis and title sizes (16),
    and saves the figure if a save_path is provided.
    
    Parameters
    ----------
    pure_fluorescence : np.ndarray
        1D array of pure fluorescence values.
    w_nm : array-like
        1D array of wavelength values.
    save_path : str, optional
        File path to save the resulting figure. If not provided, the figure is only shown.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    # Define the shift points to iterate over
    shift_points = np.arange(140, 341, 20)  # From 140 to 340 inclusive with step 20
    n_plots = len(shift_points)
    n_rows = n_plots // 3 + (1 if n_plots % 3 > 0 else 0)
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 5))
    fig.subplots_adjust(hspace=0.4)
    
    # Set font sizes to 16
    axis_font_size = 18
    title_font_size = 18
    tick_label_size = 16
    line_width = 1.5

    # Loop over each shift point to create the subplots
    for i, shift_point in enumerate(shift_points):
        # Compute shift offset relative to the peak of pure_fluorescence
        peak_position = np.argmax(pure_fluorescence)
        shift_offset = shift_point - peak_position

        # Create a shifted x-axis and interpolate the fluorescence
        shifted_x = np.arange(len(pure_fluorescence)) - shift_offset
        interpolator = interp1d(shifted_x, pure_fluorescence, fill_value="extrapolate")
        shifted_fluorescence = interpolator(np.arange(len(pure_fluorescence)))
        
        # Determine subplot indices
        row = i // 3
        col = i % 3
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Plot the original pure fluorescence (scaled) and the shifted version
        ax.plot(w_nm, pure_fluorescence / 10000, color='#FFA500', linewidth=line_width)
        ax.plot(w_nm, shifted_fluorescence / 10000, color='#4682B4', linewidth=line_width)
        ax.set_title(f"Shift Point {shift_point}", fontsize=title_font_size)
        
        # Set labels only for edge subplots
        if row == n_rows - 1:
            ax.set_xlabel('Wavelength (nm)', fontsize=axis_font_size, fontname='Arial', labelpad=8)
        if col == 0:
            ax.set_ylabel('Raman intensities (a.u.) x 10$^{4}$', fontsize=axis_font_size, fontname='Arial', labelpad=8)
        
        ax.tick_params(axis='both', which='both', labelsize=tick_label_size)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Remove any unused subplots if necessary
    total_axes = n_rows * 3
    if n_plots < total_axes:
        for j in range(n_plots, total_axes):
            row = j // 3
            col = j % 3
            fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.show()
    
    return fig
    
    

def plot_feature_importance(F_decaying_results_dict, initial_F_levels, wavelengths_new, 
                            feature_importance_dict, max_argmax_rmsep_df, fig_save_path):
    """
    Plots feature importance alongside background spectra for each fluorescence (F) level.
    
    The function creates a figure with two columns per F level: the left shows the conventional 
    background spectra and the right shows the SERDS difference spectra. It overlays markers and 
    vertical lines for features that exceed 50% of the maximum PLSR loadings.
    
    Parameters
    ----------
    F_decaying_results_dict : dict
        Dictionary with simulation results for each F level (keys: "specs_1_bg_noise_mean" and "serds_bg_noise_mean").
    initial_F_levels : list
        List of fluorescence levels to plot.
    wavelengths_new : np.ndarray
        1D array of wavelength values.
    feature_importance_dict : dict
        Dictionary containing feature importance for each F level (with keys 'conv' and 'serds').
    max_argmax_rmsep_df : pandas.DataFrame
        DataFrame summarizing maximum coefficient values and their indices.
    fig_save_path : str
        File path where the resulting figure will be saved.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axs : numpy.ndarray
        Array of axes used in the figure.
    """
    font = 'Arial'
    tick_label_size = 20
    axis_label_size = 24

    n_f = len(initial_F_levels)
    fig_height = 20 * n_f / 8 if n_f < 8 else 20
    fig, axs = plt.subplots(n_f, 2, figsize=(20, fig_height))
    if n_f == 1:
        axs = np.array([axs])
    plt.subplots_adjust(hspace=0.5, wspace=0.2)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    
    threshold_percentage = 0.5
    thresholds_conv = [threshold_percentage * max_val 
                       for max_val in max_argmax_rmsep_df[max_argmax_rmsep_df['Data Type'] == 'conv']['Max Value']]
    thresholds_serds = [threshold_percentage * max_val 
                        for max_val in max_argmax_rmsep_df[max_argmax_rmsep_df['Data Type'] == 'serds']['Max Value']]

    ref_F = 0 if 0 in initial_F_levels else initial_F_levels[0]
    specs_1_bg_noise_mean_ref = F_decaying_results_dict.get(ref_F, {}).get("specs_1_bg_noise_mean")
    serds_bg_noise_mean_ref = F_decaying_results_dict.get(ref_F, {}).get("serds_bg_noise_mean")
    if specs_1_bg_noise_mean_ref is None or serds_bg_noise_mean_ref is None:
        raise ValueError("Reference background data not found for F level {}.".format(ref_F))

    # Plot the reference background for scaling
    for i, (ax1, ax2) in enumerate(axs):
        for ax, data in zip([ax1, ax2], [specs_1_bg_noise_mean_ref, serds_bg_noise_mean_ref]):
            if data is None:
                continue
            max_intensity = np.max(data)
            key = "specs_1_bg_noise_mean" if ax == ax1 else "serds_bg_noise_mean"
            current_data = F_decaying_results_dict.get(initial_F_levels[i], {}).get(key)
            if current_data is None:
                continue
            max_intensity_row = np.max(current_data)
            scaling_factor = max_intensity_row / max_intensity if max_intensity != 0 else 1
            ax.plot(wavelengths_new, data.T * scaling_factor, color='lightgray', alpha=0.6, rasterized=True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    for i, initial_F_level in enumerate(initial_F_levels):
        axs[i, 0].yaxis.set_major_formatter(formatter)
        axs[i, 1].yaxis.set_major_formatter(formatter)
        target = F_decaying_results_dict.get(initial_F_level)
        if target is not None:
            conv_data = target.get("specs_1_bg_noise_mean")
            serds_data = target.get("serds_bg_noise_mean")
            if conv_data is not None and serds_data is not None:
                axs[i, 0].plot(wavelengths_new, conv_data.T, rasterized=True)
                axs[i, 1].plot(wavelengths_new, serds_data.T, rasterized=True)
                axs[i, 1].set_ylabel("")
                for j in range(2):
                    axs[i, j].tick_params(axis='both', labelsize=tick_label_size)
                    axs[i, j].spines['right'].set_visible(True)
                
                feat_imp_conv = feature_importance_dict[initial_F_level]['conv']
                imp_idx_conv = np.where(feat_imp_conv > thresholds_conv[i])[0]
                ax_conv_twin = axs[i, 0].twinx()
                ax_conv_twin.yaxis.set_major_formatter(formatter)
                ax_conv_twin.tick_params(axis='y', labelsize=tick_label_size)
                ax_conv_twin.plot(wavelengths_new[imp_idx_conv],
                                 feat_imp_conv[imp_idx_conv],
                                 'ro', markersize=2, rasterized=True)
                ax_conv_twin.vlines(wavelengths_new[imp_idx_conv],
                                   0, feat_imp_conv[imp_idx_conv],
                                   linestyle='--', color='red', linewidth=0.5, rasterized=True)
                
                feat_imp_serds = feature_importance_dict[initial_F_level]['serds']
                imp_idx_serds = np.where(feat_imp_serds > thresholds_serds[i])[0]
                ax_serds_twin = axs[i, 1].twinx()
                ax_serds_twin.yaxis.set_major_formatter(formatter)
                ax_serds_twin.tick_params(axis='y', labelsize=tick_label_size)
                ax_serds_twin.plot(wavelengths_new[imp_idx_serds],
                                  feat_imp_serds[imp_idx_serds],
                                  'bo', markersize=2, rasterized=True)
                ax_serds_twin.vlines(wavelengths_new[imp_idx_serds],
                                     0, feat_imp_serds[imp_idx_serds],
                                     linestyle='--', color='blue', linewidth=0.5, rasterized=True)

    for ax in axs[-1, :]:
        ax.set_xlabel("Wavelength (nm)", fontsize=axis_label_size, fontname=font, labelpad=15)

    fig.text(0.07, 0.5, "Raman Intensity (a.u.)", va='center', ha='center',
             fontsize=axis_label_size, fontname=font, rotation=90)
    fig.text(0.96, 0.5, "Top 50% Important Features by PLSR Loadings", va='center', ha='center',
             fontsize=axis_label_size, fontname=font, rotation=-90)

    for ax in axs.flatten():
        ax.yaxis.set_major_formatter(formatter)

    os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)
    plt.savefig(fig_save_path, dpi=150)
    plt.show()
    plt.draw()

    return fig, axs


def plot_feature_importance_dataset5(F_decaying_results_dict, FSR_combinations, wavelengths_new, 
                                     feature_importance_dict, max_argmax_rmsep_df, fig_save_path):
    """
    Plots feature importance for Dataset 5, where data are heterogeneous combinations of FSR levels.
    
    Similar to plot_feature_importance, but expects F_decaying_results_dict to be keyed by FSR combinations
    (tuples). The titles of each row are set to display both FSR values (e.g., "FSR1: X, FSR2: Y").
    
    Parameters
    ----------
    F_decaying_results_dict : dict
        Dictionary with simulation results for each FSR combination. Each value should have keys:
            "specs_1_bg_noise_mean" and "serds_bg_noise_mean".
    FSR_combinations : list of tuple
        List of FSR combination tuples (e.g., [(0,1), (1,10), ...]) to be plotted.
    wavelengths_new : np.ndarray
        1D array of wavelength values.
    feature_importance_dict : dict
        Dictionary with feature importance for each FSR combination (keys 'conv' and 'serds').
    max_argmax_rmsep_df : pandas.DataFrame
        DataFrame summarizing maximum coefficient values and their indices.
    fig_save_path : str
        File path (including filename) to save the figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axs : numpy.ndarray
        Array of axes used in the figure.
    """
    font = 'Arial'
    tick_label_size = 20
    axis_label_size = 24

    n_rows = len(FSR_combinations)
    fig_height = 20 * n_rows / 8 if n_rows < 8 else 20
    fig, axs = plt.subplots(n_rows, 2, figsize=(20, fig_height))
    if n_rows == 1:
        axs = np.array([axs])
    plt.subplots_adjust(hspace=0.5, wspace=0.2)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    
    threshold_percentage = 0.5
    thresholds_conv = [threshold_percentage * max_val 
                       for max_val in max_argmax_rmsep_df[max_argmax_rmsep_df['Data Type'] == 'conv']['Max Value']]
    thresholds_serds = [threshold_percentage * max_val 
                        for max_val in max_argmax_rmsep_df[max_argmax_rmsep_df['Data Type'] == 'serds']['Max Value']]

    # Use the first available combination as reference for scaling
    ref_comb = FSR_combinations[0]
    specs_1_bg_noise_mean_ref = F_decaying_results_dict.get(ref_comb, {}).get("specs_1_bg_noise_mean")
    serds_bg_noise_mean_ref = F_decaying_results_dict.get(ref_comb, {}).get("serds_bg_noise_mean")
    if specs_1_bg_noise_mean_ref is None or serds_bg_noise_mean_ref is None:
        raise ValueError("Reference background data not found for FSR combination {}.".format(ref_comb))

    for i, (ax1, ax2) in enumerate(axs):
        for ax, data in zip([ax1, ax2], [specs_1_bg_noise_mean_ref, serds_bg_noise_mean_ref]):
            if data is None:
                continue
            max_intensity = np.max(data)
            key = "specs_1_bg_noise_mean" if ax == ax1 else "serds_bg_noise_mean"
            current_data = F_decaying_results_dict.get(FSR_combinations[i], {}).get(key)
            if current_data is None:
                continue
            max_intensity_row = np.max(current_data)
            scaling_factor = max_intensity_row / max_intensity if max_intensity != 0 else 1
            ax.plot(wavelengths_new, data.T * scaling_factor, color='lightgray', alpha=0.6, rasterized=True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    for i, fsr_pair in enumerate(FSR_combinations):
        axs[i, 0].yaxis.set_major_formatter(formatter)
        axs[i, 1].yaxis.set_major_formatter(formatter)
        target = F_decaying_results_dict.get(fsr_pair)
        if target is not None:
            conv_data = target.get("specs_1_bg_noise_mean")
            serds_data = target.get("serds_bg_noise_mean")
            if conv_data is not None and serds_data is not None:
                axs[i, 0].plot(wavelengths_new, conv_data.T, rasterized=True)
                axs[i, 1].plot(wavelengths_new, serds_data.T, rasterized=True)
                axs[i, 1].set_ylabel("")
                for j in range(2):
                    axs[i, j].tick_params(axis='both', labelsize=tick_label_size)
                    axs[i, j].spines['right'].set_visible(True)
                
                feat_imp_conv = feature_importance_dict[fsr_pair]['conv']
                imp_idx_conv = np.where(feat_imp_conv > thresholds_conv[i])[0]
                ax_conv_twin = axs[i, 0].twinx()
                ax_conv_twin.yaxis.set_major_formatter(formatter)
                ax_conv_twin.tick_params(axis='y', labelsize=tick_label_size)
                ax_conv_twin.plot(wavelengths_new[imp_idx_conv],
                                 feat_imp_conv[imp_idx_conv],
                                 'ro', markersize=2, rasterized=True)
                ax_conv_twin.vlines(wavelengths_new[imp_idx_conv],
                                   0, feat_imp_conv[imp_idx_conv],
                                   linestyle='--', color='red', linewidth=0.5, rasterized=True)
                
                feat_imp_serds = feature_importance_dict[fsr_pair]['serds']
                imp_idx_serds = np.where(feat_imp_serds > thresholds_serds[i])[0]
                ax_serds_twin = axs[i, 1].twinx()
                ax_serds_twin.yaxis.set_major_formatter(formatter)
                ax_serds_twin.tick_params(axis='y', labelsize=tick_label_size)
                ax_serds_twin.plot(wavelengths_new[imp_idx_serds],
                                  feat_imp_serds[imp_idx_serds],
                                  'bo', markersize=2, rasterized=True)
                ax_serds_twin.vlines(wavelengths_new[imp_idx_serds],
                                     0, feat_imp_serds[imp_idx_serds],
                                     linestyle='--', color='blue', linewidth=0.5, rasterized=True)
    
        # Use a title that displays both FSR values
        axs[i, 0].set_title(f'FSR1: {fsr_pair[0]}, FSR2: {fsr_pair[1]}', fontsize=tick_label_size, fontname=font)
        axs[i, 1].set_title(f'FSR1: {fsr_pair[0]}, FSR2: {fsr_pair[1]}', fontsize=tick_label_size, fontname=font)

    for ax in axs[-1, :]:
        ax.set_xlabel("Wavelength (nm)", fontsize=axis_label_size, fontname=font, labelpad=15)

    fig.text(0.07, 0.5, "Raman Intensity (a.u.)", va='center', ha='center',
             fontsize=axis_label_size, fontname=font, rotation=90)
    fig.text(0.96, 0.5, "Top 50% Important Features by PLSR Loadings", va='center', ha='center',
             fontsize=axis_label_size, fontname=font, rotation=-90)

    for ax in axs.flatten():
        ax.yaxis.set_major_formatter(formatter)

    os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)
    plt.savefig(fig_save_path, dpi=150)
    plt.show()
    plt.draw()

    return fig, axs


def plot_calibration(plsr_results_df_base, calib_save_path):
    """
    Plots calibration scatter plots for PLSR model predictions (regular datasets).
    
    For each plot, measured versus predicted values are plotted for both conventional and SERDS PLSR predictions.
    A reference line of perfect prediction is added.
    
    Parameters
    ----------
    plsr_results_df_base : pandas.DataFrame
        DataFrame containing modeling results for each sample.
    calib_save_path : str
        Path (including filename) to save the calibration figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The calibration figure.
    axs : numpy.ndarray
        Array of axes used in the figure.
    """
    font = 'Arial'
    axis_label_size = 18
    tick_label_size = 18

    total_plots = plsr_results_df_base.shape[0]
    if total_plots == 1:
        nrows, ncols = 1, 1
    elif total_plots == 2:
        nrows, ncols = 1, 2
    else:
        ncols = 4
        nrows = int(np.ceil(total_plots / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10), squeeze=False)
    
    red_color = '#FE420F'
    black_color = 'black'
    
    for idx, ax in enumerate(axs.flat):
        if idx < total_plots:
            row = plsr_results_df_base.iloc[idx]
            y_true = row['y_test']
            y_pred_conv = row['y_pred_conv_plsr']
            y_pred_serds = row['y_pred_serds_plsr']
            ax.scatter(y_true, y_pred_conv, color=red_color, label='Conv PLSR Predictions', marker='o')
            ax.scatter(y_true, y_pred_serds, color=black_color, label='SERDS PLSR Predictions', marker='*')
            limits = [np.min(y_true), np.max(y_true)]
            ax.plot(limits, limits, 'k--', lw=2)
            ax.set_xlabel('Measured proportions', fontsize=axis_label_size, fontname=font)
            ax.set_ylabel('Predicted proportions', fontsize=axis_label_size, fontname=font)
            ax.set_title(f'FSR: {row["F level"]}', fontsize=axis_label_size, fontname=font, y=0.95)
        else:
            ax.set_visible(False)
    
    for i, ax in enumerate(axs.flat):
        row_idx = i // axs.shape[1]
        col_idx = i % axs.shape[1]
        if col_idx != 0:
            ax.set_ylabel('')
        if row_idx != axs.shape[0] - 1:
            ax.set_xlabel('')
    
    for ax in axs.flat:
        ax.tick_params(axis='both', labelsize=tick_label_size)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(calib_save_path), exist_ok=True)
    plt.savefig(calib_save_path, dpi=800)
    plt.show()
    plt.draw()

    return fig, axs


def plot_calibration_dataset5(plsr_results_df, calib_save_path):
    """
    Plots calibration scatter plots for Dataset 5.
    
    In this version, each subplot title displays both FSR values (e.g., "FSR1: X, FSR2: Y").
    The plots show measured vs. predicted proportions for conventional and SERDS PLSR predictions,
    along with a reference line.
    
    Parameters
    ----------
    plsr_results_df : pandas.DataFrame
        DataFrame containing modeling results for each heterogeneous sample.
        Expected columns include 'F level1', 'F level2', 'y_test', 'y_pred_conv_plsr', and 'y_pred_serds_plsr'.
    calib_save_path : str
        File path (including filename) where the calibration figure will be saved.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The calibration figure.
    axs : numpy.ndarray
        Array of axes used in the figure.
    """
    font = 'Arial'
    axis_label_size = 18
    tick_label_size = 18

    total_plots = plsr_results_df.shape[0]
    # Here you may decide on a fixed layout (e.g., 2 rows x 4 columns) or a dynamic one:
    nrows = 2  
    ncols = 4  

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10), squeeze=False)
    
    red_color = '#FE420F'
    black_color = 'black'
    
    for idx, ax in enumerate(axs.flat):
        if idx < total_plots:
            row = plsr_results_df.iloc[idx]
            y_true = row['y_test']
            y_pred_conv = row['y_pred_conv_plsr']
            y_pred_serds = row['y_pred_serds_plsr']
            ax.scatter(y_true, y_pred_conv, color=red_color, label='Conv PLSR Predictions', marker='o')
            ax.scatter(y_true, y_pred_serds, color=black_color, label='SERDS PLSR Predictions', marker='*')
            limits = [np.min(y_true), np.max(y_true)]
            ax.plot(limits, limits, 'k--', lw=2)
            ax.set_xlabel('Measured proportions', fontsize=axis_label_size, fontname=font)
            ax.set_ylabel('Predicted proportions', fontsize=axis_label_size, fontname=font)
            # Use a title that displays both FSR values:
            ax.set_title(f'FSR1: {row["F level1"]}, FSR2: {row["F level2"]}', fontsize=axis_label_size, fontname=font, y=0.95)
        else:
            ax.set_visible(False)
    
    for i, ax in enumerate(axs.flat):
        row_idx = i // axs.shape[1]
        col_idx = i % axs.shape[1]
        if col_idx != 0:
            ax.set_ylabel('')
        if row_idx != axs.shape[0] - 1:
            ax.set_xlabel('')
    
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(calib_save_path), exist_ok=True)
    plt.savefig(calib_save_path, dpi=800)
    plt.show()
    plt.draw()

    return fig, axs

def plot_dataset5_paper_figure(raw_results_dict, als_results_dict, wavelengths_new, 
                               name_size=16, font='Arial', tick_label_size=11.43,
                               x_label_padding=10, y_label_padding=10,
                               fig_save_path="Results/Dataset5_paper_figure.pdf"):
    # For Dataset 5, assume the raw_results_dict has keys 0 and 1.
    np.random.seed(123)
    data_0 = raw_results_dict[0]
    data_1 = raw_results_dict[1]
    idx0 = np.random.choice(data_0['specs_1_bg_noise_mean'].shape[0], size=1)[0]
    idx1 = np.random.choice(data_1['specs_1_bg_noise_mean'].shape[0], size=1)[0]
    
    diff_0 = data_0['specs_1_bg_noise_mean'][idx0] - data_0['specs_2_bg_noise_mean'][idx0]
    diff_1 = data_1['specs_1_bg_noise_mean'][idx1] - data_1['specs_2_bg_noise_mean'][idx1]
    
    als_data_0 = als_results_dict[0]
    als_data_1 = als_results_dict[1]
    after_als_0 = als_data_0['specs_1_bg_noise_mean'][idx0]
    after_als_diff_0 = als_data_0['specs_1_bg_noise_mean'][idx0] - als_data_0['specs_2_bg_noise_mean'][idx0]
    after_als_1 = als_data_1['specs_1_bg_noise_mean'][idx1]
    after_als_diff_1 = als_data_1['specs_1_bg_noise_mean'][idx1] - als_data_1['specs_2_bg_noise_mean'][idx1]
    
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    
    # First row (FSR 0)
    axs[0, 0].set_title("(a) Conventional \nbefore background removal", fontsize=name_size, fontname=font)
    axs[0, 0].plot(wavelengths_new, data_0['specs_1_bg_noise_mean'][idx0].T, color='plum')
    axs[0, 0].set_xlabel("Wavelength ($nm$)", fontsize=name_size, fontname=font, labelpad=x_label_padding)
    axs[0, 0].set_ylabel("Raman Intensity ($a.u.$)", fontsize=name_size, fontname=font, labelpad=y_label_padding)
    axs[0, 0].tick_params(axis='both', labelsize=tick_label_size)
    
    axs[0, 1].set_title("(b) SERDS \nbefore background removal", fontsize=name_size, fontname=font)
    axs[0, 1].plot(wavelengths_new, diff_0.T, color='plum')
    axs[0, 1].set_xlabel("Wavelength ($nm$)", fontsize=name_size, fontname=font, labelpad=x_label_padding)
    axs[0, 1].set_ylabel("", fontsize=name_size, fontname=font, labelpad=y_label_padding)
    axs[0, 1].tick_params(axis='both', labelsize=tick_label_size)
    
    axs[0, 2].set_title("(c) Conventional \nafter background removal", fontsize=name_size, fontname=font)
    axs[0, 2].plot(wavelengths_new, after_als_0.T, color='plum')
    axs[0, 2].set_xlabel("Wavelength ($nm$)", fontsize=name_size, fontname=font, labelpad=x_label_padding)
    axs[0, 2].set_ylabel("", fontsize=name_size, fontname=font, labelpad=y_label_padding)
    axs[0, 2].tick_params(axis='both', labelsize=tick_label_size)
    
    axs[0, 3].set_title("(d) SERDS \nafter background removal", fontsize=name_size, fontname=font)
    axs[0, 3].plot(wavelengths_new, after_als_diff_0.T, color='plum')
    axs[0, 3].set_xlabel("Wavelength ($nm$)", fontsize=name_size, fontname=font, labelpad=x_label_padding)
    axs[0, 3].set_ylabel("", fontsize=name_size, fontname=font, labelpad=y_label_padding)
    axs[0, 3].tick_params(axis='both', labelsize=tick_label_size)
    
    # Second row (FSR 1)
    axs[1, 0].set_title("(e) Conventional \nbefore background removal", fontsize=name_size, fontname=font)
    axs[1, 0].plot(wavelengths_new, data_1['specs_1_bg_noise_mean'][idx1].T, color='darkslateblue')
    axs[1, 0].set_xlabel("Wavelength ($nm$)", fontsize=name_size, fontname=font, labelpad=x_label_padding)
    axs[1, 0].set_ylabel("Raman Intensity ($a.u.$)", fontsize=name_size, fontname=font, labelpad=y_label_padding)
    axs[1, 0].tick_params(axis='both', labelsize=tick_label_size)
    
    axs[1, 1].set_title("(f) SERDS \nbefore background removal", fontsize=name_size, fontname=font)
    axs[1, 1].plot(wavelengths_new, diff_1.T, color='darkslateblue')
    axs[1, 1].set_xlabel("Wavelength ($nm$)", fontsize=name_size, fontname=font, labelpad=x_label_padding)
    axs[1, 1].set_ylabel("", fontsize=name_size, fontname=font, labelpad=y_label_padding)
    axs[1, 1].tick_params(axis='both', labelsize=tick_label_size)
    
    axs[1, 2].set_title("(g) Conventional \nafter background removal", fontsize=name_size, fontname=font)
    axs[1, 2].plot(wavelengths_new, after_als_1.T, color='darkslateblue')
    axs[1, 2].set_xlabel("Wavelength ($nm$)", fontsize=name_size, fontname=font, labelpad=x_label_padding)
    axs[1, 2].set_ylabel("", fontsize=name_size, fontname=font, labelpad=y_label_padding)
    axs[1, 2].tick_params(axis='both', labelsize=tick_label_size)
    
    axs[1, 3].set_title("(h) SERDS \nafter background removal", fontsize=name_size, fontname=font)
    axs[1, 3].plot(wavelengths_new, after_als_diff_1.T, color='darkslateblue')
    axs[1, 3].set_xlabel("Wavelength ($nm$)", fontsize=name_size, fontname=font, labelpad=x_label_padding)
    axs[1, 3].set_ylabel("", fontsize=name_size, fontname=font, labelpad=y_label_padding)
    axs[1, 3].tick_params(axis='both', labelsize=tick_label_size)
    
    for ax in axs.flat:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    plt.tight_layout(w_pad=0.5)
    os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)
    plt.savefig(fig_save_path, dpi=800)
    plt.show()
    return fig, axs

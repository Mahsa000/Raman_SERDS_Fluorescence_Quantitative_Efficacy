"""
Module: simulation_data.py
Description: This module contains functions to simulate various datasets (Dataset 1/2, 3, 4, and 4a)
             for Raman/SERDS experiments. The simulation includes scaling of input spectra, simulating
             decayed fluorescence with noise addition, and generating SERDS difference spectra.
             
             For Dataset 1 and 2, the difference is that Dataset 1 uses a decay_rate of 0 (no decay)
             and Dataset 2 uses a decay_rate of 0.005 (with decay).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.interpolate import interp1d

def generate_fluorescence_shapes(shift_points, original_fluorescence):
    """
    Generates a list of shifted fluorescence shapes based on the original fluorescence signal.
    
    For each value in shift_points, the function:
      - Computes a shift offset relative to the peak of the original fluorescence,
      - Creates a shifted x-axis,
      - Uses linear interpolation to extrapolate the fluorescence curve along the shifted axis.
    
    This is used in the simulation of datasets 4 and 4a to create variable fluorescence shapes.
    
    Parameters
    ----------
    shift_points : array-like
        Sequence of shift values used to generate different fluorescence shapes.
    original_fluorescence : array-like
        1D array of the original fluorescence spectrum.
    
    Returns
    -------
    fluorescence_shapes : list of np.ndarray
        List containing the generated (shifted) fluorescence shapes.
    """
    from scipy.integrate import trapz
    from scipy.interpolate import interp1d

    fluorescence_shapes = []
    for shift_point in shift_points:
        # Calculate the area under the curve for values less than the shift point (not used further here)
        area_before_shift = trapz(original_fluorescence[original_fluorescence < shift_point])
        # Determine the shift offset relative to the peak position
        peak_position = np.argmax(original_fluorescence)
        shift_offset = shift_point - peak_position
        # Create a shifted x-axis and interpolate to generate the new shape
        shifted_x = np.arange(len(original_fluorescence)) - shift_offset
        interpolator = interp1d(shifted_x, original_fluorescence, fill_value="extrapolate")
        shifted_fluorescence = interpolator(np.arange(len(original_fluorescence)))
        fluorescence_shapes.append(shifted_fluorescence)
    return fluorescence_shapes


# =============================================================================
# Function: simulate_dataset12
# =============================================================================
def simulate_dataset12(specs_1_cut, specs_2_cut, pure_fluorescence, wavelengths_new, 
                       lambda_1, lambda_2, norm_mean, decay_rate):
    """
    Simulates datasets for Dataset 1 and Dataset 2 by generating decayed fluorescence spectra
    and adding Poisson-based noise.

    This function scales the input spectra and pure fluorescence signals, then simulates 
    repeated acquisitions where the fluorescence contribution decays over time at a rate specified 
    by the 'decay_rate' parameter. Noise is added to mimic experimental conditions, and SERDS 
    difference spectra are computed.

    Parameters
    ----------
    specs_1_cut : np.ndarray
        2D array (n_samples x n_points) containing spectra from laser 1.
    specs_2_cut : np.ndarray
        2D array (n_samples x n_points) containing spectra from laser 2.
    pure_fluorescence : np.ndarray
        2D array containing the pure fluorescence signal.
    wavelengths_new : np.ndarray
        1D array of wavelength values (for plotting).
    lambda_1 : float
        Laser 1 wavelength (used for labeling).
    lambda_2 : float
        Laser 2 wavelength (used for labeling).
    norm_mean : callable
        A normalization function (reserved for potential normalization operations).
    decay_rate : float
        Decay rate for the fluorescence signal. Use 0 for Dataset 1 (no decay) or 0.005 for Dataset 2.

    Returns
    -------
    results_dict_orig : dict
        Dictionary keyed by each initial fluorescence level containing:
            "specs_1_bg_noise_mean", "specs_2_bg_noise_mean",
            "serds_bg_noise_mean", "specs_1_bg_noise", "serds_2d_noise".
    initial_F_levels : list
        List of initial fluorescence levels used.
    fluorescence_decayed_array_shape : tuple
        Shape of the fluorescence decayed array.
    """
    # Scale the input spectra and pure fluorescence
    specs_1_cut_scaled = specs_1_cut / 1048
    specs_2_cut_scaled = specs_2_cut / 1048
    pure_fluorescence_scaled = pure_fluorescence[:, 0] / (1.4342505360698254 * 1048)
    
    # Define time resolution parameters
    ncycles = 2          # Total number of cycles for the sine curve
    fs = 64              # Sampling frequency [Hz]
    resolution = ncycles * fs  # Number of repeats in the simulation
    start = 34           # Starting index for pure fluorescence

    
    # Initialize background arrays and fluorescence levels
    noise = True
    initial_F_levels = [0, 1, 10, 100, 1000, 2000, 3000, 4000]  
    specs_1_bg = np.zeros(specs_1_cut.shape)
    specs_2_bg = np.zeros(specs_2_cut.shape)
    
    for spectrum in range(specs_1_cut.shape[0]):
        specs_1_bg[spectrum, :] = specs_1_cut_scaled[spectrum, :]
        specs_2_bg[spectrum, :] = specs_2_cut_scaled[spectrum, :]
    
    # Set random seed for reproducibility
    np.random.seed(123)
    results_dict_orig = {}
    
    # Loop over initial fluorescence levels
    for initial_F_level in initial_F_levels:
        current_F_level = initial_F_level
        if noise:
            # Repeat background arrays along a new time dimension (resolution)
            specs_1_bg_repeats = np.repeat(specs_1_bg[:, :, np.newaxis], resolution, axis=2)
            specs_2_bg_repeats = np.repeat(specs_2_bg[:, :, np.newaxis], resolution, axis=2)
            noise_2d_1 = np.zeros(specs_1_bg_repeats.shape)
            noise_2d_2 = np.zeros(specs_2_bg_repeats.shape)
            noise_2d_aplid = np.zeros(specs_2_bg_repeats.shape)
            start = 34
            fluorescence_decayed_array = np.zeros((len(pure_fluorescence_scaled[start:]), specs_1_bg_repeats.shape[2]))
            
            for repeat_dim in range(specs_1_bg_repeats.shape[2]):
                current_F_level = current_F_level * (1 - decay_rate)
                fluorescence_decayed = pure_fluorescence_scaled[start:].T * current_F_level
                fluorescence_decayed_array[:, repeat_dim] = fluorescence_decayed
                
                specs_1_bg_repeats[:, :, repeat_dim] += pure_fluorescence_scaled[start:].T * current_F_level
                specs_2_bg_repeats[:, :, repeat_dim] += pure_fluorescence_scaled[start:].T * current_F_level
                
                # Add noise scaled with sqrt(n) for each repeat
                noise_2d_1[:, :, repeat_dim] = (np.random.poisson(1, specs_1_bg_repeats[:, :, repeat_dim].shape) - 1) * np.sqrt(specs_1_bg_repeats[:, :, repeat_dim])
                noise_2d_2[:, :, repeat_dim] = (np.random.poisson(1, specs_2_bg_repeats[:, :, repeat_dim].shape) - 1) * np.sqrt(specs_2_bg_repeats[:, :, repeat_dim])
                noise_2d_aplid[:, :, repeat_dim] = np.random.poisson(1, specs_2_bg_repeats[:, :, repeat_dim].shape) * np.sqrt((specs_1_bg_repeats[:, :, repeat_dim] + specs_2_bg_repeats[:, :, repeat_dim]) / 2)
            
            # Compute noise-added spectra and mean values
            specs_1_bg_noise = specs_1_bg_repeats + noise_2d_1
            specs_2_bg_noise = specs_2_bg_repeats + noise_2d_2
            specs_1_bg_noise_mean = np.mean(specs_1_bg_noise, axis=2)
            specs_2_bg_noise_mean = np.mean(specs_2_bg_noise, axis=2)
    
            serds_specs_1_bg_noise = specs_1_bg_repeats[:, :, ::2] + noise_2d_1[:, :, ::2]
            serds_specs_2_bg_noise = specs_2_bg_repeats[:, :, 1::2] + noise_2d_2[:, :, 1::2]
    
            specs_1_bg_mean = np.mean(serds_specs_1_bg_noise, axis=2)
            specs_2_bg_mean = np.mean(serds_specs_2_bg_noise, axis=2)
            serds_bg_noise_mean = specs_1_bg_mean - specs_2_bg_mean
            serds_2d_noise = serds_specs_1_bg_noise - serds_specs_2_bg_noise
            results_dict_orig[initial_F_level] = {
                "specs_1_bg_noise_mean": specs_1_bg_noise_mean,
                "specs_2_bg_noise_mean": specs_2_bg_noise_mean,
                "serds_bg_noise_mean": serds_bg_noise_mean,
                "specs_1_bg_noise": specs_1_bg_noise,
                "serds_2d_noise": serds_2d_noise
            }
    
        else:
            serds_bg = specs_1_bg - specs_2_bg     
            plt.figure()
            plt.plot(wavelengths_new, specs_1_bg[2], 'b', label=f'$\lambda1$={lambda_1}')
            plt.plot(wavelengths_new, specs_2_bg[2], 'r', label=f'$\lambda2$={lambda_2}')
            plt.title('Actual & Shifted Raman Spectra with BG|Fixed x axis', fontsize=12)
            plt.xlabel('Wavelength $[nm]$', fontsize=10)
            plt.ylabel('Intensity', fontsize=10)
            plt.legend()
            serds_bg_noise_norm = np.vstack([norm_mean(s) for s in serds_bg])
            results_dict_orig[initial_F_level] = {
                "specs_1_bg_noise_mean": serds_bg_noise_norm
            }
    
    print("Fluorescence decayed array shape:", fluorescence_decayed_array.shape)
    return results_dict_orig, initial_F_levels, fluorescence_decayed_array.shape

# =============================================================================
# Function: simulate_dataset3
# =============================================================================
def simulate_dataset3(specs_1_cut, specs_2_cut, pure_fluorescence, wavelengths_new, 
                      lambda_1, lambda_2, norm_mean, noise=True):
    """
    Generates simulated SERDS data for Dataset 3.

    Parameters
    ----------
    specs_1_cut : np.ndarray
        Input spectra from laser 1.
    specs_2_cut : np.ndarray
        Input spectra from laser 2.
    pure_fluorescence : np.ndarray
        Pure fluorescence signal.
    wavelengths_new : np.ndarray
        Wavelength values (for plotting).
    lambda_1 : float
        Laser 1 wavelength (for labeling).
    lambda_2 : float
        Laser 2 wavelength (for labeling).
    norm_mean : callable
        Normalization function applied to spectra.
    noise : bool, optional
        Whether to simulate and add noise. If False, only background subtraction is performed (default is True).

    Returns
    -------
    results_dict : dict
        When noise is True, a dict keyed by initial_F_level with keys:
            "specs_1_bg_noise_mean", "specs_2_bg_noise_mean", "serds_bg_noise_mean",
            "specs_1_bg_noise", "serds_2d_noise".
        When noise is False, a dict containing 'serds_bg' and 'serds_bg_noise_norm'.
    initial_F_levels : list
        List of initial fluorescence levels used.
    resolution : int
        The time resolution (number of repeats) used in the simulation.
    """
    # Scale the input arrays
    specs_1_cut_scaled = specs_1_cut / 1048
    specs_2_cut_scaled = specs_2_cut / 1048
    pure_fluorescence_scaled = pure_fluorescence[:, 0] / (1.4342505360698254 * 1048)

    # Define simulation parameters
    ncycles = 2          # Total number of cycles for the sine curve
    fs = 64              # Sampling frequency [Hz]
    resolution = ncycles * fs  # Number of repeats in the simulation
    start = 34           # Starting index for pure fluorescence

    # Define initial fluorescence levels
    initial_F_levels = [0, 1, 10, 100, 1000, 2000, 3000, 4000]

    # Initialize background arrays
    specs_1_bg = np.zeros_like(specs_1_cut)
    specs_2_bg = np.zeros_like(specs_2_cut)
    for spectrum in range(specs_1_cut.shape[0]):
        specs_1_bg[spectrum, :] = specs_1_cut_scaled[spectrum, :]
        specs_2_bg[spectrum, :] = specs_2_cut_scaled[spectrum, :]

    # Define scaling factors and set seed for reproducibility
    scaling_factors = [-0.01, 0.01, -0.1, 0.1, -0.5, 0.5]
    np.random.seed(123)
    results_dict = {}

    if noise:
        # Loop over each initial fluorescence level
        for initial_F_level in initial_F_levels:
            specs_1_bg_repeats = np.repeat(specs_1_bg[:, :, np.newaxis], resolution, axis=2)
            specs_2_bg_repeats = np.repeat(specs_2_bg[:, :, np.newaxis], resolution, axis=2)
            noise_2d_1 = np.zeros_like(specs_1_bg_repeats)
            noise_2d_2 = np.zeros_like(specs_2_bg_repeats)
            noise_2d_aplid = np.zeros_like(specs_2_bg_repeats)
            decay_rate = 0.005
            fluorescence_decayed_varying = np.zeros((pure_fluorescence_scaled[start:].shape[0], resolution))
            
            # Loop over samples and time repeats to simulate decay and add fluorescence
            for sample_index in range(specs_1_bg.shape[0]):
                scaling_factor = np.random.choice(scaling_factors)
                scaled_fluorescence = pure_fluorescence_scaled[start:].T * (1 + scaling_factor)
                current_F_level = initial_F_level
                for repeat_dim in range(resolution):
                    current_F_level *= (1 - decay_rate)
                    fluorescence_decayed = scaled_fluorescence * current_F_level
                    fluorescence_decayed_varying[:, repeat_dim] = fluorescence_decayed
                    specs_1_bg_repeats[sample_index, :, repeat_dim] += scaled_fluorescence * current_F_level
                    specs_2_bg_repeats[sample_index, :, repeat_dim] += scaled_fluorescence * current_F_level

            # Add Poisson-based noise to each repeat
            for repeat_dim in range(resolution):
                shape1 = specs_1_bg_repeats[:, :, repeat_dim].shape
                shape2 = specs_2_bg_repeats[:, :, repeat_dim].shape
                noise_2d_1[:, :, repeat_dim] = (np.random.poisson(1, shape1) - 1) * np.sqrt(specs_1_bg_repeats[:, :, repeat_dim])
                noise_2d_2[:, :, repeat_dim] = (np.random.poisson(1, shape2) - 1) * np.sqrt(specs_2_bg_repeats[:, :, repeat_dim])
                noise_2d_aplid[:, :, repeat_dim] = np.random.poisson(1, shape2) * np.sqrt((specs_1_bg_repeats[:, :, repeat_dim] +
                                                                                          specs_2_bg_repeats[:, :, repeat_dim]) / 2)
            specs_1_bg_noise = specs_1_bg_repeats + noise_2d_1
            specs_2_bg_noise = specs_2_bg_repeats + noise_2d_2
            specs_1_bg_noise_mean = np.mean(specs_1_bg_noise, axis=2)
            specs_2_bg_noise_mean = np.mean(specs_2_bg_noise, axis=2)
            serds_specs_1_bg_noise = specs_1_bg_repeats[:, :, ::2] + noise_2d_1[:, :, ::2]
            serds_specs_2_bg_noise = specs_2_bg_repeats[:, :, 1::2] + noise_2d_2[:, :, 1::2]
            specs_1_bg_mean = np.mean(serds_specs_1_bg_noise, axis=2)
            specs_2_bg_mean = np.mean(serds_specs_2_bg_noise, axis=2)
            serds_bg_noise_mean = specs_1_bg_mean - specs_2_bg_mean
            serds_2d_noise = serds_specs_1_bg_noise - serds_specs_2_bg_noise
            
            results_dict[initial_F_level] = {
                "specs_1_bg_noise_mean": specs_1_bg_noise_mean,
                "specs_2_bg_noise_mean": specs_2_bg_noise_mean,
                "serds_bg_noise_mean": serds_bg_noise_mean,
                "specs_1_bg_noise": specs_1_bg_noise,
                "serds_2d_noise": serds_2d_noise
            }
    else:
        serds_bg = specs_1_bg - specs_2_bg     
        serds_bg_noise_norm = np.vstack([norm_mean(s) for s in serds_bg])
        results_dict = {"serds_bg": serds_bg, "serds_bg_noise_norm": serds_bg_noise_norm}
        plt.figure()
        plt.plot(wavelengths_new, specs_1_bg[2], 'b', label=f'$\lambda1$={lambda_1}')
        plt.plot(wavelengths_new, specs_2_bg[2], 'r', label=f'$\lambda2$={lambda_2}')
        plt.title('Actual & Shifted Raman Spectra with BG|Fixed x axis', fontsize=12)
        plt.xlabel('Wavelength [nm]', fontsize=10)
        plt.ylabel('Intensity', fontsize=10)
        plt.legend()
        plt.show()

    return results_dict, initial_F_levels, resolution

# =============================================================================
# Function: simulate_dataset4
# =============================================================================
def simulate_dataset4(specs_1_cut, specs_2_cut, pure_fluorescence, 
                      wavelengths_new, lambda_1, lambda_2, norm_mean):
    """
    Generates synthetic data for Dataset 4 by adding fluorescence shapes and noise 
    to the background spectra.

    The function scales the input spectra and pure fluorescence, generates a series of
    fluorescence shapes (using an external function 'generate_fluorescence_shapes'), and
    then, for each fluorescence level, applies a decaying fluorescence contribution across 
    repeated acquisitions. Poisson-based noise is added and SERDS difference spectra are computed.

    Parameters
    ----------
    specs_1_cut : np.ndarray
        2D array (n_samples x n_points) for laser 1 background spectra.
    specs_2_cut : np.ndarray
        2D array (n_samples x n_points) for laser 2 background spectra.
    pure_fluorescence : np.ndarray
        2D array containing the pure fluorescence signal.
    wavelengths_new : np.ndarray
        1D array of wavelengths for plotting.
    lambda_1 : float
        Laser 1 wavelength (for labeling).
    lambda_2 : float
        Laser 2 wavelength (for labeling).
    norm_mean : callable
        Normalization function for spectra.

    Returns
    -------
    plsr_results_df_shape_new : dict
        Dictionary containing noisy spectra and SERDS results for each initial fluorescence level.
    shapes_dict : dict
        Dictionary with the actual decayed fluorescence shapes used per sample and repeat.
    """
    # Scale the spectra and fluorescence
    specs_1_cut_scaled = specs_1_cut / 1048
    specs_2_cut_scaled = specs_2_cut / 1048
    pure_fluorescence_scaled = pure_fluorescence[:, 0] / (1.4342505360698254 * 1048)

    # Define shift points and generate fluorescence shapes 
    shift_points = np.arange(0, 500, 5)
    shapes = generate_fluorescence_shapes(shift_points, pure_fluorescence_scaled)

    # Define time parameters
    ncycles = 2          # Total number of cycles for the sine curve
    fs = 64              # Sampling frequency [Hz]
    resolution = ncycles * fs  # Number of repeats in the simulation
    start = 34           # Starting index for pure fluorescence

    noise = True
    initial_F_levels = [0, 1, 10, 100, 1000, 2000, 3000, 4000]

    # Prepare baseline spectra arrays
    specs_1_bg = np.zeros(specs_1_cut.shape)
    specs_2_bg = np.zeros(specs_2_cut.shape)
    for spectrum in range(specs_1_cut.shape[0]):
        specs_1_bg[spectrum, :] = specs_1_cut_scaled[spectrum, :]
        specs_2_bg[spectrum, :] = specs_2_cut_scaled[spectrum, :]

    np.random.seed(123)
    plsr_results_df_shape_new = {}
    shapes_dict = {}
    decay_rate = 0.005
    start = 34

    for initial_F_level in initial_F_levels:
        specs_1_bg_repeats = np.repeat(specs_1_bg[:, :, np.newaxis], resolution, axis=2)
        specs_2_bg_repeats = np.repeat(specs_2_bg[:, :, np.newaxis], resolution, axis=2)
        samples_shapes_for_level = []

        for sample_index in range(specs_1_bg.shape[0]):
            sample_shape = shapes[np.random.choice(len(shapes))]
            repeats_shapes_for_sample = []
            for repeat_dim in range(resolution):
                current_F_level = initial_F_level * (1 - decay_rate * repeat_dim)
                fluorescence_decayed = sample_shape[start:].T * current_F_level
                repeats_shapes_for_sample.append(fluorescence_decayed)
                specs_1_bg_repeats[sample_index, :, repeat_dim] += fluorescence_decayed
                specs_2_bg_repeats[sample_index, :, repeat_dim] += fluorescence_decayed
            samples_shapes_for_level.append(repeats_shapes_for_sample)

        shapes_dict[initial_F_level] = samples_shapes_for_level

        noise_2d_1 = (np.random.poisson(1, specs_1_bg_repeats.shape) - 1) * np.sqrt(specs_1_bg_repeats)
        noise_2d_2 = (np.random.poisson(1, specs_2_bg_repeats.shape) - 1) * np.sqrt(specs_2_bg_repeats)

        specs_1_bg_noise = specs_1_bg_repeats + noise_2d_1
        specs_2_bg_noise = specs_2_bg_repeats + noise_2d_2

        specs_1_bg_noise_mean = np.mean(specs_1_bg_noise, axis=2)
        specs_2_bg_noise_mean = np.mean(specs_2_bg_noise, axis=2)

        serds_specs_1_bg_noise = specs_1_bg_repeats[:, :, ::2] + noise_2d_1[:, :, ::2]
        serds_specs_2_bg_noise = specs_2_bg_repeats[:, :, 1::2] + noise_2d_2[:, :, 1::2]

        specs_1_bg_mean = np.mean(serds_specs_1_bg_noise, axis=2)
        specs_2_bg_mean = np.mean(serds_specs_2_bg_noise, axis=2)
        serds_bg_noise_mean = specs_1_bg_mean - specs_2_bg_mean
        serds_2d_noise = serds_specs_1_bg_noise - serds_specs_2_bg_noise

        plsr_results_df_shape_new[initial_F_level] = {
            "specs_1_bg_noise_mean": specs_1_bg_noise_mean,
            "specs_2_bg_noise_mean": specs_2_bg_noise_mean,
            "serds_bg_noise_mean": serds_bg_noise_mean,
            "specs_1_bg_noise": specs_1_bg_noise,
            "serds_2d_noise": serds_2d_noise
        }

    return plsr_results_df_shape_new, initial_F_levels, shapes_dict

# =============================================================================
# Function: simulate_dataset4a
# =============================================================================
def simulate_dataset4a(specs_1_cut, specs_2_cut, pure_fluorescence, 
                       wavelengths_new, lambda_1, lambda_2, norm_mean):
    """
    Replicates the simulation for Dataset 4a by generating decayed fluorescence 
    using two slightly varied fluorescence shapes per sample and adding noise.

    For each initial fluorescence level, the function randomly selects a base fluorescence shape
    and a variant (within ±5 indices) for the second laser. Decayed fluorescence is added to the 
    baseline spectra, Poisson-based noise is applied, and SERDS difference spectra are computed.

    Parameters
    ----------
    specs_1_cut : np.ndarray
        2D array (n_samples x n_points) for laser 1 background spectra.
    specs_2_cut : np.ndarray
        2D array (n_samples x n_points) for laser 2 background spectra.
    pure_fluorescence : np.ndarray
        2D array containing the pure fluorescence signal.
    wavelengths_new : np.ndarray
        1D array of wavelengths for plotting.
    lambda_1 : float
        Laser 1 wavelength (for labeling).
    lambda_2 : float
        Laser 2 wavelength (for labeling).
    norm_mean : callable
        Normalization function for spectra.

    Returns
    -------
    plsr_results_df_shape_new : dict
        Dictionary keyed by each fluorescence level containing SERDS and noise-added spectra.
    initial_F_levels : list
        List of initial fluorescence levels used.
    shapes_dict : dict
        Dictionary storing the chosen base and variant shapes for each sample.
    """
    # Scale the spectra and pure fluorescence
    specs_1_cut_scaled = specs_1_cut / 1048
    specs_2_cut_scaled = specs_2_cut / 1048
    pure_fluorescence_scaled = pure_fluorescence[:, 0] / (1.4342505360698254 * 1048)
    
    # Define shift points and generate fluorescence shapes
    shift_points = np.arange(0, 500, 5)
    shapes = generate_fluorescence_shapes(shift_points, pure_fluorescence_scaled)
    
    # Define time resolution and related parameters
    ncycles = 2          # Total number of cycles for the sine curve
    fs = 64              # Sampling frequency [Hz]
    resolution = ncycles * fs  # Number of repeats in the simulation
    start = 34           # Starting index for pure fluorescence
    
    noise = True
    initial_F_levels = [0, 1, 10, 100, 1000, 2000, 3000, 4000]
    
    # Create baseline spectra arrays
    specs_1_bg = np.zeros(specs_1_cut.shape)
    specs_2_bg = np.zeros(specs_2_cut.shape)
    for spectrum in range(specs_1_cut.shape[0]):
        specs_1_bg[spectrum, :] = specs_1_cut_scaled[spectrum, :]
        specs_2_bg[spectrum, :] = specs_2_cut_scaled[spectrum, :]
    
    np.random.seed(123)
    plsr_results_df_shape_new = {}
    shapes_dict = {} 
    decay_rate = 0.005
    shape_variation_range = 5  # Allowed variation range for shape selection
    start = 34
    
    for initial_F_level in initial_F_levels:
        specs_1_bg_repeats = np.repeat(specs_1_bg[:, :, np.newaxis], resolution, axis=2)
        specs_2_bg_repeats = np.repeat(specs_2_bg[:, :, np.newaxis], resolution, axis=2)
        
        samples_shapes_for_level = {}
    
        for sample_index in range(specs_1_bg.shape[0]):
            base_shape_index = np.random.choice(len(shapes))
            shape_1 = shapes[base_shape_index]
    
            min_index = max(base_shape_index - shape_variation_range, 0)
            max_index = min(base_shape_index + shape_variation_range, len(shapes) - 1)
            shape_2_index = np.random.choice(range(min_index, max_index + 1))
            shape_2 = shapes[shape_2_index]
    
            samples_shapes_for_level[sample_index] = {'laser_1': shape_1, 'laser_2': shape_2}
    
            for repeat_dim in range(resolution):
                current_F_level = initial_F_level * (1 - decay_rate * repeat_dim)
                fluorescence_decayed_1 = shape_1[start:].T * current_F_level
                fluorescence_decayed_2 = shape_2[start:].T * current_F_level
                
                specs_1_bg_repeats[sample_index, :, repeat_dim] += fluorescence_decayed_1
                specs_2_bg_repeats[sample_index, :, repeat_dim] += fluorescence_decayed_2
    
        shapes_dict[initial_F_level] = samples_shapes_for_level
    
        noise_2d_1 = (np.random.poisson(1, specs_1_bg_repeats.shape) - 1) * np.sqrt(specs_1_bg_repeats)
        noise_2d_2 = (np.random.poisson(1, specs_2_bg_repeats.shape) - 1) * np.sqrt(specs_2_bg_repeats)
    
        specs_1_bg_noise = specs_1_bg_repeats + noise_2d_1
        specs_2_bg_noise = specs_2_bg_repeats + noise_2d_2
        specs_1_bg_noise_mean = np.mean(specs_1_bg_noise, axis=2)
        specs_2_bg_noise_mean = np.mean(specs_2_bg_noise, axis=2)
    
        serds_specs_1_bg_noise = specs_1_bg_repeats[:, :, ::2] + noise_2d_1[:, :, ::2]
        serds_specs_2_bg_noise = specs_2_bg_repeats[:, :, 1::2] + noise_2d_2[:, :, 1::2]
    
        specs_1_bg_mean = np.mean(serds_specs_1_bg_noise, axis=2)
        specs_2_bg_mean = np.mean(serds_specs_2_bg_noise, axis=2)
        serds_bg_noise_mean = specs_1_bg_mean - specs_2_bg_mean
        serds_2d_noise = serds_specs_1_bg_noise - serds_specs_2_bg_noise
    
        plsr_results_df_shape_new[initial_F_level] = {
            "specs_1_bg_noise_mean": specs_1_bg_noise_mean,
            "specs_2_bg_noise_mean": specs_2_bg_noise_mean,
            "serds_bg_noise_mean": serds_bg_noise_mean,
            "specs_1_bg_noise": specs_1_bg_noise,
            "serds_2d_noise": serds_2d_noise
        }
    
    return plsr_results_df_shape_new, initial_F_levels, shapes_dict


def simulate_dataset5(specs_1_cut, specs_2_cut, pure_fluorescence, wavelengths_new,
                      lambda1, lambda2, norm_mean, gen_y, combinations=None):
    """
    Generates heterogeneous Dataset 5 by combining paired FSR levels from simulated data.

    This function is self-contained—it first generates simulated data using 
    simulate_dataset12 (with decay_rate=0.005, as used in Dataset 2) and then, for each 
    specified pair of initial fluorescence levels (FSRs), it randomly selects half of the samples 
    from each FSR group and concatenates them to create heterogeneous samples. This approach 
    enables studying combination effects (e.g., mixing low and high fluorescence levels) in a single 
    dataset. More complex heterogeneity cases (using additional FSR levels or different proportions) 
    can be implemented by modifying the `combinations` parameter.

    Parameters
    ----------
    specs_1_cut : np.ndarray
        2D array (n_samples x n_points) of spectra from laser 1.
    specs_2_cut : np.ndarray
        2D array (n_samples x n_points) of spectra from laser 2.
    pure_fluorescence : np.ndarray
        2D array containing the pure fluorescence signal.
    wavelengths_new : np.ndarray
        1D array of wavelength values.
    lambda1 : float
        Wavelength for laser 1 (used for labeling).
    lambda2 : float
        Wavelength for laser 2 (used for labeling).
    norm_mean : callable
        A normalization function for the spectra.
    gen_y : array-like
        Target variable values corresponding to the samples. It is assumed that the targets
        are consistent across all FSR levels.
    combinations : list of tuple, optional
        List of FSR level pairs to combine. The default is:
        [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]

    Returns
    -------
    dataset5_dict : dict
        Dictionary keyed by each FSR pair (tuple) containing the concatenated heterogeneous data.
        Each value is a dictionary with the following keys:
          - "specs_1_bg_noise_mean"
          - "specs_2_bg_noise_mean"
          - "serds_bg_noise_mean"
          - "specs_1_bg_noise"
          - "serds_2d_noise"
          - "y"  (the corresponding target variable values for the selected samples)
    """
    # Use default combinations if none provided
    if combinations is None:
        combinations = [(0, 1), (1, 10), (10, 100), (100, 1000),
                        (1000, 2000), (2000, 3000), (3000, 4000)]
    
    # Generate simulated data (Dataset 2) with decay_rate 0.005
    results_dict, initial_F_levels, _ = simulate_dataset12(
        specs_1_cut, specs_2_cut, pure_fluorescence, wavelengths_new,
        lambda1, lambda2, norm_mean, decay_rate=0.005)
    
    dataset5_dict = {}
    np.random.seed(123)
    
    # For each specified FSR combination, randomly select half the samples from each group
    for FSR1, FSR2 in combinations:
        data1 = results_dict[FSR1]
        data2 = results_dict[FSR2]
        
        n_samples1 = data1['serds_bg_noise_mean'].shape[0]
        n_samples2 = data2['serds_bg_noise_mean'].shape[0]
        half_n1 = n_samples1 // 2
        half_n2 = n_samples2 // 2
        
        indices1 = np.random.choice(n_samples1, size=half_n1, replace=False)
        indices2 = np.random.choice(n_samples2, size=half_n2, replace=False)
        
        # Build selected data dictionaries for each FSR level
        selected_data1 = {
            "specs_1_bg_noise_mean": data1['specs_1_bg_noise_mean'][indices1],
            "specs_2_bg_noise_mean": data1['specs_2_bg_noise_mean'][indices1],
            "serds_bg_noise_mean": data1['serds_bg_noise_mean'][indices1],
            "specs_1_bg_noise": data1['specs_1_bg_noise'][indices1],
            "serds_2d_noise": data1['serds_2d_noise'][indices1],
            "y": gen_y[indices1]
        }
        selected_data2 = {
            "specs_1_bg_noise_mean": data2['specs_1_bg_noise_mean'][indices2],
            "specs_2_bg_noise_mean": data2['specs_2_bg_noise_mean'][indices2],
            "serds_bg_noise_mean": data2['serds_bg_noise_mean'][indices2],
            "specs_1_bg_noise": data2['specs_1_bg_noise'][indices2],
            "serds_2d_noise": data2['serds_2d_noise'][indices2],
            "y": gen_y[indices2]
        }
        
        # Concatenate data from the two FSR levels
        concatenated_data = {
            "specs_1_bg_noise_mean": np.concatenate([selected_data1["specs_1_bg_noise_mean"],
                                                       selected_data2["specs_1_bg_noise_mean"]], axis=0),
            "specs_2_bg_noise_mean": np.concatenate([selected_data1["specs_2_bg_noise_mean"],
                                                       selected_data2["specs_2_bg_noise_mean"]], axis=0),
            "serds_bg_noise_mean": np.concatenate([selected_data1["serds_bg_noise_mean"],
                                                     selected_data2["serds_bg_noise_mean"]], axis=0),
            "specs_1_bg_noise": np.concatenate([selected_data1["specs_1_bg_noise"],
                                                  selected_data2["specs_1_bg_noise"]], axis=0),
            "serds_2d_noise": np.concatenate([selected_data1["serds_2d_noise"],
                                               selected_data2["serds_2d_noise"]], axis=0),
            "y": np.concatenate([selected_data1["y"], selected_data2["y"]], axis=0)
        }
        dataset5_dict[(FSR1, FSR2)] = concatenated_data
    
    return dataset5_dict

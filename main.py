#!/usr/bin/env python
"""
main.py
-------
Entry point for the project. This script orchestrates dataset selection,
simulation, optional preprocessing (including advanced ALS extrapolation for datasets 4/4a),
modeling (PLSR), and plotting (feature importance, calibration, and paper figure for Dataset 5).

The available datasets are:
    1. Dataset 1: Simulated spectra with decay_rate = 0.
    2. Dataset 2: Simulated spectra with decay_rate = 0.005.
    3. Dataset 3: Simulated spectra with fluctuating spectral intensity.
    4. Dataset 4: Simulated spectra with variable fluorescence shapes.
    4a. Dataset 4a: SERDS-only scenario with independently varying fluorescence shapes.
    5. Dataset 5: Heterogeneous samples combining paired FSR levels.
"""

import os
import sys
sys.path.append('/Users/mahsazarei/Grantlab Brugada Dropbox/TerraMera/Gitlab/Mahsa/AA_APLID_SERDS/SERDS_CONV_Simulation_Project')
from helper_functions_APLID import *

# Import utility functions
from utils.file_io import save_simulation_data, load_simulation_data
from utils.datasets import get_dataset_info, get_simulation_data

# Import simulation functions
from simulation.simulation_data import (
    simulate_dataset12, simulate_dataset3, simulate_dataset4, simulate_dataset4a, simulate_dataset5
)

# Import preprocessing functions
from preprocessing.pre_processing import (
    preprocess_DWT, preprocess_ALS, preprocess_als_extrapolate
)

# Import modeling functions
from modeling.modeling import run_pls_modeling_all, run_pls_modeling_dataset5

# Import plotting functions
from plotting.plotting import (
    plot_feature_importance, plot_calibration, 
    plot_feature_importance_dataset5, plot_calibration_dataset5, plot_dataset5_paper_figure
)

# Assume norm_mean and asymmetric_least_squares are available (either defined globally or imported)
# They should be imported if defined in separate modules.
# For this example, we assume they are imported from their corresponding modules.
from preprocessing.pre_processing import asymmetric_least_squares  # if defined here
# from some_module import norm_mean   # Ensure norm_mean is defined appropriately

def main():
    # Dataset selection
    datasets = get_dataset_info()
    print("Available Datasets:")
    for key, desc in datasets.items():
        print(f"  Dataset {key}: {desc}")
    dataset_key = input("Select a dataset (e.g., 1,2,3,4,4a,5): ").strip()
    print(f"You selected Dataset {dataset_key}: {datasets.get(dataset_key, 'Unknown dataset')}")
    
    # Retrieve simulation parameters (e.g., spectra, fluorescence, wavelengths, lambda values)
    specs_1_cut, specs_2_cut, pure_fluorescence, wavelengths_new, lambda_1, lambda_2 = get_simulation_data()
    
    # Set number of PLSR components (Dataset 3 uses 2; others use 1)
    n_components = 2 if dataset_key == "3" else 1
    dataset_prefix = f"Dataset{dataset_key}"
    
    # Create folder for results for this dataset
    results_folder = os.path.join("Results", dataset_prefix)
    os.makedirs(results_folder, exist_ok=True)
    
    # Simulation data file
    sim_data_filename = os.path.join(results_folder, f"GeneratedData_{dataset_prefix}.pkl")
    
    if os.path.exists(sim_data_filename):
        sim_data = load_simulation_data(sim_data_filename)
        results_dict = sim_data["results_dict"]
        initial_F_levels = sim_data["initial_F_levels"]
        fluoro_shape = sim_data["fluoro_shape"]
        print("Loaded simulation data from file.")
    else:
        if dataset_key == "1":
            results_dict, initial_F_levels, fluoro_shape = simulate_dataset12(
                specs_1_cut, specs_2_cut, pure_fluorescence, wavelengths_new,
                lambda_1, lambda_2, norm_mean, decay_rate=0.0
            )
        elif dataset_key == "2":
            results_dict, initial_F_levels, fluoro_shape = simulate_dataset12(
                specs_1_cut, specs_2_cut, pure_fluorescence, wavelengths_new,
                lambda_1, lambda_2, norm_mean, decay_rate=0.005
            )
        elif dataset_key == "3":
            results_dict, initial_F_levels, fluoro_shape = simulate_dataset3(
                specs_1_cut, specs_2_cut, pure_fluorescence, wavelengths_new,
                lambda_1, lambda_2, norm_mean
            )
        elif dataset_key == "4":
            results_dict, initial_F_levels, fluoro_shape = simulate_dataset4(
                specs_1_cut, specs_2_cut, pure_fluorescence, wavelengths_new,
                lambda_1, lambda_2, norm_mean
            )
        elif dataset_key == "4a":
            results_dict, initial_F_levels, fluoro_shape = simulate_dataset4a(
                specs_1_cut, specs_2_cut, pure_fluorescence, wavelengths_new,
                lambda_1, lambda_2, norm_mean
            )
        elif dataset_key == "5":
            # For Dataset 5, simulate Dataset 2 first and then combine paired FSR levels.
            results_dict_2, init_F_levels_2, fluoro_shape_2 = simulate_dataset12(
                specs_1_cut, specs_2_cut, pure_fluorescence, wavelengths_new,
                lambda_1, lambda_2, norm_mean, decay_rate=0.005
            )
            # Assume gen_y (target variable) is defined or generated; it must be available.
            # For example, gen_y = ... (load or generate your target variable)
            from some_target_module import gen_y  # Ensure gen_y is available
            results_dict = simulate_dataset5(results_dict_2, gen_y)
            initial_F_levels = list(results_dict.keys())
            fluoro_shape = fluoro_shape_2
        else:
            print("Invalid dataset selection. Exiting.")
            return
        
        sim_data = {"results_dict": results_dict, "initial_F_levels": initial_F_levels, "fluoro_shape": fluoro_shape}
        save_simulation_data(sim_data_filename, sim_data)
        print(f"Simulation data saved to {sim_data_filename}")
    
    # Decide what results to generate
    if dataset_key != "5":
        print("\nWhat results would you like to generate?")
        print("  1: Raw results (Simulation, PLSR Modeling, Feature Importance, Calibration)")
        print("  2: Preprocessed results (choose DWT or ALS)")
        print("  3: All (both raw and one preprocessed option)")
        result_choice = input("Enter your choice (1, 2, or 3): ").strip()
        
        if result_choice == "1":
            prefix = f"{dataset_prefix}_Raw"
            csv_dir = os.path.join(results_folder, f"{prefix}_PLSR")
            plsr_results_df, max_argmax_rmsep_df, feat_imp_dict = run_pls_modeling_all(
                results_dict, gen_y, initial_F_levels, test_size=0.2, random_state=42,
                n_components=n_components, csv_save_dir=csv_dir)
            fig1, axs1 = plot_feature_importance(
                results_dict, initial_F_levels, wavelengths_new, feat_imp_dict, max_argmax_rmsep_df,
                fig_save_path=os.path.join(results_folder, f"{prefix}_Feature_Importance.pdf"))
            fig2, axs2 = plot_calibration(
                plsr_results_df, calib_save_path=os.path.join(results_folder, f"{prefix}_calibration.pdf"))
        
        elif result_choice == "2":
            print("\nChoose a preprocessing method:")
            print("  1: DWT")
            print("  2: ALS (Extrapolate version for datasets 4/4a)")
            prep_choice = input("Enter your choice (1 or 2): ").strip()
            if prep_choice == "1":
                prefix = f"{dataset_prefix}_DWT"
                csv_dir = os.path.join(results_folder, f"{prefix}_PLSR")
                scales_dict = {
                    "1": [6, 6, 5, 5, 5, 5, 5, 5],
                    "2": [6, 6, 5, 5, 5, 5, 5, 5],
                    "3": [6, 6, 5, 5, 6, 6, 5, 5],
                    "4": [6, 6, 5, 5, 5, 5, 5, 5],
                    "4a": [6, 6, 5, 5, 5, 5, 5, 5]
                }
                scales = scales_dict.get(dataset_key, [6, 6, 5, 5, 5, 5, 5, 5])
                dwt_preprocessed = preprocess_DWT(results_dict, initial_F_levels, scales, dwt_iterative_bg_rm)
                plsr_results_dwt, max_argmax_rmsep_dwt, feat_imp_dwt = run_pls_modeling_all(
                    dwt_preprocessed, gen_y, initial_F_levels, test_size=0.2, random_state=42,
                    n_components=n_components, csv_save_dir=csv_dir)
                fig_dwt, axs_dwt = plot_feature_importance(
                    dwt_preprocessed, initial_F_levels, wavelengths_new, feat_imp_dwt, max_argmax_rmsep_dwt,
                    fig_save_path=os.path.join(results_folder, f"{prefix}_Feature_Importance.pdf"))
                fig_cal_dwt, axs_cal_dwt = plot_calibration(
                    plsr_results_dwt, calib_save_path=os.path.join(results_folder, f"{prefix}_calibration.pdf"))
            elif prep_choice == "2":
                prefix = f"{dataset_prefix}_ALS"
                csv_dir = os.path.join(results_folder, f"{prefix}_PLSR")
                pad_length = 200
                als_preprocessed = preprocess_als_extrapolate(results_dict, initial_F_levels, pad_length, asymmetric_least_squares)
                plsr_results_als, max_argmax_rmsep_als, feat_imp_als = run_pls_modeling_all(
                    als_preprocessed, gen_y, initial_F_levels, test_size=0.2, random_state=42,
                    n_components=n_components, csv_save_dir=csv_dir)
                fig_als, axs_als = plot_feature_importance(
                    als_preprocessed, initial_F_levels, wavelengths_new, feat_imp_als, max_argmax_rmsep_als,
                    fig_save_path=os.path.join(results_folder, f"{prefix}_Feature_Importance.pdf"))
                fig_cal_als, axs_cal_als = plot_calibration(
                    plsr_results_als, calib_save_path=os.path.join(results_folder, f"{prefix}_calibration.pdf"))
            else:
                print("Invalid choice. Exiting.")
                return
        elif result_choice == "3":
            prefix_raw = f"{dataset_prefix}_Raw"
            csv_dir = os.path.join(results_folder, f"{prefix_raw}_PLSR")
            plsr_results_df, max_argmax_rmsep_df, feat_imp_dict = run_pls_modeling_all(
                results_dict, gen_y, initial_F_levels, test_size=0.2, random_state=42,
                n_components=n_components, csv_save_dir=csv_dir)
            fig1, axs1 = plot_feature_importance(
                results_dict, initial_F_levels, wavelengths_new, feat_imp_dict, max_argmax_rmsep_df,
                fig_save_path=os.path.join(results_folder, f"{prefix_raw}_Feature_Importance.pdf"))
            fig2, axs2 = plot_calibration(
                plsr_results_df, calib_save_path=os.path.join(results_folder, f"{prefix_raw}_calibration.pdf"))
            print("\nNow choose a preprocessing method for additional results:")
            print("  1: DWT")
            print("  2: ALS (Extrapolate version for datasets 4/4a)")
            prep_choice = input("Enter your choice (1 or 2): ").strip()
            if prep_choice == "1":
                prefix_prep = f"{dataset_prefix}_DWT"
                csv_dir = os.path.join(results_folder, f"{prefix_prep}_PLSR")
                scales_dict = {
                    "1": [6, 6, 5, 5, 5, 5, 5, 5],
                    "2": [6, 6, 5, 5, 5, 5, 5, 5],
                    "3": [6, 6, 5, 5, 6, 6, 5, 5],
                    "4": [6, 6, 5, 5, 5, 5, 5, 5],
                    "4a": [6, 6, 5, 5, 5, 5, 5, 5]
                }
                scales = scales_dict.get(dataset_key, [6, 6, 5, 5, 5, 5, 5, 5])
                dwt_preprocessed = preprocess_DWT(results_dict, initial_F_levels, scales, dwt_iterative_bg_rm)
                plsr_results_dwt, max_argmax_rmsep_dwt, feat_imp_dwt = run_pls_modeling_all(
                    dwt_preprocessed, gen_y, initial_F_levels, test_size=0.2, random_state=42,
                    n_components=n_components, csv_save_dir=csv_dir)
                fig_dwt, axs_dwt = plot_feature_importance(
                    dwt_preprocessed, initial_F_levels, wavelengths_new, feat_imp_dwt, max_argmax_rmsep_dwt,
                    fig_save_path=os.path.join(results_folder, f"{prefix_prep}_Feature_Importance.pdf"))
                fig_cal_dwt, axs_cal_dwt = plot_calibration(
                    plsr_results_dwt, calib_save_path=os.path.join(results_folder, f"{prefix_prep}_calibration.pdf"))
            elif prep_choice == "2":
                prefix_prep = f"{dataset_prefix}_ALS"
                csv_dir = os.path.join(results_folder, f"{prefix_prep}_PLSR")
                pad_length = 200
                als_preprocessed = preprocess_als_extrapolate(results_dict, initial_F_levels, pad_length, asymmetric_least_squares)
                plsr_results_als, max_argmax_rmsep_als, feat_imp_als = run_pls_modeling_all(
                    als_preprocessed, gen_y, initial_F_levels, test_size=0.2, random_state=42,
                    n_components=n_components, csv_save_dir=csv_dir)
                fig_als, axs_als = plot_feature_importance(
                    als_preprocessed, initial_F_levels, wavelengths_new, feat_imp_als, max_argmax_rmsep_als,
                    fig_save_path=os.path.join(results_folder, f"{prefix_prep}_Feature_Importance.pdf"))
                fig_cal_als, axs_cal_als = plot_calibration(
                    plsr_results_als, calib_save_path=os.path.join(results_folder, f"{prefix_prep}_calibration.pdf"))
            else:
                print("Invalid choice. Exiting.")
                return
        else:
            print("Invalid result choice. Exiting.")
            return
    else:
        # For Dataset 5, use dedicated functions.
        prefix = f"{dataset_prefix}_Raw"
        csv_dir = os.path.join(results_folder, f"{prefix}_PLSR")
        plsr_results_df, _, _ = run_pls_modeling_dataset5(
            results_dict, gen_y, test_size=0.2, random_state=42,
            n_components=n_components, csv_save_dir=csv_dir)
        fig_cal, axs_cal = plot_calibration_dataset5(
            plsr_results_df, calib_save_path=os.path.join(results_folder, f"{prefix}_calibration.pdf"))
        # For the paper figure, assume results_dict_2 exists (or re-use results_dict if applicable)
        raw_results_dict_2 = {}
        for F in [0, 1]:
            raw_results_dict_2[F] = results_dict[F]
        pad_length = 200
        als_preprocessed_dict = preprocess_ALS(raw_results_dict_2, [0, 1], pad_length, asymmetric_least_squares)
        fig_paper, axs_paper = plot_dataset5_paper_figure(
            raw_results_dict_2, als_preprocessed_dict, wavelengths_new,
            fig_save_path=os.path.join(results_folder, f"{prefix}_PaperFigure.pdf"))
    
    print("Processing complete. Check the Results folder for saved outputs.")

if __name__ == '__main__':
    main()

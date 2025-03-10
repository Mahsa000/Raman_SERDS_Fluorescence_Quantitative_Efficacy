"""
This module contains functions to assess model variability and stability
by performing repeated PLSR evaluations using different random states.
It computes metrics such as R², MSE, MAE, and RMSEP over a range of random seeds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression

def evaluate_pls_variability(F_decaying_results_dict, gen_y, initial_F_levels, random_states=None, n_components=2):
    """
    Evaluate PLSR model performance over multiple random states for each FSR level.

    Parameters
    ----------
    F_decaying_results_dict : dict
        Dictionary with keys as fluorescence levels (FSR) and values containing
        spectra for conventional Raman ("specs_1_bg_noise_mean") and SERDS ("serds_bg_noise_mean").
    gen_y : array-like
        The target variable values.
    initial_F_levels : list
        List of FSR levels to evaluate.
    random_states : list, optional
        List of random seed integers for repeated train/test splits. Default:
        [0, 10, 20, 30, 40, 48, 60, 70, 80, 90].
    n_components : int, optional
        Number of components for the PLSR models.

    Returns
    -------
    plsr_results_df : pandas.DataFrame
        DataFrame containing evaluation metrics (R², MSE, MAE, RMSEP) for each FSR and random state.
    """
    if random_states is None:
        random_states = [0, 10, 20, 30, 40, 48, 60, 70, 80, 90]

    # DataFrame to store results
    plsr_results_df = pd.DataFrame(columns=[
        'F level', 'Random State', 
        'Conventional (PLSR) R^2', 'SERDS (PLSR) R^2',
        'Conventional (PLSR) MSE', 'SERDS (PLSR) MSE',
        'Conventional (PLSR) MAE', 'SERDS (PLSR) MAE',
        'Conventional (PLSR) RMSEP', 'SERDS (PLSR) RMSEP'
    ])

    for fsr in initial_F_levels:
        rmsep_values_conv = []  # Store RMSEP values for conventional Raman
        rmsep_values_serds = []  # Store RMSEP values for SERDS

        target_results = F_decaying_results_dict.get(fsr)
        if target_results is None:
            continue

        X_conv = target_results.get("specs_1_bg_noise_mean")
        X_serds = target_results.get("serds_bg_noise_mean")
        if X_conv is None or X_serds is None:
            continue

        for state in random_states:
            X_conv_train, X_conv_test, y_train, y_test = train_test_split(
                X_conv, gen_y, test_size=0.2, random_state=state
            )
            X_serds_train, X_serds_test, _, _ = train_test_split(
                X_serds, gen_y, test_size=0.2, random_state=state
            )

            # Fit PLSR models
            plsr_conv = PLSRegression(n_components=n_components)
            plsr_serds = PLSRegression(n_components=n_components)
            plsr_conv.fit(X_conv_train, y_train)
            plsr_serds.fit(X_serds_train, y_train)

            y_pred_conv = plsr_conv.predict(X_conv_test)
            y_pred_serds = plsr_serds.predict(X_serds_test)

            # Compute metrics
            r2_conv = r2_score(y_test, y_pred_conv)
            r2_serds = r2_score(y_test, y_pred_serds)
            mse_conv = mean_squared_error(y_test, y_pred_conv)
            mse_serds = mean_squared_error(y_test, y_pred_serds)
            mae_conv = mean_absolute_error(y_test, y_pred_conv)
            mae_serds = mean_absolute_error(y_test, y_pred_serds)
            rmsep_conv = np.sqrt(mse_conv)
            rmsep_serds = np.sqrt(mse_serds)

            rmsep_values_conv.append(rmsep_conv)
            rmsep_values_serds.append(rmsep_serds)

            # Append to DataFrame
            plsr_results_df = pd.concat([plsr_results_df, pd.DataFrame({
                'F level': [fsr],
                'Random State': [state],
                'Conventional (PLSR) R^2': [r2_conv],
                'SERDS (PLSR) R^2': [r2_serds],
                'Conventional (PLSR) MSE': [mse_conv],
                'SERDS (PLSR) MSE': [mse_serds],
                'Conventional (PLSR) MAE': [mae_conv],
                'SERDS (PLSR) MAE': [mae_serds],
                'Conventional (PLSR) RMSEP': [rmsep_conv],
                'SERDS (PLSR) RMSEP': [rmsep_serds]
            })], ignore_index=True)

        # Print summary for current FSR level
        print(f"F level {fsr}:")
        print("Mean RMSEP Conv:", np.mean(rmsep_values_conv),
              "Std RMSEP Conv:", np.std(rmsep_values_conv))
        print("Mean RMSEP SERDS:", np.mean(rmsep_values_serds),
              "Std RMSEP SERDS:", np.std(rmsep_values_serds))
        print("-" * 50)

    return plsr_results_df

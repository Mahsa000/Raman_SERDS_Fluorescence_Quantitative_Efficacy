#############################################
# DATASET SELECTION & USER-INTERFACE (CALL PART)
#############################################

def get_dataset_info():
    """
    Returns a dictionary of available datasets with brief descriptions.
    """
    datasets = {
        "1": "Dataset 1: Simulated spectra with FSR values 0, 1, 10, 100, 1000, 2000, 3000, 4000 (decay_rate = 0).",
        "2": "Dataset 2: Same as Dataset 1 but with decay_rate = 0.005.",
        "3": "Dataset 3: Tests prediction accuracy under sample-to-sample fluctuating spectral intensity (random scaling factors ±1–50%) with decay_rate = 0.005.",
        "4": "Dataset 4: Represents granular natural systems with variable fluorescence shapes (100 shapes) from sample to sample with decay_rate = 0.005.",
        "4a": "Dataset 4a: A SERDS-only scenario with independently varying fluorescence shapes for each laser and each sample (11 shapes) with decay_rate = 0.005.",
        "5": "Dataset 5: Combines paired FSR levels (heterogeneous samples) to create training/validation sets, simulated with decay_rate = 0.005."
    }
    return datasets

def get_simulation_data():
    global specs_1_cut, specs_2_cut, pure_fluorescence, wavelengths_new, lambda_1, lambda_2
    return specs_1_cut, specs_2_cut, pure_fluorescence, wavelengths_new, lambda_1, lambda_2
# Raman_SERDS_Fluorescence_Quantitative_Efficacy

This repository contains the code, data, and documentation for our study **On the quantitative fidelity of shifted excitation Raman difference spectroscopy (SERDS) in the presence of dominant fluorescence** which explores the quantitative performance of conventional Raman spectroscopy versus SERDS in the presence of extensive fluorescence interference. Our work leverages large‐scale simulations (a 12‑million spectrum database) of benzophenone/alanine mixtures with varied fluorescence conditions and applies advanced preprocessing (ALS with extrapolation, DWT), multivariate regression (PLSR), and diagnostic plotting.

---

## Table of Contents


- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Modules and Packages](#modules-and-packages)
- [Contributing](#contributing)
- [License](#license)
- [Paper Abstract](#paper-abstract)
- [References](#references)
- [Contact](#contact)

---
## Overview
1. **Dataset Selection:**  
   The script presents six simulated datasets with different fluorescence characteristics:
   - **Dataset 1:** Simulated spectra with decay_rate = 0.
   - **Dataset 2:** Simulated spectra with decay_rate = 0.005.
   - **Dataset 3:** Simulated spectra with fluctuating spectral intensity.
   - **Dataset 4:** Simulated spectra with variable fluorescence shapes.
   - **Dataset 4a:** SERDS-only scenario with independently varying fluorescence shape for each laser and also each sample.
   - **Dataset 5:** Heterogeneous samples combining paired FSR levels.

2. **Data Retrieval/Generation:**  
   - **Existing Data:** If simulation data (spectra, fluorescence, wavelength values, and lambda values) is already saved, the script loads it.
   - **New Simulation:** Otherwise, it generates new simulation data using functions from the `simulation` module.

3. **Preprocessing (Optional):**  
   Users can choose to preprocess the data using one of two methods:
   - **DWT (Discrete Wavelet Transform)**
   - **ALS with advanced extrapolation** (recommended for datasets 4/4a)

4. **Modeling:**  
   The script builds Partial Least Squares Regression (PLSR) models to predict target variables:
   - For most datasets, a single PLSR component is used (Dataset 3 uses two).
   - Dataset 5 is modeled with a dedicated function to handle its heterogeneous nature.

5. **Plotting:**  
   Diagnostic plots are generated to visualize:
   - **Feature Importance:** Highlights significant spectral features.
   - **Calibration Curves:** Compares predicted vs. measured values.
   - **Publication Figures:** Special figures for Dataset 5 are created for inclusion in papers.

6. **Evaluation:**
An evaluation branch computes ensemble performance metrics. By performing model training over multiple randomized splits (e.g., 10 different random states), the script calculates the mean and standard deviation of RMSEP (and other error metrics) for each fluorescence scenario. The evaluation results are saved in a separate “Evaluation” folder under the dataset’s results folder.

7. **Output:**  
   All simulation data, modeling results, and plots are saved in the `Results/` directory (subfolders by dataset).


---

## Repository Structure

- Raman_SERDS_Fluorescence_Quantitative_Efficacy/
  - LICENSE
  - README.md
  - requirements.txt
  - main.py
  - SERDS_Raman_simulation_main.ipynb
  - Abstract/
    - README_Abstract.md
  - data_processing/
    - __init__.py
    - data_loader.py
    - synthetic_generation.py
    - oscillations.py
    - spectral_processing.py
    - laser_experiment.py
  - modeling/
    - __init__.py
    - modeling.py
  - evaluation/
    - __init__.py
    - evaluation.py
  - plotting/
    - __init__.py
    - plotting.py
  - preprocessing/
    - __init__.py
    - pre_processing.py
    - hyperparameter_inspection.py
  - pure_data_library/
    - pure_data_df.csv
  - simulation/
    - simulation_data.py
  - utils/
    - __init__.py
    - file_io.py
    - datasets.py
  - Results/        


---

## Installation and Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/Raman_SERDS_Fluorescence_Quantitative_Efficacy.git
   cd Raman_SERDS_Fluorescence_Quantitative_Efficacy
   
2. **Create and Activate a Virtual Environment**

Using conda:

  ```bash
  conda create --name serds_env python=3.9
  conda activate ramansim_env
  pip install -r requirements.txt
```
3. **(Optional) Launch Jupyter Notebook:**
  ```bash
  jupyter notebook
```
----

## Usage
Running the Main Script:
To run the entire pipeline (simulation, optional preprocessing, modeling, and plotting), simply execute:

```bash
python main.py
```
Follow the on-screen prompts to select the dataset and analysis options.

Jupyter Notebook:
The SERDS_Raman_simulation_main.ipynb notebook demonstrates the complete workflow interactively from base pure spectra. It is ideal for understanding and testing individual steps.

----
## Modules and Packages

* data_processing/
Functions for data loading, synthetic mixture generation, oscillations (APLID), and spectral processing.

* simulation/
Functions to generate simulated Raman and SERDS datasets.

* preprocessing/
Methods for spectral preprocessing (ALS, DWT, hyperparameter tuning).

* modeling/
Functions for PLSR modeling and evaluation.

* plotting/
Functions to generate diagnostic and publication-quality plots (feature importance, calibration curves, etc.).

* utils/
Utility functions for file I/O and dataset management.

---
## Contributing

Contributions are welcome! Please fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

---
## License
This project is licensed under the MIT License. See the LICENSE file for details.

---
## Paper Abstract

While Raman spectroscopy offers notable experimental advantages as a probe of complex mixtures, its application is often hindered by overwhelming fluorescence. This project explores the quantitative efficacy of conventional Raman and SERDS, using extensive stochastic simulations to generate a 12-million spectrum database of benzophenone-alanine mixtures under varied fluorescence conditions. Our findings reveal that while SERDS produces visually enhanced spectra, conventional Raman spectroscopy often achieves comparable or better multivariate regression accuracy, except under extreme fluorescence variability. Advanced preprocessing techniques (ALS and DWT) further improve model performance. This study emphasizes the importance of selecting the appropriate Raman analysis strategy based on specific fluorescence conditions, challenges assumptions about the superiority of visually distinct SERDS spectra, and provides new insights into leveraging Raman spectroscopy in real-world, fluorescence-rich environments.

----
## References
Zarei, M., Solomatova, N. V., Aghaei, H., Rothwell, A., Wiens, J., Melo, L., ... & Grant, E. (2023). Machine learning analysis of Raman spectra to quantify the organic constituents in complex organic–mineral mixtures. Analytical Chemistry, 95(43), 15908-15916.

----
## Contact
mzarei@chem.ubc.ca





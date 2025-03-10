import os
import pandas as pd
import numpy as np

def listdir_nohidden(path):
    """Yield non-hidden filenames in the given directory."""
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def data_loader_by_path(data_dir, column_name='Raw', skiprows=45):
    """
    Loads spectral data from CSV files in the specified directory.
    
    Parameters
    ----------
    data_dir : str
        Directory containing CSV files.
    column_name : str, optional
        Substring to match columns containing spectral data (default 'Raw').
    skiprows : int, optional
        Number of rows to skip when reading CSV files.
        
    Returns
    -------
    specs : np.ndarray
        2D array containing the concatenated spectra.
    waves_cm : np.ndarray
        Array of wavenumbers (converted from wavelengths).
    waves_nm : np.ndarray
        Array of wavelengths.
    id_out : np.ndarray
        Array of identifiers for the loaded files.
    """
    soil_list = list(listdir_nohidden(data_dir))
    id_out = []
    specs = []
    for soil_id in soil_list:
        if soil_id.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_dir, soil_id), skiprows=skiprows, encoding_errors='ignore')
            columns = [c for c in df.columns if column_name in c]
            spec_file = df[columns].values
            if len(specs) == 0:
                specs = spec_file
            else:
                specs = np.concatenate([specs, np.array(spec_file, dtype=float)], axis=1)
            id_out.append([soil_id[:-4]] * len(columns))
            waves_nm = df['Wavelength'].values
            from .spectral_processing import wavelength_to_wavenumber
            waves_cm = wavelength_to_wavenumber(waves_nm, 784.816)
    return np.vstack(specs), waves_cm, waves_nm, np.array(np.hstack(id_out))

def y_loader_by_path(data_dir, index_col='vial #'):
    """
    Loads y-values (e.g., amino acid ratios) from an Excel file.
    
    Parameters
    ----------
    data_dir : str
        Path to the Excel file.
    index_col : str, optional
        Column to use as index (default 'vial #').
    
    Returns
    -------
    df : DataFrame
        The loaded Excel file.
    y : np.ndarray
        Array of y-values.
    y_names : np.ndarray
        Array of column names.
    names : np.ndarray
        Array of index names.
    """
    df = pd.read_excel(data_dir, engine='openpyxl', index_col=index_col).dropna(how='any')
    names = list(df.index)
    y = df.loc[names, :].values
    y_names = list(df.columns.values)
    return df, np.array(y), np.array(y_names), np.array(names)

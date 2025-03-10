import numpy as np

def generate_normalized_y(total_number, y_columns):
    """
    Generate 'total_number' random y-values for 'y_columns' that sum to 1 using a Dirichlet distribution.
    """
    y_new = np.random.dirichlet(np.ones(y_columns), size=total_number)
    name_new = np.array([f"group_{i}" for i in range(total_number)])
    return y_new, name_new

def gen_mix(specs_pure, names_pure, total_number, dim=None, y_columns=None, seed=42):
    """
    Generate synthetic spectra by randomly combining pure spectra.
    
    Parameters
    ----------
    specs_pure : np.ndarray
        Array of pure spectra.
    names_pure : np.ndarray
        Array of labels for pure spectra.
    total_number : int
        Total number of synthetic spectra to generate.
    dim : int
        Dimensionality of the output spectra.
    y_columns : int
        Number of columns for y labels.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    gen_spec : np.ndarray
        Generated synthetic spectra.
    gen_y : np.ndarray
        Corresponding y labels.
    gen_name : np.ndarray
        Generated group names.
    """
    np.random.seed(seed)
    n_groups = len(np.unique(names_pure))
    raveled_spec = np.zeros((n_groups, dim))
    unique_names = np.unique(names_pure)
    for i in range(n_groups):
        spec_subset = specs_pure[names_pure == unique_names[i]]
        if len(spec_subset) == 0:
            idx = np.random.choice(len(specs_pure))
            raveled_spec[i] = specs_pure[idx]
        else:
            idx = np.random.choice(len(spec_subset))
            raveled_spec[i] = spec_subset[idx]
    gen_y, gen_name = generate_normalized_y(total_number, y_columns)
    gen_spec = gen_y @ raveled_spec
    return gen_spec, gen_y, gen_name

def plot_generated_y_histograms(y_values, num_bins=20, figsize=(12,5), exclude_columns=None):
    """
    Plots histograms for each column of y_values.
    
    Parameters
    ----------
    y_values : np.ndarray
        Array of y-values.
    num_bins : int, optional
        Number of histogram bins.
    figsize : tuple, optional
        Figure size.
    exclude_columns : list, optional
        Column indices to exclude.
    """
    import matplotlib.pyplot as plt
    if exclude_columns is None:
        exclude_columns = []
    y_values = np.delete(y_values, exclude_columns, axis=1)
    num_columns = y_values.shape[1]
    num_subplots = num_columns
    num_rows = (num_subplots + 1) // 2
    num_cols = (num_subplots + num_rows - 1) // num_rows
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    for column_index in range(num_columns):
        axs[column_index].hist(y_values[:, column_index], bins=num_bins, density=True, alpha=0.7)
        axs[column_index].set_title(f'Distribution for column {column_index}')
        axs[column_index].set_xlabel('Value')
        axs[column_index].set_ylabel('Density')
    for i in range(num_columns, len(axs)):
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

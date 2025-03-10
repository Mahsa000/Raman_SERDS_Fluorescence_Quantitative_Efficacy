import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d, CubicSpline
import pywt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def wavenumber_to_wavelength(wavenumbers, excitation_wavelength):
    """
    Convert wavenumbers to wavelengths.
    """
    return 1 / ((1 / excitation_wavelength) - wavenumbers * 1E-7)

def wavelength_to_wavenumber(wavelengths, excitation_wavelength):
    """
    Convert wavelengths to wavenumbers.
    """
    return ((1 / excitation_wavelength) - (1 / wavelengths)) * 1E7

def shifted_raman_spectra_generator(wavenumbers, gen_spec, lambda_1, lambda_2, start=34, end=1024, plot=True, sample=0):
    """
    Generates shifted Raman spectra by interpolating the spectra from one wavelength scale onto another.
    
    Parameters
    ----------
    wavenumbers : array-like
        Array of wavenumbers.
    gen_spec : 2D numpy array
        Generated spectra.
    lambda_1 : float
        Excitation wavelength for laser 1.
    lambda_2 : float
        Excitation wavelength for laser 2.
    start : int, optional
        Index to start cutting the spectrum.
    end : int, optional
        End index.
    plot : bool, optional
        Whether to produce plots.
    sample : int, optional
        Sample index to plot.
    
    Returns
    -------
    wavelengths_new : np.ndarray
        The new wavelength array after cutting.
    specs_1_cut : np.ndarray
        The shifted and cut spectrum for laser 1.
    specs_2_cut : np.ndarray
        The shifted and cut spectrum for laser 2.
    (and figures if plot=True)
    """
    def wavenumber_to_wavelength_local(wavenumbers, excitation_wavelength):
        return 1 / ((1 / excitation_wavelength) - wavenumbers * 1E-7)
    
    wavelengths_1 = wavenumber_to_wavelength_local(wavenumbers, lambda_1)
    wavelengths_2 = wavenumber_to_wavelength_local(wavenumbers, lambda_2)
    
    if plot:
        fig1 = plt.figure(figsize=(8,6))
        plt.plot(wavelengths_1, gen_spec[sample], 'b', label=f'$\\lambda1$={lambda_1}')
        plt.plot(wavelengths_2, gen_spec[sample], 'r', label=f'$\\lambda2$={lambda_2}')
        plt.title(f'Actual & Shifted Raman Spectra | sample {sample}', fontsize=16)
        plt.xlabel('Wavelength (nm)', fontsize=16)
        plt.ylabel('Raman Intensity (a.u.)', fontsize=16)
        plt.legend()
    
    specs_1 = np.zeros(gen_spec.shape)
    specs_2 = np.zeros(gen_spec.shape)
    specs_1_cut = np.zeros((gen_spec.shape[0], gen_spec.shape[1]-start))
    specs_2_cut = np.zeros((gen_spec.shape[0], gen_spec.shape[1]-start))
    
    for spectrum in range(gen_spec.shape[0]):
        f_2 = CubicSpline(wavelengths_2, gen_spec[spectrum, :], bc_type='natural')
        specs_1[spectrum, :] = gen_spec[spectrum, :]
        specs_2[spectrum, :] = f_2(wavelengths_1)
        wavelengths_new = wavelengths_1[start:]
        specs_1_cut[spectrum, :] = specs_1[spectrum, :][start:]
        specs_2_cut[spectrum, :] = specs_2[spectrum, :][start:]
    
    if plot:
        fig2 = plt.figure(figsize=(8,6))
        plt.plot(wavelengths_new, specs_1_cut[sample], 'b', label=f'$\\lambda1$={lambda_1}')
        plt.plot(wavelengths_new, specs_2_cut[sample], 'r', label=f'$\\lambda2$={lambda_2}')
        plt.title('Actual & Shifted Raman Spectra | Fixed x-axis', fontsize=16)
        plt.xlabel('Wavelength (nm)', fontsize=16)
        plt.ylabel('Raman Intensity (a.u.)', fontsize=16)
        plt.legend()
    
        fig3 = plt.figure(figsize=(8,6))
        plt.plot(wavelengths_new, specs_1_cut[sample] - specs_2_cut[sample], 'black', label=f'$\\lambda1$ - $\\lambda2$')
        plt.title('SERDS | Fixed x-axis', fontsize=16)
        plt.xlabel('Wavelength (nm)', fontsize=16)
        plt.ylabel('Raman Intensity (a.u.)', fontsize=16)
        plt.legend()
        return wavelengths_new, specs_1_cut, specs_2_cut, fig1, fig2, fig3
    else:
        return wavelengths_new, specs_1_cut, specs_2_cut

def dwt_multilevel_filter(spectrum, wavelet, scale, apprx_rm, low_cut, high_cut):
    """
    Applies a multilevel DWT filter by zeroing selected coefficients.
    """
    coeffs = pywt.wavedec(spectrum, wavelet, level=scale)
    for c_ix in range(len(coeffs)):
        if (c_ix == 0 and apprx_rm) or (c_ix > 0 and c_ix <= low_cut) or c_ix > scale - high_cut:
            coeffs[c_ix] = np.zeros_like(coeffs[c_ix])
    return pywt.waverec(coeffs, wavelet)

def dwt_iterative_bg_rm(spectrum, wavelet, scale, iterations):
    """
    Iteratively applies DWT background removal.
    """
    bg_approx = spectrum.copy()
    for ix in range(iterations):
        apprx_ix = dwt_multilevel_filter(bg_approx, wavelet, scale, apprx_rm=False, low_cut=0, high_cut=scale)
        if bg_approx.shape[0] % 2 == 0:
            bg_approx = np.minimum(bg_approx, apprx_ix)
        else:
            bg_approx = np.minimum(bg_approx, apprx_ix[:-1])
    spectrum_bg_removed = spectrum - bg_approx
    return spectrum_bg_removed, bg_approx

def norm_mean(x):
    """Normalize by subtracting mean and dividing by standard deviation."""
    return (x - np.mean(x)) / np.std(x)

def norm_min(x):
    """Normalize using min-max scaling."""
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def asymmetric_least_squares(y, lam, p, niter=10):
    """
    Performs asymmetric least squares background removal.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    L = len(y)
    D = sparse.spdiags([-1*np.ones(L), 2*np.ones(L), -1*np.ones(L)], [-1, 0, 1], L, L).tocsc()
    D = lam * (D.T @ D)
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def calculate_SNR(specs_bg, pivot1=800, pivot2=1010, norm=True):
    """
    Calculates signal-to-noise ratio using ALS-removed spectrum.
    """
    if not norm:
        specs_norm = np.vstack([(s - np.mean(s)) / np.std(s) for s in specs_bg])
    else:
        specs_norm = specs_bg
    SNR = []
    for noisy_spec in specs_norm:
        als_fit = asymmetric_least_squares(noisy_spec, 10**5, 0.5, niter=10)
        als_spec_flat = noisy_spec - als_fit
        SNR.append(np.divide(np.max(als_spec_flat), np.std(als_spec_flat[pivot1:pivot2])))
    return np.mean(SNR)

def plot_ALS_snr_measurement(pure_fluorescent, lam=10**5, p=0.5, niter=10):
    """
    Plots ALS background removal and displays the SNR measurement.
    """
    pivot1 = 800
    pivot2 = 1010
    a = pure_fluorescent
    x = np.arange(len(a))
    pad_length = 200
    padded_a = np.pad(a, pad_width=pad_length, mode='edge')
    als_fit = asymmetric_least_squares(padded_a, lam, p, niter)
    als_spec_flat = als_fit[pad_length:-pad_length]
    als_removed_spec = a - als_spec_flat
    snr = calculate_SNR([als_removed_spec], pivot1=pivot1, pivot2=pivot2, norm=False)
    fig, ax = plt.subplots()
    ax.plot(x, a, label='Raw Spectrum', color='red')
    ax.plot(x, als_spec_flat, label='Fitted ALS Curve', color='orange')
    ax.plot(x, als_removed_spec, label='Residual after ALS Removal', color='purple')
    ax.plot(x[pivot1:pivot2], als_removed_spec[pivot1:pivot2], color='black', label='Noise Region')
    ax.legend()
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Intensity')
    ax.set_title(f'ALS bg removal\nSNR={snr:.1f}')
    plt.show()

def generate_fluorescence_shapes(shift_points, original_fluorescence):
    """
    Generates shifted fluorescence shapes by interpolating the original fluorescence.
    
    Parameters
    ----------
    shift_points : array-like
        Shift values used to generate new shapes.
    original_fluorescence : array-like
        The original fluorescence signal (1D).
    
    Returns
    -------
    fluorescence_shapes : list of np.ndarray
        List of shifted fluorescence shapes.
    """
    from scipy.integrate import trapz
    fluorescence_shapes = []
    for shift_point in shift_points:
        area_before_shift = trapz(original_fluorescence[original_fluorescence < shift_point])
        peak_position = np.argmax(original_fluorescence)
        shift_offset = shift_point - peak_position
        shifted_x = np.arange(len(original_fluorescence)) - shift_offset
        interpolator = interp1d(shifted_x, original_fluorescence, fill_value="extrapolate")
        shifted_fluorescence = interpolator(np.arange(len(original_fluorescence)))
        fluorescence_shapes.append(shifted_fluorescence)
    return fluorescence_shapes

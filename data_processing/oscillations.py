import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display
from PIL import Image

def sine_wave(t, phi, freq, offset=0):
    """
    Returns a sinusoidal wave value for given time, phase, frequency, and offset.
    """
    return np.sin(2 * np.pi * freq * t + phi) + offset

def oscillation(spec_1, spec_2, phi1, phi2, freq, t):
    """
    Oscillates two spectra with a pi/2 phase difference.
    
    Returns modulated spectra and their sum.
    """
    osci_spec_1 = np.zeros((len(spec_1), len(t)))
    osci_spec_2 = np.zeros((len(spec_1), len(t)))
    for i in range(len(t)):
        osci_spec_1[:, i] = spec_1 * sine_wave(i, phi1, freq)**2
        osci_spec_2[:, i] = spec_2 * sine_wave(i, phi2, freq)**2
    osci_spec_added = osci_spec_1 + osci_spec_2
    return osci_spec_1, osci_spec_2, osci_spec_added

def oscillation_2theta(spec_1, spec_2, phi, freq, t):
    """
    Oscillation function for 2θ demodulation.
    """
    osci_spec_added = np.zeros((len(spec_1), len(t)))
    for i in range(len(t)):
        osci_spec_added[:, i] = ((spec_1 - spec_2) / 2) * sine_wave(i, phi, 2 * freq) + ((spec_1 + spec_2) / 2)
    return osci_spec_added

def find_phases(phi1, phi2, resolution):
    """
    Generates linearly spaced phase values between phi1 and phi2.
    """
    return np.linspace(phi1, phi2, resolution)

def plot_oscillogram(oscillograms2, wavelengths_new=None, sample_index=0, title='Oscillogram'):
    """
    Creates a 3D surface and contour plot of the oscillogram data.
    """
    tt = np.arange(oscillograms2.shape[2])
    x, y = np.meshgrid(tt, wavelengths_new)
    z = oscillograms2[sample_index, :, :]
    my_cmap = plt.get_cmap('coolwarm_r')
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(x, y, z, cmap=my_cmap, linewidth=0, antialiased=False)
    ax1.set_title(title + ' - Parametric view', fontsize=16)
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Wavelength [nm]', fontsize=14)
    ax1.set_zlabel('Intensity (a.u.)', fontsize=14)
    ax1.view_init(45, 45)
    
    ax2 = fig.add_subplot(122, projection='3d')
    cset = ax2.contourf(x, y, z, zdir='z', offset=np.min(z), cmap=my_cmap)
    ax2.set_title(title + ' - Projected view', fontsize=16)
    ax2.set_xlabel('Time', fontsize=14)
    ax2.set_ylabel('Wavelength [nm]', fontsize=14)
    ax2.set_zlabel('Intensity (a.u.)', fontsize=14)
    ax2.view_init(45, 45)
    plt.tight_layout()
    plt.show()
    return fig, ax1, ax2

def plot_oscillogram_prametric(oscillograms2, wavelengths_new=None, sample_index=0, title='Oscillogram'):
    """
    Creates and saves a 3D parametric oscillogram plot, then displays it.
    """
    tt = np.arange(oscillograms2.shape[2])
    x, y = np.meshgrid(tt, wavelengths_new)
    z = oscillograms2[sample_index, :, :]
    my_cmap = plt.get_cmap('coolwarm_r')
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111, projection='3d')
    surf1 = ax1.plot_surface(x, y, z, cmap=my_cmap, linewidth=0, antialiased=False)
    ax1.set_title(title + '\n Parametric View', fontsize=16)
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Wavelength (nm)', fontsize=14)
    ax1.set_zlabel('Intensity', fontsize=14)
    ax1.view_init(45, 45)
    plt.savefig('Figures/oscillogram.png', dpi=400)
    plt.close()
    from PIL import Image
    img = Image.open('Figures/oscillogram.png')
    display(img)


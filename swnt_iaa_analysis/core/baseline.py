"""Baseline correction functions."""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter1d


def lieberfit(spectrum, order=5, tot_iter=100):
    """
    Perform iterative polynomial baseline correction (lieberfit).
    
    Parameters:
    -----------
    spectrum : array-like
        Input spectrum
    order : int
        Polynomial order for baseline fitting
    tot_iter : int
        Number of iterations
    
    Returns:
    --------
    corrected_spectrum : array
        Baseline-corrected spectrum
    baseline : array
        Fitted baseline
    """
    pix_size = len(spectrum)
    polyspec_iter = np.array(spectrum)
    x = np.arange(pix_size)
    
    for _ in range(tot_iter):
        p_order = np.polyfit(x, polyspec_iter, order)
        polyspec_order = np.polyval(p_order, x)
        polyspec_iter = np.minimum(polyspec_order, polyspec_iter)
    
    baseline = polyspec_iter
    corrected_spectrum = spectrum - baseline
    return corrected_spectrum, baseline


def apply_als_baseline(y, lam=1e7, p=0.001, niter=20):
    """
    Apply Asymmetric Least Squares (ALS) baseline correction.
    
    Parameters:
    -----------
    y : array-like
        Input signal
    lam : float
        Lambda parameter (smoothness penalty)
    p : float
        Asymmetry parameter (0 < p < 1)
    niter : int
        Number of iterations
    
    Returns:
    --------
    baseline : array
        Fitted baseline
    """
    y = np.asarray(y, dtype=float)
    L = len(y)
    D = diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2), dtype=float, format='csc')
    w = np.ones(L, dtype=float)
    for i in range(niter):
        W = diags(w, 0, shape=(L, L), format='csc')
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def apply_gaussian_smoothing(signal, sigma=25):
    """
    Apply Gaussian smoothing to signal.
    
    Parameters:
    -----------
    signal : array-like
        Input signal
    sigma : float
        Standard deviation for Gaussian smoothing
    
    Returns:
    --------
    smoothed : array
        Smoothed signal
    """
    return gaussian_filter1d(signal, sigma=sigma)


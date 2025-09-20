import numpy as np
from scipy.ndimage import gaussian_filter

def vesselness2D(I, sigmas, spacing=(1, 1), tau=1.0, brightondark=False):
    """
    Vesselness filter for 2D images (Jerman et al. 2014).
    
    Parameters
    ----------
    I : ndarray
        2D input image.
    sigmas : list or range
        Scales at which to compute vesselness.
    spacing : tuple of floats
        Pixel spacing (dy, dx).
    tau : float
        Between 0.5 and 1, controls response uniformity.
    brightondark : bool
        If True, vessels are bright on dark background.
    
    Returns
    -------
    vesselness : ndarray
        Vesselness probability map.
    """
    I = I.astype(np.float32)
    vesselness = np.zeros_like(I, dtype=np.float32)

    for sigma in sigmas:
        # Compute Hessian and eigenvalues
        Lambda1, Lambda2 = imageEigenvalues(I, sigma, spacing, brightondark)
        if brightondark:
            Lambda2 = -Lambda2

        # Proposed filter response
        Lambda3 = Lambda2.copy()
        Lambda_rho = Lambda3.copy()
        max_val = np.max(Lambda3)
        if max_val > 0:
            mask = (Lambda3 > 0) & (Lambda3 <= tau * max_val)
            Lambda_rho[mask] = tau * max_val
        Lambda_rho[Lambda3 <= 0] = 0

        denom = (Lambda2 + Lambda_rho) ** 3
        with np.errstate(divide='ignore', invalid='ignore'):
            response = (Lambda2 ** 2) * (Lambda_rho - Lambda2) * 27.0 / denom

        response[(Lambda2 >= Lambda_rho / 2) & (Lambda_rho > 0)] = 1
        response[(Lambda2 <= 0) | (Lambda_rho <= 0)] = 0
        response[~np.isfinite(response)] = 0

        vesselness = np.maximum(vesselness, response)

    if np.max(vesselness) > 0:
        vesselness /= np.max(vesselness)
    vesselness[vesselness < 1e-2] = 0
    return vesselness


def imageEigenvalues(I, sigma, spacing=(1, 1), brightondark=False):
    """Compute 2D Hessian eigenvalues."""
    Hxx, Hyy, Hxy = Hessian2D(I, sigma, spacing)
    c = sigma ** 2
    Hxx *= c
    Hxy *= c
    Hyy *= c

    B1 = -(Hxx + Hyy)
    B2 = Hxx * Hyy - Hxy ** 2
    T = np.ones_like(B1, dtype=bool)

    if brightondark:
        T[B1 < 0] = 0
        T[(B2 == 0) & (B1 == 0)] = 0
    else:
        T[B1 > 0] = 0
        T[(B2 == 0) & (B1 == 0)] = 0

    Hxx_sel = Hxx[T]
    Hyy_sel = Hyy[T]
    Hxy_sel = Hxy[T]

    L1, L2 = eigvalOfHessian2D(Hxx_sel, Hxy_sel, Hyy_sel)

    Lambda1 = np.zeros_like(I, dtype=np.float32)
    Lambda2 = np.zeros_like(I, dtype=np.float32)
    Lambda1[T] = L1
    Lambda2[T] = L2

    Lambda1[~np.isfinite(Lambda1)] = 0
    Lambda2[~np.isfinite(Lambda2)] = 0
    Lambda1[np.abs(Lambda1) < 1e-4] = 0
    Lambda2[np.abs(Lambda2) < 1e-4] = 0

    return Lambda1, Lambda2


def Hessian2D(I, sigma, spacing=(1, 1)):
    """Compute 2D Hessian (second-order derivatives) with Gaussian smoothing."""
    if sigma > 0:
        F = imgaussian(I, sigma, spacing)
    else:
        F = I

    Dy = gradient2(F, axis=0)
    Dyy = gradient2(Dy, axis=0)

    Dx = gradient2(F, axis=1)
    Dxx = gradient2(Dx, axis=1)
    Dxy = gradient2(Dx, axis=0)

    return Dxx, Dyy, Dxy


def gradient2(F, axis):
    """Compute gradient along given axis (0 = y, 1 = x)."""
    D = np.zeros_like(F, dtype=np.float32)
    if axis == 0:  # y-axis
        D[0, :] = F[1, :] - F[0, :]
        D[-1, :] = F[-1, :] - F[-2, :]
        D[1:-1, :] = (F[2:, :] - F[:-2, :]) / 2
    elif axis == 1:  # x-axis
        D[:, 0] = F[:, 1] - F[:, 0]
        D[:, -1] = F[:, -1] - F[:, -2]
        D[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / 2
    return D


def imgaussian(I, sigma, spacing=(1, 1), siz=None):
    """Apply separable Gaussian filter."""
    if siz is None:
        siz = int(sigma * 6)

    # Gaussian smoothing with spacing correction
    sy, sx = spacing
    out = gaussian_filter(I, sigma=[sigma / sy, sigma / sx], mode='nearest')
    return out


def eigvalOfHessian2D(Dxx, Dxy, Dyy):
    """Compute eigenvalues of Hessian matrix, sorted by abs value."""
    tmp = np.sqrt((Dxx - Dyy) ** 2 + 4 * Dxy ** 2)
    mu1 = 0.5 * (Dxx + Dyy + tmp)
    mu2 = 0.5 * (Dxx + Dyy - tmp)

    check = np.abs(mu1) > np.abs(mu2)
    Lambda1 = mu1.copy()
    Lambda2 = mu2.copy()
    Lambda1[check] = mu2[check]
    Lambda2[check] = mu1[check]
    return Lambda1, Lambda2

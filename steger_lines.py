import numpy as np
from scipy.ndimage import convolve

def steger_lines(I, sigma, fast=1, denom=10):
    """
    Line detection using Gaussian derivatives and eigenvalue analysis.
    Args:
        I: 2D numpy array (image)
        sigma: Gaussian sigma
        fast: if True, use convolution; else, use imfilter-like (not implemented)
        denom: denominator for thresholding
    Returns:
        BW: binary line map
        lmax, lmin: max/min eigenvalues
        Hdet, Htrace: Hessian determinant and trace
        ux, uy: eigenvector components
    """
    sigma2 = sigma * sigma
    wsize = int(np.ceil(4 * sigma))
    x, y = np.meshgrid(np.arange(-wsize, wsize+1), np.arange(-wsize, wsize+1))
    G = np.exp(-(x**2 + y**2) / (2 * sigma2)) / (2 * np.pi * sigma2)
    Gx = -(x / sigma2) * G * sigma2
    Gy = -(y / sigma2) * G * sigma2
    Gxx = sigma2 * (x**2 - sigma2) * G / (sigma2 * sigma2)
    Gxy = sigma2 * (x * y) * G / (sigma2 * sigma2)
    Gyy = sigma2 * (y**2 - sigma2) * G / (sigma2 * sigma2)

    I = I.astype(float)
    Ixx = convolve(I, Gxx, mode='reflect')
    Ixy = convolve(I, Gxy, mode='reflect')
    Iyy = convolve(I, Gyy, mode='reflect')
    Ix  = convolve(I, Gx,  mode='reflect')
    Iy  = convolve(I, Gy,  mode='reflect')

    Hdet = Ixx * Iyy - Ixy ** 2
    Htrace = Ixx + Iyy

    riza = np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy ** 2)
    l1 = (Ixx + Iyy + riza) / 2
    l2 = (Ixx + Iyy - riza) / 2

    lmax = np.where(np.abs(l1) > np.abs(l2), l1, l2)
    lmin = np.where(np.abs(l1) < np.abs(l2), l1, l2)

    temp_mat = (lmax > np.max(lmax) / denom).astype(float)
    ux = 1.0 / np.sqrt(1.0 + ((lmax * temp_mat - Ixx * temp_mat) ** 2) / ((Ixy * temp_mat) ** 2 + 1e-12))
    uy = ((lmax * temp_mat - Ixx * temp_mat) * ux) / (Ixy * temp_mat + 1e-12)

    t = -(ux * Ix * temp_mat + uy * Iy * temp_mat) / (
        ux * ux * Ixx * temp_mat + uy * uy * Iyy * temp_mat + 2 * ux * uy * Ixy * temp_mat + 1e-12)
    p = ((np.abs(t * ux) + np.abs(t * uy) < 1) & (np.abs(t * ux) + np.abs(t * uy) > 0)).astype(float)
    BW = np.zeros_like(lmax)
    BW += 255 * p * temp_mat

    return BW, lmax, lmin, Hdet, Htrace, ux, uy

import numpy as np
from ct_detect import ct_detect
import scipy
from scipy import interpolate


def ct_calibrate(photons, material, sinogram, scale):

    """ct_calibrate convert CT detections to linearised attenuation
    sinogram = ct_calibrate(photons, material, sinogram, scale) takes the CT detection sinogram
    in x (angles x samples) and returns a linear attenuation sinogram
    (angles x samples). photons is the source energy distribution, material is the
    material structure containing names, linear attenuation coefficients and
    energies in mev, and scale is the size of each pixel in x, in cm."""

    # Get dimensions and work out detection for just air of twice the side
    # length (has to be the same as in ct_scan.py)
    n = sinogram.shape[1]

    air_energy = ct_detect(photons, material.coeff('Air'), 2*n*scale)[0]

    # perform calibration

    for i in range(len(sinogram)):
        for j in range(n):
            sinogram[i][j] = -np.log(sinogram[i][j] / air_energy)

    return sinogram

import numpy as np
from ct_detect import ct_detect
import scipy
from scipy import interpolate


def ct_calibrate(photons, material, sinogram, scale, number_detectors=None, noise=True):

    """ct_calibrate convert CT detections to linearised attenuation
    sinogram = ct_calibrate(photons, material, sinogram, scale) takes the CT detection sinogram
    in x (angles x samples) and returns a linear attenuation sinogram
    (angles x samples). photons is the source energy distribution, material is the
    material structure containing names, linear attenuation coefficients and
    energies in mev, and scale is the size of each pixel in x, in cm."""

    # Get dimensions and work out detection for just air of twice the side length (has to be the same as in ct_scan.py)
    if number_detectors is None:
        number_detectors = sinogram.shape[1]

    air_residual = ct_detect(
        photons, material.coeff("Air"), 2 * number_detectors * scale, noise=noise
    )  # taking the residual intensity through air at a single depth of double the side length

    # performing calibration relative to residual energy through just air
    sinogram = np.log(air_residual / sinogram)

    # performing calibration for beam hardening

    # have an array of thicknesses to scan water at 256 or number of detectors, whichever is larger
    t_w = scale * np.arange(np.max([number_detectors, 256]))
    p_w = np.log(
        air_residual / ct_detect(photons, material.coeff("Water"), t_w, noise=noise)
    )

    # fit as described in the handout

    coeffs = np.polyfit(p_w, t_w, 4)
    t_w_m = np.polyval(coeffs, sinogram)

    C = 0.2
    sinogram = t_w_m * C

    return sinogram

def ct_calibrate_real(sinogram, air_residual):

    """ct_calibrate convert CT detections to linearised attenuation
    sinogram = ct_calibrate(photons, material, sinogram, scale) takes the CT detection sinogram
    in x (angles x samples) and returns a linear attenuation sinogram
    (angles x samples). photons is the source energy distribution, material is the
    material structure containing names, linear attenuation coefficients and
    energies in mev, and scale is the size of each pixel in x, in cm."""

    # # Get dimensions and work out detection for just air of twice the side length (has to be the same as in ct_scan.py)
    # if number_detectors is None:
    #     number_detectors = sinogram.shape[1]

    # performing calibration relative to residual energy through just air
    sinogram = np.log(air_residual / sinogram)

    # # performing calibration for beam hardening
    #
    # # have an array of thicknesses to scan water at 256 or number of detectors, whichever is larger
    # t_w = scale * np.arange(np.max([number_detectors, 256]))
    # p_w = np.log(
    #     air_residual / ct_detect(photons, material.coeff("Water"), t_w, noise=noise)
    # )
    #
    # # fit as described in the handout
    #
    # coeffs = np.polyfit(p_w, t_w, 4)
    # t_w_m = np.polyval(coeffs, sinogram)
    #
    # C = 0.2
    # sinogram = t_w_m * C

    return sinogram
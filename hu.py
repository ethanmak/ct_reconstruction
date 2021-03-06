
from ct_calibrate import *
from ct_our_functions import *

# in order to turn to hounsfield units, we perform a linear regression by comparing with reconstructed data
# _points = ((0, -1000), (0.6144575495418269, 1450))
#
# _m = (_points[1][1] - _points[0][1]) / (_points[1][0] - _points[0][0])
# _b = _points[0][1] - _m * _points[0][0]

_mu_w = log_interp(np.array([0.04, 0.05]), np.array([0.2683, 0.2269]), 0.75 * 0.06)

def hu(photons, material, reconstruction, scale, n_detectors, noise=True):
	"""convert CT reconstruction output to Hounsfield Units
	calibrated = hu(photons, material, reconstruction, scale, n_detectors) converts the reconstruction into Hounsfield
	Units, using the material coefficients, photon energy photons and scale given."""
	
	# n_detectors is taken as an arguments to calculate the depth through which to calibrate the normalised
	# water and air attenuation coefficients. For a ct_detect function that does not compensate for beam hardening,
	# this value does not affect the result.
	
	# use water to calibrate, using the same calibration process as the normal CT data
	size = 2.0 * n_detectors * scale  # size only matters if ct_detect attempts to compensate for beam hardening
	# taking the residual intensity through water at a single depth of double the side length
	water_residual = ct_detect(photons, material.coeff('Water'), size, noise=noise)
	# performing calibration relative to residual energy through just air
	mu_water = ct_calibrate(photons, material, water_residual, scale, n_detectors, noise=noise) / size
	# according to the above calibration, mu_air is 0
	
	# using result to convert to hounsfield units
	reconstruction = 1000.0 * (reconstruction - mu_water) / mu_water
	
	# clamping between -1024 and 3072
	return np.clip(reconstruction, -1024.0, 3072.0)


def hu_real(reconstruction, scale=None):
	"""convert CT reconstruction output to Hounsfield Units
	calibrated = hu(photons, material, reconstruction, scale, n_detectors) converts the reconstruction into Hounsfield
	Units, using the material coefficients, photon energy photons and scale given."""

	# using result to convert to hounsfield units
	# reconstruction = _m * reconstruction + _b

	# gotten from https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html and assuming a logarithmic linear
	# relationship between energy and attenuation coefficient
	# Average kev gotten from https://www.researchgate.net/figure/Average-kiloelectron-voltage-keV-given-the-kilovoltage-peak-kVp-7_tbl1_262228564

	reconstruction = 1000.0 * (reconstruction - _mu_w) / _mu_w

	# clamping between -1024 and 3072
	return np.clip(reconstruction, -1024.0, 3072.0)
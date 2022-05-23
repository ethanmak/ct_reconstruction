import numpy as np
from attenuate import *
from ct_calibrate import *

def hu(photons, material, reconstruction, n_detectors, scale):
	"""convert CT reconstruction output to Hounsfield Units
	calibrated = hu(photons, material, reconstruction, n_detectors, scale) converts the reconstruction into Hounsfield
	Units, using the material coefficients, photon energy photons and scale given."""
	
	# n_detectors is taken as an arguments to calculate the depth through which to calibrate the normalised water and air attenuation coefficients. For a ct_detect function that does not compensate for beam hardening, this value does not affect the result.
	
	# use water to calibrate, using the same calibration process as the normal CT data
	size = 2.0*n_detectors*scale # size only matters if ct_detect attempts to compensate for beam hardening
	water_residual = ct_detect(photons, material.coeff('Water'), size) # taking the residual intensity through water at a single depth of double the side length
	air_residual = ct_detect(photons, material.coeff('Air'), size) # taking the residual intensity through air at a single depth of double the side length
	mu_water = np.log(air_residual / water_residual) / size # performing calibration relative to residual energy through just air
	# according to the above calibration, mu_air is 0
	
	# using result to convert to hounsfield units
	reconstruction = 1000.0*(reconstruction - mu_water)/mu_water
	
	# clamping between -1024 and 3072
	return np.clip(reconstruction, -1024.0, 3072.0)

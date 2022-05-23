import numpy as np
from attenuate import *
from ct_calibrate import *


def hu(p, material, reconstruction, scale):
    """convert CT reconstruction output to Hounsfield Units
    calibrated = hu(p, material, reconstruction, scale) converts the reconstruction into Hounsfield
    Units, using the material coefficients, photon energy p and scale given."""

    # use water to calibrate

    # put this through the same calibration process as the normal CT data

    # use result to convert to hounsfield units
    # limit minimum to -1024, which is normal for CT data.
    
	# get average photon energy of source:
	energy_sum = 0.0
	for (i, v) in enumerate(photons):
		energy_sum += (i + 1.0)*v
	energy_index = int(np.round(energy_sum/sum(photons))) - 1
	# get corresponding attenuation coefficients of water and air:
	# the coeffs array gives attenuation coefficients in intervals of 0.001 MeV or 1.0 keV, starting from 0.001 [MeV]. So, the value at index 'keVp - 1' is the the attenuation coefficient of the material for photons of energy $keVp [keVp].
	mu_water = material.coeff("Water")[energy_index]
	mu_air = material.coeff("Air")[energy_index]
	# apply normalisation:
	back_proj = 1000.0*(back_proj - mu_water)/(mu_water - mu_air)

    return reconstruction

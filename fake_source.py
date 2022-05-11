import numpy as np
import math

def fake_source(mev, mvp, coeff=None, thickness=0, method='normal'):

	""" fake_source can generate typical CT X-ray source energies

	fake_source((mev, mvp, coeffs, thickness) creates a vector y of photons per
	cm^2 per keV, at the energies given by the vector mev and for a source
	with the given mvp maximum energy in MeV.

	The source will be filtered by a thickness (in mm) of the material
	whose mass attenuation coefficients are given by coeff, which are
	presumed to be at the same set of energies as given in mev.

	fake_source(mev, mvp, coeffs, thickness, 'ideal') creates output energies
	for an 'ideal' source with a very narrow energy range."""

	# check for energies
	energies = len(mev)

	if method == 'ideal':
	
		# single energy, at about the peak of the broader energy radiation
		source = np.zeros(energies)
		m = np.abs(mev - mvp * 0.7)
		source[np.where(m == np.amin(m))] = 1e10
	
	else:

		# experimental function to match expected form of source radiation
		alpha = 100
		sigma = mvp / 2
		offset = -sigma

		source = -(pow((mev - offset), 2)) / ( 2 * pow(sigma, 2))

		source = (1 / pow((2 * math.pi), 2)) * np.exp(source) * pow(np.abs(mev - offset), (1 / alpha))

		source[np.where(mev>mvp)] = 0

		for index, value in np.ndenumerate(mev):
			if source[index] != 0 and value > (0.8 * mvp):
				source[index] = source[index] * pow((mvp - value) / (0.2 * mvp), .3)

		source = source * 1.5e9

	# add any additional metal filter
	if coeff is not None:
		source = source * np.exp(-coeff * thickness / 10)

	return source
import numpy as np
from attenuate import attenuate

def ct_detect(p, coeffs, depth, mas=10000):

	"""ct_detect returns detector photons for given material depths.
	y = ct_detect(p, coeffs, depth, mas) takes a source energy
	distribution photons (energies), a set of material linear attenuation
	coefficients coeffs (materials, energies), and a set of material depths
	in depth (materials, samples) and returns the detections at each sample
	in y (samples).

	mas defines the current-time-product which affects the noise distribution
	for the linear attenuation"""

	# check p for number of energies
	if type(p) != np.ndarray:
		p = np.array([p])
	if p.ndim > 1:
		raise ValueError('input p has more than one dimension')
	energies = len(p)

	# check coeffs is of (materials, energies)
	if type(coeffs) != np.ndarray:
		coeffs = np.array([coeffs]).reshape((1, 1))
	elif coeffs.ndim == 1:
		coeffs = coeffs.reshape((1, len(coeffs)))
	elif coeffs.ndim != 2:
		raise ValueError('input coeffs has more than two dimensions')
	if coeffs.shape[1] != energies:
		raise ValueError('input coeffs has different number of energies to input p')
	materials = coeffs.shape[0]

	# check depth is of (materials, samples)
	if type(depth) != np.ndarray:
		depth = np.array([depth]).reshape((1,1))
	elif depth.ndim == 1:
		if materials == 1:
			depth = depth.reshape(1, len(depth))
		else:
			depth = depth.reshape(len(depth), 1)
	elif depth.ndim != 2:
		raise ValueError('input depth has more than two dimensions')
	if depth.shape[0] != materials:
		raise ValueError('input depth has different number of materials to input coeffs')
	samples = depth.shape[1]

	# extend source photon array so it covers all samples
	detector_photons = np.zeros([energies, samples])
	for e in range(energies):
		detector_photons[e] = p[e]

	# calculate array of residual mev x samples for each material in turn
	for m in range(materials):
		detector_photons = attenuate(detector_photons, coeffs[m], depth[m])

	# sum this over energies
	detector_photons = np.sum(detector_photons, axis=0)

	# model noise

	# minimum detection is one photon
	detector_photons = np.clip(detector_photons, 1, None)

	return detector_photons
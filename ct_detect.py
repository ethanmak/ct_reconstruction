import numpy as np
from attenuate import attenuate

# From https://www.gov.uk/government/publications/ionising-radiation-dose-comparisons/ionising-radiation-dose-comparisons#:~:text=In%20the%20UK%2C%20Public%20Health,the%20body%20to%20differing%20degrees.
# Average background radiation is about 2700 microsieverts a year which converts to 0.9 counts per second
# Since 1 uSv/hr = 0.0057 CPM

_background_noise_mean = 0.9
_scattering_noise_scaling = 1e-3
_scattering_variance_scaling = 1e9
_transmission_noise_scaling = 1e3 # 2e4


def ct_detect(photons, coeffs, depth, mas=10000, noise=True):

    """ct_detect returns detector photons for given material depths.
    y = ct_detect(photons, coeffs, depth, mas) takes a source energy
    distribution photons (energies), a set of material linear attenuation
    coefficients coeffs (materials, energies), and a set of material depths
    in depth (materials, samples) and returns the detections at each sample
    in y (samples).

    mas defines the current-time-product which affects the noise distribution
    for the linear attenuation"""

    # check photons for number of energies
    if type(photons) != np.ndarray:
        photons = np.array([photons])
    if photons.ndim > 1:
        raise ValueError("input photons has more than one dimension")
    energies = len(photons)

    # check coeffs is of (materials, energies)
    if type(coeffs) != np.ndarray:
        coeffs = np.array([coeffs]).reshape((1, 1))
    elif coeffs.ndim == 1:
        coeffs = coeffs.reshape((1, len(coeffs)))
    elif coeffs.ndim != 2:
        raise ValueError("input coeffs has more than two dimensions")
    if coeffs.shape[1] != energies:
        raise ValueError("input coeffs has different number of energies to input photons")
    materials = coeffs.shape[0]

    # check depth is of (materials, samples)
    if type(depth) != np.ndarray:
        depth = np.array([depth]).reshape((1, 1))
    elif depth.ndim == 1:
        if materials == 1:
            depth = depth.reshape(1, len(depth))
        else:
            depth = depth.reshape(len(depth), 1)
    elif depth.ndim != 2:
        raise ValueError("input depth has more than two dimensions")
    if depth.shape[0] != materials:
        raise ValueError(
            "input depth has different number of materials to input coeffs"
        )
    samples = depth.shape[1]

    # extend source photon array so it covers all samples
    detector_photons = np.zeros([energies, samples])

    for e in range(energies):
        mean = photons[e]
        detector_photons[e] = mean # np.random.poisson(mean)

    source_photons = np.sum(detector_photons, axis=0)

    # calculate array of residual mev x samples for each material in turn
    for m in range(materials):
        detector_photons = attenuate(detector_photons, coeffs[m], depth[m])

    # sum this over energies
    detector_photons = np.sum(detector_photons, axis=0)

    # model noise
    # transmission noise modeled by approximate normal distribution
    if noise:
        detector_photons = np.random.poisson(detector_photons / _transmission_noise_scaling,
                                            detector_photons.shape).astype(np.float) * _transmission_noise_scaling
        background_noise = np.random.poisson(_background_noise_mean * mas,
                                            detector_photons.shape).astype(np.float)
        scattering_noise = _scattering_noise_scaling * _scattering_variance_scaling * np.random.poisson(source_photons / _scattering_variance_scaling,
                                                                        source_photons.shape).astype(np.float)
        detector_photons += scattering_noise + background_noise


    # minimum detection is one photon
    detector_photons = np.clip(detector_photons, 1, None)

    return detector_photons

import math
import numpy as np
import numpy.matlib


def ramp_filter(sinogram, scale, alpha=0.001):
    """Ram-Lak filter with raised-cosine for CT reconstruction

    fs = ramp_filter(sinogram, scale) filters the input in sinogram (angles x samples)
    using a Ram-Lak filter.

    fs = ramp_filter(sinogram, scale, alpha) can be used to modify the Ram-Lak filter by a
    cosine raised to the power given by alpha."""

    # get input dimensions
    (angles, n) = sinogram.shape

    # Set up filter to be at least twice as long as input
    m = np.ceil(np.log(2 * n - 1) / np.log(2))
    m = int(2**m)

    # apply filter to all angles
    print("Ramp filtering")
    # define filter
    omega_max = 2*np.pi/scale

    freq = np.linspace(-omega_max,omega_max,m)
    filter = np.abs(freq)/(2*np.pi)*np.cos(freq/omega_max*np.pi/2)**alpha
    for i in range(angles):
        #take fft for value
        sin_line_ft = np.fft.fftshift(np.fft.fft(sinogram[i],m))
        filt_ft = filter*sin_line_ft
        filt_ft = np.fft.ifftshift(filt_ft)
        sinogram[i] = np.real(np.fft.ifft(filt_ft)[0:n])



    return sinogram

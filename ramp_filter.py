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
	#m = np.ceil(np.log(2 * n - 1) / np.log(2))
	#m = int(2**m)

	# apply filter to all angles
	print("Ramp filtering")
	# define filter
	omega_max = np.pi/scale # factor of 2 taken out according to 3G4 notes: omega_max of sinogram will be half that of omega_max of sinogram padded to length m with zeros
	
	freq = np.linspace(-omega_max, omega_max, 2*n, endpoint=False)
	filt = np.abs(freq)/(2*np.pi)*np.cos(freq/omega_max*np.pi*0.5)**alpha
	filt[n] = filt[n + 1] / 6.0 # value at k=0 changed to one sixth that at k=1 according to handout
	
	for i in range(angles):
		#take fft for value
		sin_line_ft = np.fft.fftshift(np.fft.fft(sinogram[i], 2*n, norm="backward"))
		filt_ft = filt*sin_line_ft
		filt_ft = np.fft.ifftshift(filt_ft)
		sinogram[i] = np.real(np.fft.ifft(filt_ft, norm="backward")[0:n])

	return sinogram

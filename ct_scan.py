import numpy as np
import scipy
from scipy import ndimage
from ct_detect import ct_detect
import math
import sys

def ct_scan(photons, material, phantom, scale, angles, mas=10000):

	"""simulate CT scanning of an object
	scan = ct_scan(photons, material, phantom, scale, angles, mas) takes a phantom
	which contains indices relating to the attenuation coefficients given in
	material.coeffs, and scans it using source energy photons, with given angles and
	current-time product mas.

	scale is the pixel size of the input array phantom, in cm per pixel.
	"""

	# find the coefficients for air
	air = material.name.index('Air')

	# get input image dimensions, and create a coordinate structure
	n = max(phantom.shape)
	xi, yi = np.meshgrid(np.arange(n) - (n/2) + 0.5, np.arange(n) - (n/2) + 0.5)

	# check which materials phantom actually contains, and create single
	# material phantoms for each of these, except for air
	materials = []
	material_phantom = []
	for m in range(0,len(material.coeffs)):
		z0 = (phantom == m).astype(float)
		if (m != air) & (z0.sum()>0):
			materials.append(m)
			material_phantom.append(z0)

	# scan one angle at a time
	scan = np.zeros((angles, n))
	for angle in range(angles):

		sys.stdout.write("Scanning angle: %d   \r" % (angle + 1) )

		# Get rotated coordinates for interpolation
		p = -math.pi / 2 - angle * math.pi / angles
		x0 = xi * math.cos(p) - yi * math.sin(p) + (n/2) - 0.5
		y0 = xi * math.sin(p) + yi * math.cos(p) + (n/2) - 0.5

		# For each material, add up how many pixels contain this on each ray
		depth = np.zeros((len(material.coeffs), n))

		for index, m in enumerate(materials):
			interpolated = scipy.ndimage.map_coordinates(material_phantom[index], [y0, x0], order=1, mode='constant', cval=0, prefilter=False)
			depth[m] = np.sum(interpolated, axis=0)

		# only necessary for more complex forms of interpolation above
		depth = np.clip(depth, 0, None)

		# ensure an appropriate amount of air is included in the calculation
		# to account for the scan being circular, but the phantom being square
		# diameter of circle taken to be twice the phantom side length
		depth[air] = 2 * n - np.sum(depth, axis=0)

		# scale the depth appropriately and calculate detections for this set of
		# materials
		depth *= scale
	
		scan[angle] = ct_detect(photons, material.coeffs, depth, mas)
	
	sys.stdout.write("\n")

	return scan
import numpy as np
import numpy.matlib
import math

def phantom(ellipses, n):
	"""generates an artificial phantom given ellipse parameters and size n"""

	#convert to numpy array
	ellipses = np.array(ellipses)

	#handle both single ellipse and arrays of ellipses
	if len(ellipses.shape) == 1:
		ellipses = np.array([ellipses])

	phantom_instance = np.zeros((n, n))

	xax = np.linspace(-1.0, 1.0, n, endpoint=True)
	xg = np.matlib.repmat(xax, n, 1) # x coordinates, the y coordinates are rot90(xg)

	for ellipse in ellipses:
		asq = ellipse[1] ** 2       # a^2
		bsq = ellipse[2] ** 2       # b^2
		phi = ellipse[5] * math.pi / 180  # rotation angle in radians
		x0 = ellipse[3]          # x offset
		y0 = ellipse[4]          # y offset
		a = ellipse[0]           # Amplitude change for this ellipse
		x_center = xg - x0                # Center the ellipse
		y_center = np.rot90(xg) - y0
		cosp = math.cos(phi)
		sinp = math.sin(phi)
		values = (((x_center * cosp + y_center * sinp) ** 2) / asq + ((y_center *cosp - x_center * sinp) ** 2) / bsq)

		for index, element in np.ndenumerate(values):
				if element <= 1:
					phantom_instance[index] = phantom_instance[index] + a

	return phantom_instance
	
def ct_phantom(names, n, type, metal=None):

	""" ct_phantom create phantom for CT scanning
		x = ct_phantom(names, n, type, metal) creates a CT phantom in x of
		size (n X n), and type given by type:

		1 - simple circle for looking at calibration issues
		2 - point attenuator for looking at resolution
		3 - single large hip replacement
		4 - bilateral hip replacement
		5 - sphere with three satellites
		6 - disc and other sphere
		7 - pelvic fixation pins
		8 - resolution phantom

		For types 1-2, the whole phantom is of type 'metal', which defaults
		to 'Soft Tissue' if not given. This must match one of the material
		names given in 'names'

		For types 3-8, the metal implants are of type 'metal', which defaults
		to 'Titanium' if not given.

		The output x has data values which correspond to indices in the names
		array, which must also contain 'Air', 'Adipose', 'Soft Tissue' and 'Bone'.
	"""  

	# Get material locations
	air = names.index('Air')
	adipose =  names.index('Adipose')
	bone =  names.index('Bone')
	if type < 3:
		if metal is None:
			tissue = names.index('Soft Tissue')
		else:
			tissue = names.index(metal)
	else:
		tissue =  names.index('Soft Tissue')
		if metal is None:
			nmetal = names.index('Titanium')
		else:
			nmetal =  names.index(metal)

	if type == 1:

		# simple circle for looking at calibration
		t = [1, 0.8, 0.8, 0.0, 0.0, 0]
		x = phantom(t, n)

		for index, value in np.ndenumerate(x):
			if value >= 1:
				x[index] = tissue

	elif type == 2:
		
		# impulse for looking at resolution
		x = np.zeros((n, n))
		x[int(n / 2)][int(n / 2)] = tissue
		
	elif type == 8:

		# resolution phantom
		t = [1, 0.8, 0.8, 0.0, 0.0, 0]
		x = phantom(t, n)

		for index, value in np.ndenumerate(x):
			if value >= 1:
				x[index] = tissue

		for r in np.arange(n * 0.04, n * 0.4, n * 0.04):
			angles = np.cumsum(np.arange(0, 2*math.pi, n * 0.002 / r))
			angles = angles[angles < (math.pi * 2)]
			for a in angles:
				x[int(round(n / 2 + r * math. cos(a)))][int(round(n / 2 + r * math.sin(a)))] = nmetal
		
	else:
		
		# This creates a generic human hip cross-section
		t =  [[1, 0.57, 0.52, -0.35, 0.1, 0],
				[1, 0.57, 0.52, 0.35, 0.1, 0],
				[1, 0.52, 0.45, 0, -0.08, 0]]
		x = phantom(t, n)

		for index, value in np.ndenumerate(x):
			if value >= 1:
				x[index] = tissue

		a = [[1, 0.55, 0.5, -0.35, 0.1, 0],
			[1, 0.55, 0.5, 0.35, 0.1, 0],
			[1, 0.5, 0.43, 0, -0.08, 0]]
		x = x + phantom(a, n)

		for index, value in np.ndenumerate(x):
			if value > tissue:
				x[index] = adipose

		t =  [[1, 0.37, 0.35, -0.42, 0.03, 0],
			[1, 0.37, 0.35, 0.42, 0.03, 0],
			[1, 0.24, 0.16, -0.3, 0.28, 20],
			[1, 0.24, 0.16, 0.3, 0.28, -20],
			[1, 0.4, 0.2, 0, -0.15, 0]]
		x = x + phantom(t, n)

		for index, value in np.ndenumerate(x):
			if value > adipose:
				x[index] = tissue

		b = [[1, 0.16, 0.12, -0.54, -0.01, 0],
			[-1, 0.11, 0.10, -0.53, -0.01, 0],
			[1, 0.16, 0.12, 0.54, -0.01, 0],
			[-1, 0.11, 0.10, 0.53, -0.01, 0],
			[1, 0.1, 0.09, -0.25, 0.25, 140],
			[-1, 0.07, 0.06, -0.25, 0.25, 140],
			[1, 0.18, 0.05, -0.05, -0.15, 100],
			[-1, 0.14, 0.03, -0.05, -0.15, 100],
			[1, 0.1, 0.09, 0.25, 0.25, -140],
			[-1, 0.07, 0.06, 0.25, 0.25, -140],
			[1, 0.18, 0.05, 0.05, -0.15, -100],
			[-1, 0.14, 0.03, 0.05, -0.15, -100]]
		x = x + phantom(b, n)

		for index, value in np.ndenumerate(x):
			if value > tissue:
				x[index] = bone
		
		# this adds a metal implant
		if nmetal > tissue:
			if type == 3:
				# single large hip replacement
				m = [100, 0.1, 0.1, -0.48, -0.01, 0]
			elif type == 4:
				# bilateral hip replacement
				m = [[100, 0.1, 0.1, -0.48, -0.01, 0],
					[100, 0.08, 0.06, 0.48, 0, 0]]
			elif type == 5:
				# sphere with three satellites
				m = [[100, 0.05, 0.05, -0.43, -0.03, 0],
					[100, 0.02, 0.02, -0.53, 0.04, 0],
					[100, 0.02, 0.02, -0.53, -0.10, 0],
					[100, 0.02, 0.02, -0.31, -0.03, 0]]
			elif type == 6:
				# disc and other sphere
				m = [[100, 0.08, 0.08, -0.58, 0.01, 0],
					[-100, 0.05, 0.05, -0.58, 0.01, 0],
					[100, 0.05, 0.05, -0.25, -0.1, 0]]
			elif type == 7:
				# pins
				m = [[100, 0.02, 0.025, -0.08, -0.03, 0],
					[100, 0.025, 0.025, -0.03, -0.25, 0],
					[100, 0.025, 0.025, -0.3, 0.25, 0],
					[100, 0.025, 0.025, -0.2, 0.25, 0]]
			
			x = x + phantom(m, n)

			for index, value in np.ndenumerate(x):
				if value > bone:
					x[index] = nmetal

	# make sure the remainder is set to air
	for index, value in np.ndenumerate(x):
		if value == 0:
			x[index] = air

	x = np.flipud(x)
	
	return x
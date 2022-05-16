import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *

def test_ct_calibrate(material, source):
	# tests 'ct_calibrate'
	# this test relies on correct functionality of the 'source' and 'material' classes and the 'ct_phantom' and 'ct_scan' functions
	
	# parameters
	photons = source.photon('80kVp, 2mm Al')
	scale = 0.1
	sinogram = ct_scan(photons, material, ct_phantom(material.name, 256, 3, 'Titanium'), scale, 256)
	
	result = ct_calibrate(photons, material, sinogram, scale)
	
	min_result = min([min(row) for row in result])
	max_result = max([max(row) for row in result])
	
	if min_result > 1e-1:
		print("Warning: calibrated sinogram has no values vlose to zero.")
	
	if min_result < -1.0e-4:
		print("Error: calibrated sinogram has negative attenuation coefficient values.")
	
	if max_result > 1e2:
		print("Warning: calibrated sinogram has attenuation coefficient values exceeding 100.")


def run_tests(material, source):
	print("Testing 'ct_calibrate' >>>")
	test_ct_calibrate(material, source)
	print("Done with 'ct_calibrate'.")
	
	

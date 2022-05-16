import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *

def test_ct_calibrate():
	# tests 'ct_calibrate'
	# this test relies on correct functionality of the 'source' and 'material' classes and the 'ct_phantom' and 'ct_scan' functions
	
	# parameters
	photons = source.photon('80kVp, 2mm Al')
	scale = 0.1
	sinogram = ct_scan(photons, material, ct_phantom(material.name, 256, 3, 'Titanium'), scale, 256)
	
	result = ct_calibrate(photons, material, sinogram, scale)
	
	print(result)


def run_tests():
	test_ct_calibrate()

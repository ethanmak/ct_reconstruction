import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *
import matplotlib.pyplot as plt
import inspect


def run_tests(material, source):
	testfunctions = [(name, obj) for name, obj in inspect.getmembers(sys.modules[__name__])
					 if (inspect.isfunction(obj) and name.startswith('test') and name != 'testall')]
	for name, func in testfunctions:
		print('Start testing \'{}\''.format(name))
		func(material, source)
		print('Finished testing \'{}\''.format(name))


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


def test_scan_and_reconstruct_shape(material,source):
	'''
	Function to create a reconstruction based on a phantom and display them side by side for checking.
	The main goal of this function is to ensure that the phantom and reconstructed images have similar
	features in shape, although the brightness will differ based on the materials of the phantom.
	The plots are drawn and saved side by side to ensure easy comparison

	:param material:
	:param source:
	:return:
	'''
	fig, ax = plt.subplots(1,2)

	ax[0].axis("off")
	ax[1].axis("off")

	photons = source.photon('80kVp, 2mm Al')
	scale = 0.1
	phantom = ct_phantom(material.name,256,3,'Titanium')
	scan = scan_and_reconstruct(photons,material,phantom,scale,angles = 256)
	ax[0].imshow(phantom, cmap = "Greys_r")
	ax[0].set_aspect("equal","box")
	ax[0].set_title("Original Phantom")
	ax[1].imshow(scan, cmap = "Greys_r")
	ax[1].set_aspect("equal", "box")
	ax[1].set_title("Reconstructed Image")
	plt.savefig("results/test_scan_and_reconstruct_shape.png")

def skip_test_scan_and_reconstruct_hounsfield(material, source):
	phantom = ct_phantom(material.name, 256, 3, 'Titanium')
	s = fake_source(source.mev, 0.08, method="ideal")
	scan = scan_and_reconstruct(s, material, phantom, 0.1, angles=256)

	max_hounsfield = np.max(scan) # this will be found at the titanium implant where the attenuation is highest

	f = open("results/scan_and_reconstruct_hounsfield.txt", mode="w")
	f.write("Max value of reconstructed image is: {}".format(max_hounsfield))
	f.close()

	assert np.abs(max_hounsfield - 9000) < 100

def test_scan_and_reconstruct_attenuation_coefficient(material, source):
	'''
	This function tests that attenuation coefficients are being properly calculated.
	This is done by creating a phantom with a titanium implant and using an ideal source
	to scan the phantom, yielding a max attenuation coefficient for the water which can be
	compared to the tabled coefficient. We accept the value if it is within 5%

	:param material: Material instance
	:param source: Source instance
	:return: None
	'''
	phantom = ct_phantom(material.name, 256, 1, 'Water')
	s = fake_source(source.mev, 0.08, method="ideal")
	scan = scan_and_reconstruct(s, material, phantom, 0.1, angles=256, hounsfield=False)

	coeff = np.mean(scan[120:130, 120:130])  # this will be found at the titanium implant where the attenuation is highest
	expected = material.coeff('Water')[np.argmax(s)]

	f = open("results/scan_and_reconstruct_attenuation_coefficient.txt", mode="w")
	f.write("Water attenuation coefficent of reconstructed image is: {} \n".format(coeff))
	f.write("Expected water attenuation coefficent of reconstructed image is: {} \n".format(expected))
	f.close()

	assert np.abs(coeff - expected) / expected < 0.05

def test_attenuate(material, source):
	'''
	Tests the attenuate function to ensure that it works as expected.
	This is done by creating mock data and inputing into the function.

	:param material: Material instance
	:param source: Source instance
	:return: None
	'''
	energies = np.array([[1., 1.], [1., 1.]])
	coeff = np.array([1, 2])
	depths = np.array([3, 5])

	output = np.array([[np.exp(-3), np.exp(-5)], [np.exp(-6), np.exp(-10)]])

	assert (attenuate(energies, coeff, depths) == output).all()
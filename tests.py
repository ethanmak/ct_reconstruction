import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *
import matplotlib.pyplot as plt

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
	


def test_scan_and_reconstruct(material,source):
	"""
	Function to create a reconstruction based on a phantom and display them side by side for checking.
	The plots are drawn and saved side by side to ensure easy comparison
	"""
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
	ax[1].set_aspect("equal","box")
	ax[1].set_title("Reconstructed Image")
	plt.savefig("saved_results/test_scan_and_reconstruct.png")
	plt.show()

def test_scan_and_reconstruct_hounsfield(material, source):
	phantom = ct_phantom(material.name, 256, 3, 'Titanium')
	s = fake_source([0.08], 0.1, method="ideal")
	scan = scan_and_reconstruct(s, material, phantom, 0.1, angles=256)

	max_hounsfield = np.max(scan) # this will be found at the titanium implant where the attenuation is highest

	f = open("results/scan_and_reconstruct_hounsfield.txt", mode="w")
	f.write("Max value of reconstructed image is: {}".format(max_hounsfield))
	f.close()

	assert np.abs(max_hounsfield - 9000) < 100

def test_attenuate(material, source):
	energies = np.array([[1., 1.], [1., 1.]])
	coeff = np.array([1, 2])
	depths = np.array([3, 5])

	output = np.array([[np.exp(-3), np.exp(-5)], [np.exp(-6), np.exp(-10)]])

	assert (attenuate(energies, coeff, depths) == output).all()
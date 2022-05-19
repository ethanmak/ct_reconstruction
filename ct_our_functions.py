import matplotlib.pyplot as plt
import numpy.matlib
import numpy as np
import scipy
import math
import pydicom
from material import *
from source import *
from attenuate import *
from ct_detect import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from ct_scan import *
from ct_calibrate import *
from back_project import *
from scan_and_reconstruct import *
from create_dicom import *
from xtreme import *

# does 'ct_detect' for multiple materials and plots the results as suggested in the handout
def ct_detect_multiple(material_name_list, source, material):
	residual_energies = []
	for material_name in material_name_list:
		residual_energy = ct_detect(source.photon('100kVp, 2mm Al'), material.coeff(material_name), np.arange(0, 10.1, 0.1), 1)
		residual_energies.append(residual_energy)
		plt.plot(np.log(residual_energy))
		plt.xlabel("Depth (cm)")
		plt.ylabel("Residual energy (mev)")
		plt.title(material_name)
		plt.show()
	return residual_energies

# does 'ct_scan' for multiple materials (at one no. of angles) and then at multiple nos. of angles (for one material) and draws them as suggested in the handout. also saves the results to files
def ct_scan_multiple(material_name_list, scan_angleN_list, index_of_angleN_for_all_materials, index_of_material_for_all_angleNs, source, material):

	phantoms = [ct_phantom(material.name, 256, 3, material_name) for material_name in material_name_list]
	
	material_sinograms = []
	for (material_name, phantom) in zip(material_name_list, phantoms):
		material_sinogram = ct_scan(source.photon('100kVp, 2mm Al'), material, phantom, 0.1, scan_angleN_list[index_of_angleN_for_all_materials])
		material_sinograms.append(material_sinogram)
		np.savetxt("sinogram_" + material_name + "_" + str(scan_angleN_list[index_of_angleN_for_all_materials]) + ".txt", material_sinogram)
	
	angleN_sinograms = []
	for angleN in scan_angleN_list:
		angleN_sinogram = ct_scan(source.photon('100kVp, 2mm Al'), material, phantoms[index_of_material_for_all_angleNs], 0.1, angleN)
		angleN_sinograms.append(angleN_sinogram)
		np.savetxt("sinogram_" + material_name_list[index_of_material_for_all_angleNs] + "_" + str(angleN) + ".txt", angleN_sinogram)
	
	for material_sinogram in material_sinograms:
		draw(material_sinogram)
	
	for angleN_sinogram in angleN_sinograms:
		draw(angleN_sinogram)
	
	return (material_sinograms, angleN_sinograms)


class EdmundsConstants:
	material_name_list = ['Air', 'Soft Tissue', 'Water', 'Bone', 'Titanium']
	scan_angleN_list = [32, 64, 128, 256, 512]

def normalize_to_greyscale(X):
	return (X - np.min(X)) / (np.max(X) - np.min(X))

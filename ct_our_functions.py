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


class EdmundsConstants:
	material_name_list = ['Air', 'Soft Tissue', 'Water', 'Bone', 'Titanium']

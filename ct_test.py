from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *
import numpy as np

#create object instances
material = Material()
source = Source()

def test_flat():
	p = ct_phantom(material.name, 256, 1)
	s = fake_source(material.mev, 0.1, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)
	plot(np.transpose([y[128,:], 0.207*np.ones([256])]))
	s = source.photons[5]
	y = scan_and_reconstruct(s, material, p, 0.05, 256)
	plot(np.transpose([y[128,:], 0.365*np.ones([256])]))
	y = scan_and_reconstruct(s, material, p, 0.5, 256)
	draw(y, caxis=[0, 0.365])

test_flat()

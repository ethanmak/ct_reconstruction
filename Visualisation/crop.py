import numpy as np
from create_dicom import *

def crop_dcms(directoryNprefix, n):
	for i in range(n):
		fileName = directoryNprefix + format(i + 1, '04d') + ".dcm"
		print(fileName)

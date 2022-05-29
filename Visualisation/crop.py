import numpy as np
from create_dicom import *
import pydicom

def crop_dcms(inputDirectory, prefix, outputDirectory, crop_rad, crop_val, first, last):
	for i in range(last - first + 1):
		nString = format(i + first, '04d')
		fileName = inputDirectory + prefix + nString + ".dcm"
		dcm = pydicom.dcmread(fileName)
		arr = dcm.pixel_array
		(height, width) = arr.shape;
		radSqd = crop_rad*crop_rad
		for r in range(height):
			dr = r - height*0.5
			for c in range(width):
				dc = c - width*0.5
				if dr*dr + dc*dc > radSqd:
					arr[r][c] = crop_val
		dcm.PixelData = arr.tobytes()
		dcm.save_as(outputDirectory + prefix + "cropped_" + nString + ".dcm")

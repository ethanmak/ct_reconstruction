import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import os

def draw(data, map='gray', caxis=None):
	"""Draw an image"""
	create_figure(data, map, caxis)
	plt.show()


def plot(data):
	"""plot a graph"""
	plt.plot(data)
	plt.show()

def save_draw(data, storage_directory, file_name, map='gray', caxis=None, title=None):
	"""save an image"""
	create_figure(data, map, caxis, title)
	full_path = get_full_path(storage_directory, file_name)
	plt.savefig(full_path)
	plt.close()

def save_plot(data, storage_directory, file_name, xlim=None, ylim=None, title=None):
	"""save a graph"""
	full_path = get_full_path(storage_directory, file_name)
	plt.plot(data)

	if xlim is not None:
		plt.xlim( xlim[0], xlim[1] )
	if ylim is not None:
		plt.ylim( ylim[0], ylim[1] )
	if title is not None:
		plt.title( title )

	plt.savefig(full_path)
	plt.close()

def save_numpy_array(data, storage_directory, file_name):
	"""save a numpy array in .npy format"""

	full_path = get_full_path(storage_directory, file_name)

	np.save(full_path, data)

def load_numpy_array(storage_directory, file_name):
	"""load a .npy file into numpy array"""

	full_path = os.path.join(storage_directory, file_name)

	#add .npy extension if needed
	if not full_path.endswith('.npy'):
		full_path = full_path + '.npy'

	if not os.path.exists(full_path):
		raise Exception('File named ' + full_path + ' does not exist')

	return np.load(full_path)

def get_full_path(storage_directory, file_name):
	#create storage_directory if needed
	if not os.path.exists(storage_directory):
		os.makedirs(storage_directory)

	full_path = os.path.join(storage_directory, file_name)

	return full_path

def create_figure(data, map, caxis=None, title=None):
	fig, ax = plt.subplots()

	plt.axis('off') # no axes

	if caxis is None:
		im = plt.imshow(data, cmap=map)
	else:
		im = plt.imshow(data, cmap=map, vmin=caxis[0], vmax=caxis[1])
	if title is not None:
		plt.title( title )

	# equal aspect ratio
	ax.set_aspect('equal', 'box')
	plt.tight_layout()

	#add colorbar
	plt.colorbar(im, orientation='vertical')

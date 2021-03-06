import numpy as np
import math
import scipy
from scipy import interpolate
import sys
import multiprocessing as mp

def interpolate_and_process(x0, sino, ns, angles):
    interpolated_func = scipy.interpolate.interp1d(np.arange(0, ns, 1), sino, kind='linear', copy=False,
                                                   assume_sorted=True, bounds_error=False, fill_value=0, axis=0)
    return interpolated_func(x0) * (np.pi / angles)

def back_project(sinogram, skip=1):

    """back_project back-projection to reconstruct CT data
    back_project(sinogram) back-projects the filtered sinogram
    (angles x samples) to create the reconstruted data (samples x
    samples)"""

    # get input dimensions
    ns = sinogram.shape[1]
    angles = sinogram.shape[0]
    n = int(math.floor((ns - 1) // skip) + 1)

    # zero output and form input coordinates
    # these have centre in the middle of the image
    reconstruction = np.zeros((n, n))
    xi, yi = np.meshgrid(
        np.arange(0, ns, skip) - (ns / 2) + 0.5, np.arange(0, ns, skip) - (ns / 2) + 0.5
    )

    # back project over each angle in turn
    for angle in range(angles):
        sys.stdout.write("Reconstructing angle: %d   \r" % (angle + 1))

        # Form rotated coordinates for output interpolation
        # the rotation is about the middle of the image,
        # but the output coordinates need to be relative to the top left
        p = math.pi / 2 + angle * math.pi / angles
        x0 = xi * math.cos(p) - yi * math.sin(p) + (ns / 2) - 0.5

        # interpolate and add this data to output
        # remembering to multiply by dtheta as well as sum
        # Either of the following options will work
        # -----
        x2 = scipy.interpolate.interp1d(np.arange(0, ns, 1), sinogram[angle], kind='linear', copy=False, assume_sorted=True, bounds_error=False, fill_value=0, axis=0)
        reconstruction = reconstruction + x2(x0) * (math.pi / angles)
        # -----
        # x2 = scipy.ndimage.map_coordinates(
        #     sinogram[angle], [x0], order=1, mode="constant", cval=0, prefilter=False
        # )
        # reconstruction = reconstruction + x2 * (math.pi / angles)
        # -----

    # ensure any data outside the reconstructed circle is set to invalid
    reconstruction[np.where((xi**2 + yi**2) > (ns / 2) ** 2)] = -1

    sys.stdout.write("\n")

    return reconstruction

def back_project_improved(sinogram, skip=1):

    """back_project back-projection to reconstruct CT data
    back_project(sinogram) back-projects the filtered sinogram
    (angles x samples) to create the reconstruted data (samples x
    samples)"""

    # get input dimensions
    ns = sinogram.shape[1]
    angles = sinogram.shape[0]
    n = int(math.floor((ns - 1) // skip) + 1)

    # zero output and form input coordinates
    # these have centre in the middle of the image
    reconstruction = np.zeros((n, n))
    xi, yi = np.meshgrid(
        np.arange(0, ns, skip) - (ns / 2) + 0.5, np.arange(0, ns, skip) - (ns / 2) + 0.5
    )

    # Form rotated coordinates for output interpolation
    # the rotation is about the middle of the image,
    # but the output coordinates need to be relative to the top left
    list_angles = np.array(range(angles))
    p = np.pi / 2 + list_angles * np.pi / angles
    params = []

    for angle in range(angles):
        params.append((xi * np.cos(p[angle]) - yi * np.sin(p[angle]) + (ns / 2) - 0.5,
                      sinogram[angle], ns, angles))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        reconstruction = np.sum(pool.starmap(interpolate_and_process, params), axis=0)

    # ensure any data outside the reconstructed circle is set to invalid
    reconstruction[np.where((xi**2 + yi**2) > (ns / 2) ** 2)] = -1

    return reconstruction

import matplotlib.pyplot as plt
import numpy.matlib
import numpy as np
import scipy
import math
import pydicom
import os
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
import tests

material = Material()
source = Source()

tests.run_tests(material, source)


import time
import sys
import numpy as np
import serial
import pyvisa
import datetime
import pathlib
# import tty
import os
import ctypes
from dataclasses import dataclass
from typing import Generator
if os.name != 'nt': # Linux
    import termios
    import tty
else: # Windows
    import msvcrt
# import fn_ctrl
from picoscope_ctrl import *

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok, mV2adc


import numpy as np
from vtk.util.numpy_support import numpy_to_vtk

arr = np.array([1, 2, 3], dtype=np.float32)
vtk_arr = numpy_to_vtk(arr)
print(vtk_arr)
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

if os.name != 'nt': # Linux
    PRINTER_SERIAL = '/dev/cu.usbserial-2130'
    SCOPE_SERIAL = 'DS1ZE26CM00690'
    GENERATOR_SERIAL = '2120'
else: # Windows
    PRINTER_SERIAL = 'COM10'



chandle = ctypes.c_int16()
status = {}
scope = PicoScope(chandle, status)

while(1):
    start = time.time()
    chA, _ = scope.read_magnitude_avg(num_samples=10)
    end = time.time()
    print(end-start, chA)
#!/usr/bin/env python3
"""
grab_fixed_duration.py
----------------------

• Captures a single trace from CH1 and CH2 over USB-TMC.
• The length of the captured record is controlled by CAPTURE_SEC.
• Saves each trace to CSV: time [s], volts [V].

Tested on Rigol DS1000Z-E family (DS1202Z-E).
"""

import pyvisa, numpy as np, pathlib, datetime
import os
import sys


# --------------------------------------------------------------------------
# USER SETTINGS
RES          = 'USB0::6833::1303::DS1ZE26CM00690::0::INSTR'   # <- your device
CAPTURE_SEC  = 10e-6    # <-- desired duration in seconds (e.g. 0.005 s = 5 ms)
RAW_MODE     = False   # False = screen memory (~12 kpts) | True = deep memory
OUTDIR       = os.path.join(os.getcwd(), "out")
os.makedirs(OUTDIR, exist_ok=1)
# --------------------------------------------------------------------------

H_DIV = 10            # DS1000Z has 10 horizontal divisions

rm     = pyvisa.ResourceManager('@py')
scope  = rm.open_resource(RES)
scope.timeout    = 10000           # ms
scope.chunk_size = 1 << 20         # 1 MiB

# ----- configure horizontal time base for the requested capture length -----
scope.write(f':TIM:MODE MAIN')
scope.write(f':TIM:SCAL {CAPTURE_SEC / H_DIV}')

def grab(ch: int):
    """Return (t, v) arrays for channel *ch*."""
    scope.write(':STOP')
    scope.write(f':WAV:SOUR CHAN{ch}')
    scope.write(f':WAV:MODE {"RAW" if RAW_MODE else "NORM"}')
    scope.write(':WAV:FORM BYTE')

    # scaling factors --------------------------------------------------------
    pre  = list(map(float, scope.query(':WAV:PRE?').split(',')))
    xinc, xorg = pre[4], pre[5]
    yinc, yorg, yref = pre[7:10]

    # binary block read (manual, robust) -------------------------------------
    scope.write(':WAV:DATA?')
    pound  = scope.read_bytes(1)        # b'#'
    ndig   = int(scope.read_bytes(1))   # number of digits that follow
    nbytes = int(scope.read_bytes(ndig))
    raw    = scope.read_bytes(nbytes)

    volts = (np.frombuffer(raw, dtype=np.uint8) - yref) * yinc + yorg
    time  = np.arange(volts.size) * xinc + xorg
    return time, volts

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
ampl = float(sys.argv[1])

for ch in (1, 2):
    t, v = grab(ch)
    csv = os.path.join(OUTDIR, f'Ampl_{ampl}_Ch{ch}_{timestamp}.csv')
    np.savetxt(csv, np.column_stack([t, v]),
               delimiter=',', header='t_s,V', comments='')
    print(f'CH{ch}: {v.size:,} points (≈{t[-1]-t[0]:.6f} s) → {csv}')

scope.write(':RUN')
scope.close()


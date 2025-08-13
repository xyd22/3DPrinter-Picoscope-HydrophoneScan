#
# Copyright (C) 2018-2022 Pico Technology Ltd. See LICENSE file for terms.
#
# PS5000A BLOCK MODE EXAMPLE
# This example opens a 5000a driver device, sets up two channels and a trigger then collects a block of data.
# This data is then plotted as mV against time in ns.

import ctypes
import time
import numpy as np
from scipy.signal import resample
from scipy.fft import fft, fftfreq
from picosdk.ps5000a import ps5000a as ps
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import subprocess
from picosdk.functions import adc2mV, assert_pico_ok, mV2adc
from scipy.signal import butter, filtfilt


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


class PicoScope():
    def __init__(self, chandle, status):
        self.chandle = chandle
        self.status = status
        self.pico_config = pico_init(chandle, status)

    def read_magnitude(self):
        return pico_read_magnitude(self.chandle, self.status, self.pico_config)

    def read_magnitude_avg(self, num_samples=10):
        return pico_read_magnitude_avg(self.chandle, self.status, self.pico_config, num_samples)

    def close(self):
        pico_close(self.chandle, self.status)


def pico_init(chandle, status):
    # Open 5000 series PicoScope
    # Resolution set to 12 Bit
    resolution =ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]
    # Returns handle to chandle for use in future API functions
    status["openunit"] = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, resolution)

    try:
        assert_pico_ok(status["openunit"])
    except: # PicoNotOkError:

        powerStatus = status["openunit"]

        if powerStatus == 286:
            status["changePowerSource"] = ps.ps5000aChangePowerSource(chandle, powerStatus)
        elif powerStatus == 282:
            status["changePowerSource"] = ps.ps5000aChangePowerSource(chandle, powerStatus)
        else:
            raise

        assert_pico_ok(status["changePowerSource"])

    # Set up channel A
    # handle = chandle
    channelA = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    # enabled = 1
    coupling_type = ps.PS5000A_COUPLING["PS5000A_DC"]
    chARange = ps.PS5000A_RANGE["PS5000A_20MV"]
    # analogue offset = 0 V
    status["setChA"] = ps.ps5000aSetChannel(chandle, channelA, 1, coupling_type, chARange, 0)
    assert_pico_ok(status["setChA"])

    # Set up channel B
    # handle = chandle
    channelB = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
    # enabled = 1
    # coupling_type = ps.PS5000A_COUPLING["PS5000A_DC"]
    chBRange = ps.PS5000A_RANGE["PS5000A_20V"]
    # analogue offset = 0 V
    status["setChB"] = ps.ps5000aSetChannel(chandle, channelB, 1, coupling_type, chBRange, 0)
    assert_pico_ok(status["setChB"])

    # Set up channel C
    # handle = chandle
    channelC = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_C"]
    # enabled = 1
    # coupling_type = ps.PS5000A_COUPLING["PS5000A_DC"]
    chCRange = ps.PS5000A_RANGE["PS5000A_2V"]
    # analogue offset = 0 V
    status["setChC"] = ps.ps5000aSetChannel(chandle, channelC, 1, coupling_type, chCRange, 0)
    assert_pico_ok(status["setChC"])

    range_config = {
        "chARange": chARange,
        "chBRange": chBRange,
        "chCRange": chCRange,
    }

    # find maximum ADC count value
    # handle = chandle
    # pointer to value = ctypes.byref(maxADC)
    maxADC = ctypes.c_int16()
    status["maximumValue"] = ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
    assert_pico_ok(status["maximumValue"])

    # Set up simple trigger
    # handle = chandle
    # enabled = 1
    # source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
    threshold = int(mV2adc(5000, chBRange, maxADC))
    # direction = PS5000A_RISING = 2
    # delay = 0 s
    # auto Trigger = 1000 ms
    status["trigger"] = ps.ps5000aSetSimpleTrigger(chandle, 1, source, threshold, 2, 0, 1000)
    assert_pico_ok(status["trigger"])

    # Set number of pre and post trigger samples to be collected (ns)
    preTriggerSamples = 250 # ~2us
    postTriggerSamples = 10000 # 125 samples/us
    maxSamples = preTriggerSamples + postTriggerSamples

    # Get timebase information
    # Warning: When using this example it may not be possible to access all Timebases as all channels are enabled by default when opening the scope.  
    # To access these Timebases, set any unused analogue channels to off.
    # handle = chandle
    timebase = 3
    # noSamples = maxSamples
    # pointer to timeIntervalNanoseconds = ctypes.byref(timeIntervalns)
    # pointer to maxSamples = ctypes.byref(returnedMaxSamples)
    # segment index = 0
    timeIntervalns = ctypes.c_float()
    returnedMaxSamples = ctypes.c_int32()
    status["getTimebase2"] = ps.ps5000aGetTimebase2(chandle, timebase, maxSamples, ctypes.byref(timeIntervalns), ctypes.byref(returnedMaxSamples), 0)
    assert_pico_ok(status["getTimebase2"])
    time_interval = timeIntervalns.value

    trigger_config = {
        "timebase": timebase,
        "time_interval": time_interval,
        "preTriggerSamples": preTriggerSamples,
        "postTriggerSamples": postTriggerSamples,
        "maxSamples": maxSamples,
    }

    # Run block capture
    # handle = chandle
    # number of pre-trigger samples = preTriggerSamples
    # number of post-trigger samples = PostTriggerSamples
    # timebase = 8 = 80 ns (see Programmer's guide for mre information on timebases)
    # time indisposed ms = None (not needed in the example)
    # segment index = 0
    # lpReady = None (using ps5000aIsReady rather than ps5000aBlockReady)
    # pParameter = None
    status["runBlock"] = ps.ps5000aRunBlock(chandle, preTriggerSamples, postTriggerSamples, timebase, None, 0, None, None)
    assert_pico_ok(status["runBlock"])

    # Check for data collection to finish using ps5000aIsReady
    ready = ctypes.c_int16(0)
    check = ctypes.c_int16(0)
    while ready.value == check.value:
        status["isReady"] = ps.ps5000aIsReady(chandle, ctypes.byref(ready))

    # Create buffers ready for assigning pointers for data collection
    bufferAMax = (ctypes.c_int16 * maxSamples)()
    bufferAMin = (ctypes.c_int16 * maxSamples)() # used for downsampling which isn't in the scope of this example
    bufferBMax = (ctypes.c_int16 * maxSamples)()
    bufferBMin = (ctypes.c_int16 * maxSamples)() # used for downsampling which isn't in the scope of this example
    bufferCMax = (ctypes.c_int16 * maxSamples)()
    bufferCMin = (ctypes.c_int16 * maxSamples)() # used for downsampling which isn't in the scope of this example
    buffer_config = {
        "bufferAMax": bufferAMax,
        "bufferAMin": bufferAMin,
        "bufferBMax": bufferBMax,
        "bufferBMin": bufferBMin,
        "bufferCMax": bufferCMax,
        "bufferCMin": bufferCMin,
    }

    # Set data buffer location for data collection from channel A
    # handle = chandle
    source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    # pointer to buffer max = ctypes.byref(bufferAMax)
    # pointer to buffer min = ctypes.byref(bufferAMin)
    # buffer length = maxSamples
    # segment index = 0
    # ratio mode = PS5000A_RATIO_MODE_NONE = 0
    status["setDataBuffersA"] = ps.ps5000aSetDataBuffers(chandle, source, ctypes.byref(bufferAMax), ctypes.byref(bufferAMin), maxSamples, 0, 0)
    assert_pico_ok(status["setDataBuffersA"])

    # Set data buffer location for data collection from channel B
    # handle = chandle
    source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
    # pointer to buffer max = ctypes.byref(bufferBMax)
    # pointer to buffer min = ctypes.byref(bufferBMin)
    # buffer length = maxSamples
    # segment index = 0
    # ratio mode = PS5000A_RATIO_MODE_NONE = 0
    status["setDataBuffersB"] = ps.ps5000aSetDataBuffers(chandle, source, ctypes.byref(bufferBMax), ctypes.byref(bufferBMin), maxSamples, 0, 0)
    assert_pico_ok(status["setDataBuffersB"])

    # Set data buffer location for data collection from channel C
    source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_C"]
    status["setDataBuffersC"] = ps.ps5000aSetDataBuffers(chandle, source, ctypes.byref(bufferCMax), ctypes.byref(bufferCMin), maxSamples, 0, 0)
    assert_pico_ok(status["setDataBuffersC"])

    pico_config = {
        "maxADC": maxADC,
        "trigger_config": trigger_config,
        "buffer_config": buffer_config,
        "range_config": range_config,
    }

    return pico_config


# Data fetch loop
def pico_read_magnitude(chandle, status, pico_config):

    # Filter parameters
    fs = 1e9 / pico_config['trigger_config']['time_interval']
    lowcut = 1e6  # xx MHz
    highcut = 2e6  # xx MHz

    cmaxSamples = ctypes.c_int32(pico_config['trigger_config']['maxSamples'])
    overflow = ctypes.c_int16()

    status["runBlock"] = ps.ps5000aRunBlock(chandle, pico_config['trigger_config']['preTriggerSamples'], 
                                            pico_config['trigger_config']['postTriggerSamples'], 
                                            pico_config['trigger_config']['timebase'], None, 0, None, None)
    ready = ctypes.c_int16(0)
    while not ready.value:
        ps.ps5000aIsReady(chandle, ctypes.byref(ready))
    
    status["getValues"] = ps.ps5000aGetValues(chandle, 0, ctypes.byref(cmaxSamples), 0, 0, 0, ctypes.byref(overflow))
    
    chA_mv = adc2mV(pico_config['buffer_config']['bufferAMax'], pico_config['range_config']['chARange'], pico_config['maxADC'])
    chC_mv = adc2mV(pico_config['buffer_config']['bufferCMax'], pico_config['range_config']['chCRange'], pico_config['maxADC'])

    chA_filtered = butter_bandpass_filter(chA_mv, lowcut, highcut, fs)
    chC_filtered = butter_bandpass_filter(chC_mv, lowcut, highcut, fs)

    chA_PeakToPeak = np.max(chA_filtered) - np.min(chA_filtered)
    chC_PeakToPeak = np.max(chC_filtered) - np.min(chC_filtered)

    return chA_PeakToPeak, chC_PeakToPeak

def pico_read_magnitude_avg(chandle, status, pico_config, num_samples=10):

    chA_avg = 0.0
    chC_avg = 0.0

    for _ in range(num_samples):
        chA_PeakToPeak, chC_PeakToPeak = pico_read_magnitude(chandle, status, pico_config)
        chA_avg += chA_PeakToPeak
        chC_avg += chC_PeakToPeak

    chA_avg /= num_samples
    chC_avg /= num_samples

    return chA_avg, chC_avg


def pico_close(chandle, status):

    # Stop the scope
    # handle = chandle
    status["stop"] = ps.ps5000aStop(chandle)
    assert_pico_ok(status["stop"])

    # Close unit Disconnect the scope
    # handle = chandle
    status["close"]=ps.ps5000aCloseUnit(chandle)
    assert_pico_ok(status["close"])

    # display status returns
    # print(status)
#!/usr/bin/env python3
"""
fn_ctrl.py - Function generator control module for volumetric pressure field measurements
Controls JDS6600 function generator to pause/resume waveform during printer movements
"""

import time
from jds.jds6600 import jds6600

class FunctionGeneratorController:
    """Simple controller for JDS6600 function generator"""
    
    def __init__(self, port="/dev/cu.usbserial-0001", channel=1):
        """
        Initialize function generator controller
        
        Args:
            port: Serial port for JDS6600 (default: /dev/cu.usbserial-0001)
            channel: Channel to use (1 or 2, default: 1)
        """
        self.port = port
        self.channel = channel
        self.fg = None
        self._is_connected = False
        
    def connect(self):
        """Connect to the function generator"""
        try:
            self.fg = jds6600(self.port)
            self._is_connected = True
            print(f"Connected to JDS6600 on {self.port}")
            
            # Get device info
            device_type = self.fg.getinfo_devicetype()
            serial_num = self.fg.getinfo_serialnumber()
            print(f"Device type: {device_type}, Serial: {serial_num}")
            
        except Exception as e:
            print(f"Failed to connect to function generator: {e}")
            self._is_connected = False
            raise
            
    def disconnect(self):
        """Disconnect from the function generator"""
        if self.fg and self.fg.ser:
            self.fg.ser.close()
            self._is_connected = False
            print("Disconnected from function generator")
            
    def setup_waveform(self, frequency=500000, amplitude=0.35, duty_cycle=33, waveform="SINE"):
        """
        Set up initial waveform parameters
        
        Args:
            frequency: Frequency in Hz (default: 500kHz)
            amplitude: Amplitude in Volts (default: 0.35V)
            duty_cycle: Duty cycle in percent (default: 33%)
            waveform: Waveform type (default: "SINE")
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to function generator")
            
        try:
            # Set waveform type
            self.fg.setwaveform(self.channel, waveform)
            
            # Set frequency (500 kHz)
            self.fg.setfrequency(self.channel, frequency)
            
            # Set amplitude (0.35V)
            self.fg.setamplitude(self.channel, amplitude)
            
            # Set duty cycle (33%)
            self.fg.setdutycycle(self.channel, duty_cycle)
            
            # Set offset to 0V
            self.fg.setoffset(self.channel, 0.0)
            
            print(f"Waveform configured:")
            print(f"  Type: {waveform}")
            print(f"  Frequency: {frequency/1000:.1f} kHz")
            print(f"  Amplitude: {amplitude} V")
            print(f"  Duty cycle: {duty_cycle}%")
            
        except Exception as e:
            print(f"Error setting waveform parameters: {e}")
            raise
            
    def pause_output(self):
        """Pause waveform generation (disable channel output)"""
        if not self._is_connected:
            raise RuntimeError("Not connected to function generator")
            
        try:
            # Disable the channel
            if self.channel == 1:
                self.fg.setchannelenable(False, True)  # Disable ch1, keep ch2 as is
            else:
                self.fg.setchannelenable(True, False)  # Keep ch1 as is, disable ch2
                
            # Small delay to ensure command is processed
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Error pausing output: {e}")
            raise
            
    def resume_output(self):
        """Resume waveform generation (enable channel output)"""
        if not self._is_connected:
            raise RuntimeError("Not connected to function generator")
            
        try:
            # Enable the channel
            if self.channel == 1:
                self.fg.setchannelenable(True, False)  # Enable ch1, disable ch2
            else:
                self.fg.setchannelenable(False, True)  # Disable ch1, enable ch2
                
            # Small delay to ensure command is processed
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Error resuming output: {e}")
            raise
            
    def get_status(self):
        """Get current status of the function generator"""
        if not self._is_connected:
            return {"connected": False}
            
        try:
            ch1_enabled, ch2_enabled = self.fg.getchannelenable()
            freq = self.fg.getfrequency(self.channel)
            ampl = self.fg.getamplitude(self.channel)
            duty = self.fg.getdutycycle(self.channel)
            wave_id, wave_name = self.fg.getwaveform(self.channel)
            
            return {
                "connected": True,
                "channel": self.channel,
                "enabled": ch1_enabled if self.channel == 1 else ch2_enabled,
                "waveform": wave_name,
                "frequency_hz": freq,
                "amplitude_v": ampl,
                "duty_cycle_pct": duty
            }
            
        except Exception as e:
            print(f"Error getting status: {e}")
            return {"connected": True, "error": str(e)}


# Convenience functions for simple usage
_controller = None

def init(port="/dev/cu.usbserial-0001", channel=1):
    """Initialize the function generator with default settings"""
    global _controller
    _controller = FunctionGeneratorController(port, channel)
    _controller.connect()
    _controller.setup_waveform()  # Use default parameters
    return _controller

def pause():
    """Pause waveform output"""
    if _controller:
        _controller.pause_output()
    else:
        raise RuntimeError("Function generator not initialized. Call init() first.")

def resume():
    """Resume waveform output"""
    if _controller:
        _controller.resume_output()
    else:
        raise RuntimeError("Function generator not initialized. Call init() first.")

def cleanup():
    """Clean up and disconnect"""
    global _controller
    if _controller:
        _controller.disconnect()
        _controller = None


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    print("=== JDS6600 Function Generator Control Test ===")
    
    # Get port from command line or use default
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbserial-0001"
    
    try:
        # Initialize
        print(f"\nConnecting to {port}...")
        fg = init(port)
        
        # Show status
        print("\nCurrent status:")
        status = fg.get_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Test pause/resume
        print("\n--- Testing pause/resume ---")
        
        print("Output is ON")
        time.sleep(2)
        
        print("Pausing output...")
        pause()
        print("Output is PAUSED")
        time.sleep(2)
        
        print("Resuming output...")
        resume()
        print("Output is ON")
        time.sleep(2)
        
        print("\nTest complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        cleanup()
        print("Disconnected.")

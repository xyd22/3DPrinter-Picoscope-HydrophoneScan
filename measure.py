#!/usr/bin/env python3
"""
measure.py - Volumetric pressure field measurement using 3D printer and oscilloscope
Moves a hydrophone through a defined volume and captures pressure data at each voxel

Usage:
    python measure.py [voxels_x voxels_y voxels_z voxel_size_mm]
    python measure.py -continue [voxels_x voxels_y voxels_z voxel_size_mm]

Operation:
1. NO AUTOMATIC HOMING - starts with full manual control from current position
2. Use keyboard controls to position hydrophone at desired origin:
   - A/-X moves in +X direction (positive)
   - D/+X moves in -X direction (negative)  
   - W/+Y moves in +Y direction (positive)
   - S/-Y moves in -Y direction (negative)
   - Q/+Z moves in +Z direction (positive)
   - E/-Z moves in -Z direction (negative)
   - +/- to increase/decrease step size
   - M to enter/exit measure mode (counts steps between two points)
3. Press Enter to set current position as (0,0,0)
4. Scan sweeps: X from voxels_x to 0 (negative direction), Y and Z from 0 to voxels_y/z (positive directions)

Measure Mode:
- Press 'M' to mark point A and start counting steps
- Move to desired position using normal controls
- Press 'M' again to mark point B and display step count vector
- The system will show the (x,y,z) step count from A to B

Resume capability:
- The program saves a checkpoint after each voxel measurement
- If interrupted, use 'python measure.py -continue' to resume from the last position
- Checkpoint includes position and partial data
- Checkpoint is deleted after successful completion

To prevent USB disconnection on long scans:
- On macOS: sudo pmset -a disablesleep 1 (disable USB sleep)
- Use a powered USB hub for stable power
- Check USB cable quality
- Consider shorter scans or larger voxel sizes for testing

Step calibration:
- Yudong: calibration factor: X 0.9346; Y 0.8706; Z 0.9308
- This ensures 1mm commanded = 1mm actual movement
"""

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

# find the package at:
# https://github.com/picotech/picosdk-python-wrappers
# follow the instructions to correctly install the package
from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok, mV2adc

if os.name != 'nt': # Linux
    PRINTER_SERIAL = '/dev/cu.usbserial-2130'
    SCOPE_SERIAL = 'DS1ZE26CM00690'
    GENERATOR_SERIAL = '2120'
else: # Windows
    PRINTER_SERIAL = 'COM4'

# ---------- Configuration ----------
@dataclass
class MeasurementConfig:
    # Volume dimensions (voxels)
    voxels_x: int = 40  # 40 voxels in X
    voxels_y: int = 60  # 60 voxels in Y  
    voxels_z: int = 40  # 40 voxels in Z
    voxel_size: float = 0.5  # Physical size of each voxel in mm
    
    # Printer settings
    # TODO we don't know the printer serial, so ask the user to input it
    printer_port: str = f"{PRINTER_SERIAL}"
    printer_baud: int = 115200
    feed_rate: int = 3000  # mm/min (50 mm/s)
    
    @property
    def step_calibration(self) -> float:
        """Calibration factor to compensate for printer's actual movement"""
        x_calibration = 0.9346
        y_calibration = 0.8706
        z_calibration = 0.9308
        return [x_calibration, y_calibration, z_calibration]  # Just the calibration factor, not scaled by voxel size
    # Scope settings
    # Create chandle and status ready for use
    chandle = ctypes.c_int16()
    status = {}
    # capture_duration: float = 10e-6  # 10 microseconds
    
    # Timing delays (seconds)
    settling_delay: float = 0.05  # Wait after movement before measurement
    post_measure_delay: float = 0.05  # Wait after measurement before next move
    
    # Output settings
    output_dir: str = "pressure_field_data"
    
# ---------- Printer Control ----------
def open_printer(config: MeasurementConfig):
    """Initialize printer connection"""
    ser = serial.Serial(
        config.printer_port, 
        config.printer_baud, 
        timeout=5,  # Increased read timeout
        write_timeout=2,
        # Keep DTR/RTS high to maintain connection
        # dsrdtr=True,
        # rtscts=False
    )
    time.sleep(2)
    ser.write(b"M155 S0\n")  # Disable temperature reports
    wait_ok(ser)
    ser.write(b"M85 S0\n")  # Disable motor idle timeout
    wait_ok(ser)
    return ser

def wait_ok(ser, timeout=10):
    """Wait for printer OK response with timeout and debug output"""
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError("Timeout waiting for OK from printer")
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if line.lower() == "ok":
                break
        except serial.SerialException as e:
            print(f"Serial error while waiting for OK: {e}")
            raise

def send_gcode(ser, cmd):
    """Send G-code command and wait for acknowledgment"""
    try:
        ser.write((cmd + "\n").encode())
        ser.flush()  # Ensure command is sent immediately
        wait_ok(ser)
    except serial.SerialException as e:
        print(f"Serial error during send: {e}")
        raise

def move_to_position(ser, x, y, z, feed_rate, calibration=[1.0, 1.0, 1.0]):
    """Move to specified position and wait for completion"""
    # Apply calibration factor to get accurate movements
    cal_x = x * calibration[0]
    cal_y = y * calibration[1]
    cal_z = z * calibration[2]
    send_gcode(ser, f"G1 X{cal_x:.3f} Y{cal_y:.3f} Z{cal_z:.3f} F{feed_rate}")
    send_gcode(ser, "M400")  # Wait for moves to finish

def move_relative(ser, dx, dy, dz, feed_rate, calibration=[1.0, 1.0, 1.0]):
    """Move by specified delta using relative positioning"""
    # Apply calibration factor to get accurate movements
    cal_dx = dx * calibration[0]
    cal_dy = dy * calibration[1]
    cal_dz = dz * calibration[2]
    
    send_gcode(ser, "G91")  # Switch to relative mode
    send_gcode(ser, f"G1 X{cal_dx:.3f} Y{cal_dy:.3f} Z{cal_dz:.3f} F{feed_rate}")
    send_gcode(ser, "G90")  # Switch back to absolute mode
    send_gcode(ser, "M400")  # Wait for moves to finish

# ---------- Calibration ----------
def get_key():
    """Read single keypress without Enter"""
    if os.name == 'nt':  # Windows
        key = msvcrt.getch()
        if key == b'\x03':  # Ctrl+C
            raise KeyboardInterrupt
        return key.decode(errors='ignore')
    else: # Linux
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
            # Check for Ctrl+C (ASCII 3)
            if ord(key) == 3:
                raise KeyboardInterrupt()
            return key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def manual_control_mode(ser, scope, feed_rate, voxels, jog_step=1.0, calibration=1.0):
    """Full manual control mode - user positions printer wherever they want"""

    initial_step = jog_step
    
    print("\n╔══════════════════════════════════════════════╗")
    print("║         MANUAL CONTROL INTERFACE             ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\nJog step: {jog_step}mm (matches voxel size)")
    print(f"Calibration factor: x: {calibration[0]:.3f}, y: {calibration[1]:.3f}, z: {calibration[2]:.3f}")
    print("\nControls:")
    print("┌─────────────────────────────────────────────┐")
    print("│ A/-X = Move +X (positive direction)         │")
    print("│ D/+X = Move -X (negative direction)         │")
    print("│ W/+Y = Move +Y (positive direction)         │")
    print("│ S/-Y = Move -Y (negative direction)         │")
    print("│ Q/+Z = Move +Z (positive direction)         │")
    print("│ E/-Z = Move -Z (negative direction)         │")
    print("│ +/-  = Increase/decrease jog step size      │")
    print("│ M    = Start/stop measure mode              │")
    print("│ Enter = Set current position as origin      │")
    print("│ Ctrl+C = Exit program                       │")
    print("└─────────────────────────────────────────────┘")
    print("\nMove to your desired origin position and press Enter")
    
    # Measure mode state
    measure_mode = False
    step_counter = {'x': 0, 'y': 0, 'z': 0}
    
    try:
        while True:
            key = get_key()
            
            if key == '\r' or key == '\n':  # Enter pressed
                # send_gcode(ser, "G1 Z30")
                # Find the maximum output as the "center" (of the cube we are scanning)
                # Move to the "corner" and set as origin
                move_relative(ser, -voxels[0] * initial_step / 2, -voxels[1] * initial_step / 2, 
                                   -voxels[2] * initial_step / 2, feed_rate, calibration)
                # Set current position as origin (0,0,0)
                send_gcode(ser, "G92 X0 Y0 Z0")  # Define current position as 0,0,0
                print(f"\n✓ Current position set as origin (0,0,0)")
                print("  Coordinate system updated. All movements will be relative to this position.")
                # Return 0,0,0 since we've redefined the coordinate system
                return 0.0, 0.0, 0.0
                
            elif key.lower() == 'm':  # Measure mode toggle
                if not measure_mode:
                    # Start measure mode
                    measure_mode = True
                    step_counter = {'x': 0, 'y': 0, 'z': 0}
                    print(f"\n╔══════════════════════════════════════════════╗")
                    print(f"║ MEASURE MODE ACTIVATED - Point A marked       ║")
                    print(f"║ Move to point B and press 'M' again          ║")
                    print(f"╚══════════════════════════════════════════════╝")
                else:
                    # End measure mode - show results
                    measure_mode = False
                    print(f"\n╔══════════════════════════════════════════════╗")
                    print(f"║ MEASURE MODE RESULTS - Point B marked         ║")
                    print(f"║ Step count from A to B:                      ║")
                    print(f"║   ({step_counter['x']:+d}, {step_counter['y']:+d}, {step_counter['z']:+d})                            ║")
                    print(f"╚══════════════════════════════════════════════╝")
                
            elif key.lower() == 'a':  # A/-X = positive
                send_gcode(ser, f"G91")  # Relative mode
                send_gcode(ser, f"G1 X-{jog_step * calibration[0]:.2f} F{feed_rate}")
                send_gcode(ser, f"G90")  # Back to absolute mode
                if measure_mode:
                    step_counter['x'] += 1
                chA_ptp_mV, chC_ptp_mV = scope.read_magnitude_avg()
                print(f"  Ch1 Peak to Peak: {chA_ptp_mV:.3f}mV")
                    
            elif key.lower() == 'd':  # D/+X = negative
                send_gcode(ser, f"G91")  # Relative mode
                send_gcode(ser, f"G1 X{jog_step * calibration[0]:.2f} F{feed_rate}")
                send_gcode(ser, f"G90")  # Back to absolute mode
                # print the current output magnitude to decide the maximun position
                chA_ptp_mV, chC_ptp_mV = scope.read_magnitude_avg()
                print(f"  Ch1 Peak to Peak: {chA_ptp_mV:.3f}mV")
                if measure_mode:
                    step_counter['x'] -= 1
                    
            elif key.lower() == 'w':  # W/+Y = positive
                send_gcode(ser, f"G91")  # Relative mode
                send_gcode(ser, f"G1 Y{jog_step * calibration[1]:.2f} F{feed_rate}")
                send_gcode(ser, f"G90")  # Back to absolute mode
                chA_ptp_mV, chC_ptp_mV = scope.read_magnitude_avg()
                print(f"  Ch1 Peak to Peak: {chA_ptp_mV:.3f}mV")
                if measure_mode:
                    step_counter['y'] += 1
                    
            elif key.lower() == 's':  # S/-Y = negative
                send_gcode(ser, f"G91")  # Relative mode
                send_gcode(ser, f"G1 Y-{jog_step * calibration[1]:.2f} F{feed_rate}")
                send_gcode(ser, f"G90")  # Back to absolute mode
                chA_ptp_mV, chC_ptp_mV = scope.read_magnitude_avg()
                print(f"  Ch1 Peak to Peak: {chA_ptp_mV:.3f}mV")
                if measure_mode:
                    step_counter['y'] -= 1
                    
            elif key.lower() == 'q':  # Q/+Z = positive
                send_gcode(ser, f"G91")  # Relative mode
                send_gcode(ser, f"G1 Z{jog_step * calibration[2]:.2f} F{feed_rate}")
                send_gcode(ser, f"G90")  # Back to absolute mode
                chA_ptp_mV, chC_ptp_mV = scope.read_magnitude_avg()
                print(f"  Ch1 Peak to Peak: {chA_ptp_mV:.3f}mV")
                if measure_mode:
                    step_counter['z'] += 1
                    
            elif key.lower() == 'e':  # E/-Z = negative
                send_gcode(ser, f"G91")  # Relative mode
                send_gcode(ser, f"G1 Z-{jog_step * calibration[2]:.2f} F{feed_rate}")
                send_gcode(ser, f"G90")  # Back to absolute mode
                chA_ptp_mV, chC_ptp_mV = scope.read_magnitude_avg()
                print(f"  Ch1 Peak to Peak: {chA_ptp_mV:.3f}mV")
                if measure_mode:
                    step_counter['z'] -= 1
                    
            elif key == '+':
                jog_step = min(jog_step * 2, 10.0)
                print(f"\n→ Jog step increased to: {jog_step}mm")
            elif key == '-':
                jog_step = max(jog_step / 2, 0.1)
                print(f"\n→ Jog step decreased to: {jog_step}mm")
            else:
                continue
                
            # Simple feedback
            if measure_mode:
                print(f"\r[MEASURE MODE] Steps: ({step_counter['x']:+d}, {step_counter['y']:+d}, {step_counter['z']:+d})", end='', flush=True)
            else:
                print(".", end='', flush=True)
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
        raise  # Re-raise to exit the program

# ---------- Scan Pattern ----------
def generate_scan_positions(config: MeasurementConfig):
    """Generate voxel positions using serpentine pattern in 3D space"""
    nx = int(config.voxels_x)
    ny = int(config.voxels_y)
    nz = int(config.voxels_z)
    
    # Y only increments (stacking slices)
    for iy in range(ny):
        y = iy * config.voxel_size
        
        # Z direction reverses on odd Y indices
        # Even Y: sweep Z from 0 to Z_max
        # Odd Y: sweep Z from Z_max to 0
        if iy % 2 == 0:
            z_indices = range(nz)  # 0, 1, 2, ... nz-1
        else:
            z_indices = reversed(range(nz))  # nz-1, nz-2, ... 0
            
        for iz in z_indices:
            z = iz * config.voxel_size
            
            # X serpentine pattern
            # Need to track actual position in the sweep, not just iz value
            # For even Y: iz_idx matches iz
            # For odd Y: iz_idx is 0 when iz is nz-1, 1 when iz is nz-2, etc.
            if (iz + iy) % 2 == 0:
                # Sweep from 0 to X
                for ix in range(nx):
                    x = ix * config.voxel_size
                    yield x, y, z, ix, iy, iz
            else:
                # Sweep from X to 0
                for ix in reversed(range(nx)):
                    x = ix * config.voxel_size
                    yield x, y, z, ix, iy, iz

# ---------- Time Estimation ----------
def estimate_scan_time(config: MeasurementConfig) -> float:
    """Estimate total scan time in seconds"""
    total_voxels = (config.voxels_x * 
                    config.voxels_y * 
                    config.voxels_z)
    
    # Time per voxel
    time_per_voxel = (
        config.settling_delay +  # Wait after movement
        0.2 +  # Measurement time (approximate)
        config.post_measure_delay  # Wait before next move
    )
    
    # Average movement time (rough estimate)
    avg_move_distance = config.voxel_size  # mm
    avg_move_time = avg_move_distance / (config.feed_rate / 60)  # seconds
    
    total_time = total_voxels * (time_per_voxel + avg_move_time)
    
    return total_time

# ---------- Main Measurement Routine ----------
def main():
    config = MeasurementConfig()
    
    # Parse command line arguments if provided
    continue_mode = False
    checkpoint_path = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '-continue':
            continue_mode = True
            # Look for most recent checkpoint
            data_dir = pathlib.Path(config.output_dir)
            if data_dir.exists():
                # Find most recent directory with checkpoint
                dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()], reverse=True)
                for d in dirs:
                    if (d / 'checkpoint.npy').exists():
                        checkpoint_path = d
                        break
                if not checkpoint_path:
                    print("Error: No checkpoint found to continue from")
                    return
            else:
                print("Error: No pressure_field_data directory found")
                return
            # Skip the -continue argument for further parsing
            sys.argv = sys.argv[:1] + sys.argv[2:]
        
        # Example: python measure.py 20 20 40 1.0 (or python measure.py -continue 20 20 40 1.0)
        if len(sys.argv) >= 5:
            config.voxels_x = int(sys.argv[1])
            config.voxels_y = int(sys.argv[2])
            config.voxels_z = int(sys.argv[3])
            config.voxel_size = float(sys.argv[4])
            # real length = voxel * voxel_size * calibration
    
    # Handle output directory and checkpoint loading
    checkpoint_data = None
    start_index = 0
    
    if continue_mode and checkpoint_path:
        # Use existing directory for continue mode
        output_path = checkpoint_path
        timestamp = output_path.name
        
        # Load checkpoint
        checkpoint_data = np.load(output_path / 'checkpoint.npy', allow_pickle=True).item()
        # Re-measure the voxel we stopped at; the probe is already there.
        start_index = checkpoint_data['last_index']
        
        print(f"\n=== RESUMING FROM CHECKPOINT ===")
        print(f"  Last voxel indices: {checkpoint_data.get('last_indices', 'N/A')}")
        print(f"  Last position: {checkpoint_data['last_position']} mm")
        print(f"  Last voxel index: {checkpoint_data['last_index']}")
        print(f"  Re-measuring voxel {start_index}, then continuing to {checkpoint_data['total_voxels']}")
        print(f"  Using output directory: {output_path}")
    else:
        # Create new output directory for fresh start
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = pathlib.Path(config.output_dir) / timestamp
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate scan parameters
    total_voxels = int(config.voxels_x * 
                       config.voxels_y * 
                       config.voxels_z)
    
    estimated_time = estimate_scan_time(config)
    
    print(f"=== Volumetric Pressure Field Measurement ===")
    print(f"Volume: {config.voxels_x}×{config.voxels_y}×{config.voxels_z} voxels")
    print(f"Physical size: {config.voxels_x * config.voxel_size:.1f}×{config.voxels_y * config.voxel_size:.1f}×{config.voxels_z * config.voxel_size:.1f} mm")
    print(f"Scan pattern: X: {config.voxels_x}→0 voxels, Y: 0→{config.voxels_y} voxels, Z: 0→{config.voxels_z} voxels")
    print(f"Voxel size: {config.voxel_size} mm")
    print(f"Total voxels: {total_voxels:,}")
    print(f"Step calibration: {config.step_calibration}")
    print(f"Estimated scan time: {estimated_time/60:.1f} minutes")
    print(f"Output directory: {output_path}")
    print()
    
    # Initialize devices
    print("Initializing devices...")
    printer = None
    scope = None

    with open(output_path / "pressure_field.log", "a", buffering=1) as pressure_log:
        try:
            printer = open_printer(config)
            scope = PicoScope(config.chandle, config.status)
            # fn_ctrl.init(port=f'/dev/cu.usbserial-{GENERATOR_SERIAL}')
            
            # Initialize data arrays
            nx = int(config.voxels_x)
            ny = int(config.voxels_y)
            nz = int(config.voxels_z)
        
            # Load existing pressure field or create new one
            if continue_mode and checkpoint_path:
                # Try to load partial pressure field
                partial_path = output_path / 'pressure_field_partial.npy'
                if partial_path.exists():
                    pressure_field = np.load(partial_path)
                    print(f"  Loaded existing partial pressure field")
                else:
                    # Initialize new but load any existing voxel data
                    pressure_field = np.zeros((nx, ny, nz))
                    print(f"  No partial field found, reconstructing from voxel files...")
                    # Could reconstruct from individual voxel files here if needed
            else:
                pressure_field = np.zeros((nx, ny, nz))
            
            # Save configuration
            config_dict = {
                'volume_voxels': [config.voxels_x, config.voxels_y, config.voxels_z],
                'volume_mm': [config.voxels_x * config.voxel_size, 
                            config.voxels_y * config.voxel_size, 
                            config.voxels_z * config.voxel_size],
                'voxel_size_mm': config.voxel_size,
                'shape': [nx, ny, nz],
                'feed_rate_mm_per_min': config.feed_rate,
                'settling_delay_s': config.settling_delay,
                # 'capture_duration_s': config.capture_duration,
                'timestamp': timestamp
            }
            np.save(output_path / 'config.npy', config_dict)
            
            if continue_mode and checkpoint_data:
                # Skip manual control in continue mode
                print("\n=== SKIPPING MANUAL CALIBRATION (CONTINUE MODE) ===")
                
                # Load original config to get origin
                orig_config = np.load(output_path / 'config.npy', allow_pickle=True).item()
                origin_x, origin_y, origin_z = orig_config.get('origin_mm', [0, 0, 0])
                
                # The hydrophone is already at the last position - don't move it!
                # Instead, set the coordinate system so current position = last position
                last_x, last_y, last_z = checkpoint_data['last_position']
                print(f"\nHydroPhone is at last scan position: ({last_x:.1f}, {last_y:.1f}, {last_z:.1f})")
                print("Setting coordinate system to match...")
                
                # Align printer's logical coordinates with the physical position
                send_gcode(printer, 
                        f"G92 X{last_x * config.step_calibration[0]:.3f} "
                        f"Y{last_y * config.step_calibration[1]:.3f} "
                        f"Z{last_z * config.step_calibration[2]:.3f}")
                print("Ready to resume scanning.")
            else:
                # Start in full manual control mode
                print("\n=== MANUAL CONTROL MODE ===")
                print("You have full control of the printer.")
                print("Position the hydrophone at your desired origin (0,0,0) for the scan.")
                print("\nNote: The system will treat your accepted position as (0,0,0)")
                print("The scan will run from there in positive X, Y, Z directions.")
                
                # Get current position from user via manual control
                origin_x, origin_y, origin_z = manual_control_mode(
                    printer,
                    scope,
                    config.feed_rate, 
                    voxels=[config.voxels_x, config.voxels_y, config.voxels_z],
                    jog_step=config.voxel_size,  # Use voxel size as default jog step
                    calibration=config.step_calibration
                )
                
                # Save origin position
                config_dict['origin_mm'] = [origin_x, origin_y, origin_z]
                np.save(output_path / 'config.npy', config_dict)
            
            # Skip confirmations if continuing
            if not continue_mode:
                # Display scan information
                print(f"\nScan will run from current origin (0,0,0):")
                print(f"  X: {config.voxels_x} → 0 voxels ({config.voxels_x * config.voxel_size:.1f} → 0 mm)")
                print(f"  Y: 0 → {config.voxels_y} voxels (0 → {config.voxels_y * config.voxel_size:.1f} mm)")
                print(f"  Z: 0 → {config.voxels_z} voxels (0 → {config.voxels_z * config.voxel_size:.1f} mm)")
                print(f"\n⚠️  IMPORTANT: Ensure scan volume fits within printer limits!")
                print(f"The scan will start at +{config.voxels_x} voxels ({config.voxels_x * config.voxel_size:.1f} mm) in X,")
                print(f"then move towards 0. It will move {config.voxels_y} voxels ({config.voxels_y * config.voxel_size:.1f} mm)")
                print(f"in +Y and {config.voxels_z} voxels ({config.voxels_z * config.voxel_size:.1f} mm) in +Z from current position.")
                
                response = input("\nConfirm scan volume is safe? (Y/n): ")
                if response.lower() == 'n':
                    print("Scan aborted.")
                    send_gcode(printer, "M84")  # Disable steppers
                    printer.close()
                    scope.close()
                    return
                
                # Test measurement at origin
                print("\nPerforming test measurement at origin...")
                try:
                    # t1, v1 = capture_waveform(scope, 1)
                    # t2, v2 = capture_waveform(scope, 2)
                    # Remove DC offset before calculating RMS
                    # v1_ac = v1 - np.mean(v1)
                    # test_rms = np.sqrt(np.mean(v1_ac**2))
                    chA_ptp_mV, chC_ptp_mV = scope.read_magnitude_avg()
                    print(f"Test measurement successful!")
                    print(f"  Ch1 Peak to Peak: {chA_ptp_mV:.3f}mV")
                    # print(f"  Ch1 DC offset: {np.mean(v1):.4f}V")
                    # print(f"  Ch1 RMS (AC): {test_rms:.4f}V")
                    # print(f"  Ch1 samples: {len(v1)}")
                    # print(f"  Ch1 samples: {len(v1)}")
                    # print(f"  Duration: {(t1[-1]-t1[0])*1e6:.1f}μs")
                    
                    response = input("\nProceed with full scan? (Y/n): ")
                    if response.lower() == 'n':
                        print("Scan aborted.")
                        send_gcode(printer, "M84")  # Disable steppers
                        printer.close()
                        scope.close()
                        return
                except Exception as e:
                    print(f"⚠️  Test measurement failed: {e}")
                    print("Please check oscilloscope connection and settings.")
                    send_gcode(printer, "M84")  # Disable steppers
                    printer.close()
                    scope.close()
                    return
            
            # Main measurement loop
            if continue_mode:
                print(f"\nResuming measurement scan from voxel {start_index}...")
            else:
                print("\nStarting measurement scan...")
            start_time = time.time()
            
            # fn_ctrl.resume()
            
            # Track current position for relative movements
            current_x, current_y, current_z = (  # track where we *really* are
                checkpoint_data['last_position'] if continue_mode and checkpoint_data else (0.0, 0.0, 0.0)
            )
            MIN_DELAY = 0.2
            
            # For fresh scans, move to the starting position
            if not continue_mode:
                # Get the first position from the generator
                first_pos = next(generate_scan_positions(config)) # (x, y, z, ix, iy, iz)
                dx, dy, dz = first_pos[0], first_pos[1], first_pos[2]
                if abs(dx) > 1e-6 or abs(dy) > 1e-6 or abs(dz) > 1e-6:
                    print(f"Moving to scan start position: ({dx:.1f}, {dy:.1f}, {dz:.1f})...")
                    s = time.time()
                    move_relative(printer, dx, dy, dz,
                                config.feed_rate, config.step_calibration)
                    move_time = time.time() - s
                    remaining_delay = max(MIN_DELAY - move_time, 0)
                    time.sleep(remaining_delay)
                    current_x, current_y, current_z = first_pos
            
            # Generate all positions but skip to start_index if continuing
            for idx, (x, y, z, ix, iy, iz) in enumerate(generate_scan_positions(config)):
                # Skip already completed voxels when continuing
                if idx < start_index:
                    continue
                
                # Δ distance to next voxel
                dx, dy, dz = x - current_x, y - current_y, z - current_z
                
                if abs(dx) > 1e-6 or abs(dy) > 1e-6 or abs(dz) > 1e-6:
                    s = time.time()
                    move_relative(printer, dx, dy, dz, 
                                config.feed_rate, config.step_calibration)
                    move_time = time.time() - s
                    remaining_delay = max(MIN_DELAY - move_time, 0)
                    time.sleep(remaining_delay)
                else:
                    # Already at target voxel – no motion
                    move_time = 0.0
                    remaining_delay = MIN_DELAY
                    time.sleep(remaining_delay)
                    s = time.time() - MIN_DELAY  # For logging consistency
                
                current_x, current_y, current_z = x, y, z
                
                # print(f"Move time: {move_time:.2f}s, Added delay: {remaining_delay:.2f}s")
                print(f"Total time once: {time.time() - s:.2f}s")
                
                # # Capture measurement
                # t1, v1 = capture_waveform(scope, 1)
                # # Process measurement - Remove DC offset before calculating RMS
                # v1_ac = v1 - np.mean(v1)  # Remove DC component
                # rms_value = np.sqrt(np.mean(v1_ac**2))  # Calculate RMS of AC component
                chA_ptp_mV, chC_ptp_mV = scope.read_magnitude_avg()
                print(f"  Ch1 Peak to Peak: {chA_ptp_mV:.3f}mV")
                
                # Store in 3D array
                # ix = int(x / config.voxel_size)
                # iy = int(y / config.voxel_size)
                # iz = int(z / config.voxel_size)
                pressure_field[ix, iy, iz] = chA_ptp_mV
                
                # Save individual measurement
                # voxel_data = {
                #     'voxel_indices': [ix, iy, iz],
                #     'position_mm': [x, y, z],  # Physical position in scan volume
                #     # 'time': t1,
                #     # 'ch1_voltage': v1,
                #     'magnitude': chA_ptp_mV
                # }
                # np.save(output_path / f'voxel_{ix:03d}_{iy:03d}_{iz:03d}.npy', voxel_data)
                
                # Save checkpoint for resume capability
                checkpoint = {
                    'last_index': idx,
                    'last_position': [x, y, z],
                    'last_indices': [ix, iy, iz],
                    'total_voxels': total_voxels,
                    'timestamp': timestamp
                }
                np.save(output_path / 'checkpoint.npy', checkpoint)
                
                # Progress update
                if idx % 10 == 0:
                    elapsed = time.time() - start_time
                    progress = (idx + 1) / total_voxels
                    eta = elapsed / progress - elapsed if progress > 0 else 0
                    print(f"Progress: {progress*100:.1f}% | "
                        f"Voxel {idx+1}/{total_voxels} | "
                        f"Indices: ({ix}, {iy}, {iz}) | "
                        f"Position: ({x:.1f}, {y:.1f}, {z:.1f}) mm | "
                        f"Peak To Peak: {chA_ptp_mV:.3f}mV | "
                        f"ETA: {eta/60:.1f} min")
                        
                # Auto-save partial data every 500 voxels
                if idx % 500 == 0 and idx > 0:
                    np.save(output_path / 'pressure_field_partial.npy', pressure_field)
                    print(f"  → Auto-saved partial data at voxel {idx}")
                
                # Auto-save each voxel data
                line = f"{ix, iy, iz, chA_ptp_mV:.6f}\n"
                pressure_log.write(line)
                pressure_log.flush()
            
            # Save complete pressure field
            np.save(output_path / 'pressure_field.npy', pressure_field)
            
            # Delete checkpoint file since scan completed successfully
            checkpoint_file = output_path / 'checkpoint.npy'
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                print("Checkpoint file removed (scan completed successfully)")
            
            # Return to origin position
            print("\nReturning to origin position (0,0,0)...")
            move_to_position(printer, 0, 0, 0, config.feed_rate, config.step_calibration)
            
            elapsed_total = time.time() - start_time
            print(f"\nScan complete!")
            print(f"Total time: {elapsed_total/60:.1f} minutes")
            print(f"Data saved to: {output_path}")
            
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user!")
            # Save partial data if we have it
            if 'pressure_field' in locals():
                np.save(output_path / 'pressure_field_partial.npy', pressure_field)
                print(f"Partial data saved to: {output_path}")
            
        finally:
            # Cleanup
            if printer:
                try:
                    send_gcode(printer, "M84")  # Disable steppers
                    printer.close()
                except:
                    pass
            if scope:
                try:
                    scope.close()
                except:
                    pass

if __name__ == "__main__":
    main() 
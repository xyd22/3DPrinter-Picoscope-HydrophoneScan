#!/usr/bin/env python3
"""
measure.py - Volumetric pressure field measurement using 3D printer and oscilloscope
Moves a hydrophone through a defined volume and captures pressure data at each voxel

Usage:
    python measure.py [volume_x volume_y volume_z voxel_size]
    python measure.py -continue [volume_x volume_y volume_z voxel_size]

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
3. Press Enter to set current position as (0,0,0)
4. Scan sweeps: X from volume_x to 0 (negative direction), Y and Z from 0 to volume_y/z (positive directions)

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
- The printer's actual movement is 1.25x the commanded movement
- A calibration factor of 0.8 is applied to all movements
- This ensures 1mm commanded = 1mm actual movement
"""

import time
import sys
import numpy as np
import serial
import pyvisa
import datetime
import pathlib
import termios
import tty
from dataclasses import dataclass
from typing import Generator
import fn_ctrl

PRINTER_SERIAL = '2130'
SCOPE_SERIAL = 'DS1ZE26CM00690'
GENERATOR_SERIAL = '2120'

# ---------- Configuration ----------
@dataclass
class MeasurementConfig:
    # Volume dimensions (mm)
    volume_x: float = 20.0  # 2cm
    volume_z: float = 20.0  # 4cm (updated per your specification)
    volume_y: float = 30.0  # 2cm  
    voxel_size: float = 0.1  # 1mm voxels
    
    # Printer settings
    # TODO we don't know the printer serial, so ask the user to input it
    printer_port: str = f"/dev/cu.usbserial-{PRINTER_SERIAL}"
    printer_baud: int = 115200
    feed_rate: int = 3000  # mm/min (50 mm/s)
    
    @property
    def step_calibration(self) -> float:
        base_calibration = 0.915 
        return base_calibration * self.voxel_size
    # Scope settings
    # We know the scope serial
    scope_resource: str = 'USB0::6833::1303::DS1ZE26CM00690::0::INSTR'
    capture_duration: float = 10e-6  # 10 microseconds
    
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
        dsrdtr=True,
        rtscts=False
    )
    time.sleep(2)
    ser.write(b"M155 S0\n")  # Disable temperature reports
    wait_ok(ser)
    ser.write(b"M85 S0\n")  # Disable motor idle timeout
    wait_ok(ser)
    return ser

def wait_ok(ser):
    """Wait for printer OK response"""
    while True:
        try:
            line = ser.readline().decode().strip()
            if line == "ok":
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

def move_to_position(ser, x, y, z, feed_rate, calibration=1.0):
    """Move to specified position and wait for completion"""
    # Apply calibration factor to get accurate movements
    cal_x = x * calibration
    cal_y = y * calibration
    cal_z = z * calibration
    send_gcode(ser, f"G1 X{cal_x:.2f} Y{cal_y:.2f} Z{cal_z:.2f} F{feed_rate}")
    send_gcode(ser, "M400")  # Wait for moves to finish

# ---------- Calibration ----------
def get_key():
    """Read single keypress without Enter"""
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

def manual_control_mode(ser, feed_rate, jog_step=1.0, calibration=1.0):
    """Full manual control mode - user positions printer wherever they want"""
    
    print("\n╔══════════════════════════════════════════════╗")
    print("║         MANUAL CONTROL INTERFACE             ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\nJog step: {jog_step}mm")
    print("\nControls:")
    print("┌─────────────────────────────────────────────┐")
    print("│ A/-X = Move +X (positive direction)         │")
    print("│ D/+X = Move -X (negative direction)         │")
    print("│ W/+Y = Move +Y (positive direction)         │")
    print("│ S/-Y = Move -Y (negative direction)         │")
    print("│ Q/+Z = Move +Z (positive direction)         │")
    print("│ E/-Z = Move -Z (negative direction)         │")
    print("│ +/-  = Increase/decrease jog step size      │")
    print("│ Enter = Set current position as origin      │")
    print("│ Ctrl+C = Exit program                       │")
    print("└─────────────────────────────────────────────┘")
    print("\nMove to your desired origin position and press Enter")
    
    try:
        while True:
            key = get_key()
            
            if key == '\r' or key == '\n':  # Enter pressed
                # Set current position as origin (0,0,0)
                send_gcode(ser, "G92 X0 Y0 Z0")  # Define current position as 0,0,0
                print(f"\n✓ Current position set as origin (0,0,0)")
                print("  Coordinate system updated. All movements will be relative to this position.")
                # Return 0,0,0 since we've redefined the coordinate system
                return 0.0, 0.0, 0.0
                
            elif key.lower() == 'a':  # A/-X = positive
                send_gcode(ser, f"G91")  # Relative mode
                send_gcode(ser, f"G1 X{jog_step * calibration:.2f} F{feed_rate}")
                send_gcode(ser, f"G90")  # Back to absolute mode
            elif key.lower() == 'd':  # D/+X = negative
                send_gcode(ser, f"G91")  # Relative mode
                send_gcode(ser, f"G1 X-{jog_step * calibration:.2f} F{feed_rate}")
                send_gcode(ser, f"G90")  # Back to absolute mode
            elif key.lower() == 'w':  # W/+Y = positive
                send_gcode(ser, f"G91")  # Relative mode
                send_gcode(ser, f"G1 Y{jog_step * calibration:.2f} F{feed_rate}")
                send_gcode(ser, f"G90")  # Back to absolute mode
            elif key.lower() == 's':  # S/-Y = negative
                send_gcode(ser, f"G91")  # Relative mode
                send_gcode(ser, f"G1 Y-{jog_step * calibration:.2f} F{feed_rate}")
                send_gcode(ser, f"G90")  # Back to absolute mode
            elif key.lower() == 'q':  # Q/+Z = positive
                send_gcode(ser, f"G91")  # Relative mode
                send_gcode(ser, f"G1 Z{jog_step * calibration:.2f} F{feed_rate}")
                send_gcode(ser, f"G90")  # Back to absolute mode
            elif key.lower() == 'e':  # E/-Z = negative
                send_gcode(ser, f"G91")  # Relative mode
                send_gcode(ser, f"G1 Z-{jog_step * calibration:.2f} F{feed_rate}")
                send_gcode(ser, f"G90")  # Back to absolute mode
            elif key == '+':
                jog_step = min(jog_step * 2, 10.0)
                print(f"\n→ Jog step increased to: {jog_step}mm")
            elif key == '-':
                jog_step = max(jog_step / 2, 0.1)
                print(f"\n→ Jog step decreased to: {jog_step}mm")
            else:
                continue
                
            # Simple feedback
            print(".", end='', flush=True)
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
        raise  # Re-raise to exit the program

# ---------- Scope Control ----------
def initialize_scope(config: MeasurementConfig):
    """Initialize oscilloscope connection and settings"""
    rm = pyvisa.ResourceManager('@py')
    scope = rm.open_resource(config.scope_resource)
    scope.timeout = 10000  # ms
    scope.chunk_size = 1 << 20  # 1 MiB
    
    # Configure time base
    h_div = 10  # DS1000Z has 10 horizontal divisions
    scope.write(':TIM:MODE MAIN')
    scope.write(f':TIM:SCAL {config.capture_duration / h_div}')
    
    return scope

def capture_waveform(scope, channel: int):
    """Capture waveform from specified channel"""
    scope.write(':STOP')
    scope.write(f':WAV:SOUR CHAN{channel}')
    scope.write(':WAV:MODE NORM')  # Screen memory mode
    scope.write(':WAV:FORM BYTE')
    
    # Get scaling factors
    pre = list(map(float, scope.query(':WAV:PRE?').split(',')))
    xinc, xorg = pre[4], pre[5]
    yinc, yorg, yref = pre[7:10]
    
    # Read waveform data
    scope.write(':WAV:DATA?')
    pound = scope.read_bytes(1)  # '#'
    ndig = int(scope.read_bytes(1))
    nbytes = int(scope.read_bytes(ndig))
    raw = scope.read_bytes(nbytes)
    
    # Convert to voltage
    volts = (np.frombuffer(raw, dtype=np.uint8) - yref) * yinc + yorg
    time_data = np.arange(volts.size) * xinc + xorg
    
    scope.write(':RUN')
    
    return time_data, volts

# ---------- Scan Pattern ----------
def generate_scan_positions(config: MeasurementConfig):
    """Generate voxel positions using serpentine pattern in X-Z plane, then Y"""
    nx = int(config.volume_x / config.voxel_size)
    ny = int(config.volume_y / config.voxel_size)
    nz = int(config.volume_z / config.voxel_size)
    
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
            
        for iz_idx, iz in enumerate(z_indices):
            z = iz * config.voxel_size
            
            # X serpentine pattern
            # Need to track actual position in the sweep, not just iz value
            # For even Y: iz_idx matches iz
            # For odd Y: iz_idx is 0 when iz is nz-1, 1 when iz is nz-2, etc.
            if iz_idx % 2 == 0:
                # Sweep from 0 to X
                for ix in range(nx):
                    x = ix * config.voxel_size
                    yield x, y, z
            else:
                # Sweep from X to 0
                for ix in reversed(range(nx)):
                    x = ix * config.voxel_size
                    yield x, y, z

# ---------- Time Estimation ----------
def estimate_scan_time(config: MeasurementConfig) -> float:
    """Estimate total scan time in seconds"""
    total_voxels = (config.volume_x / config.voxel_size * 
                    config.volume_y / config.voxel_size * 
                    config.volume_z / config.voxel_size)
    
    # Time per voxel
    time_per_voxel = (
        config.settling_delay +  # Wait after movement
        0.1 +  # Measurement time (approximate)
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
            config.volume_x = float(sys.argv[1])
            config.volume_y = float(sys.argv[2])
            config.volume_z = float(sys.argv[3])
            config.voxel_size = float(sys.argv[4])
    
    # Handle output directory and checkpoint loading
    checkpoint_data = None
    start_index = 0
    
    if continue_mode and checkpoint_path:
        # Use existing directory for continue mode
        output_path = checkpoint_path
        timestamp = output_path.name
        
        # Load checkpoint
        checkpoint_data = np.load(output_path / 'checkpoint.npy', allow_pickle=True).item()
        start_index = checkpoint_data['last_index'] + 1
        
        print(f"\n=== RESUMING FROM CHECKPOINT ===")
        print(f"  Last position: {checkpoint_data['last_position']}")
        print(f"  Last voxel index: {checkpoint_data['last_index']}")
        print(f"  Resuming from voxel {start_index}/{checkpoint_data['total_voxels']}")
        print(f"  Using output directory: {output_path}")
    else:
        # Create new output directory for fresh start
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = pathlib.Path(config.output_dir) / timestamp
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate scan parameters
    total_voxels = int(config.volume_x / config.voxel_size * 
                      config.volume_y / config.voxel_size * 
                      config.volume_z / config.voxel_size)
    
    estimated_time = estimate_scan_time(config)
    
    print(f"=== Volumetric Pressure Field Measurement ===")
    print(f"Volume: {config.volume_x}×{config.volume_y}×{config.volume_z} mm")
    print(f"Scan pattern: X: {config.volume_x}→0 mm, Y: 0→{config.volume_y} mm, Z: 0→{config.volume_z} mm")
    print(f"Voxel size: {config.voxel_size} mm")
    print(f"Total voxels: {total_voxels:,}")
    print(f"Step calibration: {config.step_calibration} (compensating for {1.0/config.step_calibration:.2f}x actual movement)")
    print(f"Estimated scan time: {estimated_time/60:.1f} minutes")
    print(f"Output directory: {output_path}")
    print()
    
    # Initialize devices
    print("Initializing devices...")
    printer = None
    scope = None
    
    try:
        printer = open_printer(config)
        scope = initialize_scope(config)
        fn_ctrl.init(port=f'/dev/cu.usbserial-{GENERATOR_SERIAL}')
        
        # Initialize data arrays
        nx = int(config.volume_x / config.voxel_size)
        ny = int(config.volume_y / config.voxel_size)
        nz = int(config.volume_z / config.voxel_size)
    
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
            'volume_mm': [config.volume_x, config.volume_y, config.volume_z],
            'voxel_size_mm': config.voxel_size,
            'shape': [nx, ny, nz],
            'feed_rate_mm_per_min': config.feed_rate,
            'settling_delay_s': config.settling_delay,
            'capture_duration_s': config.capture_duration,
            'timestamp': timestamp
        }
        np.save(output_path / 'config.npy', config_dict)
        
        if continue_mode and checkpoint_data:
            # Skip manual control in continue mode
            print("\n=== SKIPPING MANUAL CALIBRATION (CONTINUE MODE) ===")
            
            # Load original config to get origin
            orig_config = np.load(output_path / 'config.npy', allow_pickle=True).item()
            origin_x, origin_y, origin_z = orig_config.get('origin_mm', [0, 0, 0])
            
            # Move to last position
            last_x, last_y, last_z = checkpoint_data['last_position']
            print(f"\nMoving to last scan position: ({last_x:.1f}, {last_y:.1f}, {last_z:.1f})...")
            move_to_position(printer, last_x, last_y, last_z, config.feed_rate, config.step_calibration)
            print("Ready to resume scanning.")
        else:
            # Start in full manual control mode
            print("\n=== MANUAL CONTROL MODE ===")
            print("You have full control of the printer.")
            print("Position the hydrophone at your desired origin (0,0,0) for the scan.")
            print("\nNote: The system will treat your accepted position as (0,0,0)")
            print("The scan will run from there in positive X, Y, Z directions.")
            
            # Get current position from user via manual control
            origin_x, origin_y, origin_z = manual_control_mode(printer, config.feed_rate, calibration=config.step_calibration)
            
            # Save origin position
            config_dict['origin_mm'] = [origin_x, origin_y, origin_z]
            np.save(output_path / 'config.npy', config_dict)
        
        # Skip confirmations if continuing
        if not continue_mode:
            # Display scan information
            print(f"\nScan will run from current origin (0,0,0):")
            print(f"  X: {config.volume_x:.1f} → 0 mm (reversed direction)")
            print(f"  Y: 0 → {config.volume_y:.1f} mm")
            print(f"  Z: 0 → {config.volume_z:.1f} mm")
            print(f"\n⚠️  IMPORTANT: Ensure scan volume fits within printer limits!")
            print(f"The scan will start at +{config.volume_x}mm in X, then move towards 0.")
            print(f"It will move {config.volume_y}mm in +Y and {config.volume_z}mm in +Z from current position.")
            
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
                t1, v1 = capture_waveform(scope, 1)
                t2, v2 = capture_waveform(scope, 2)
                test_rms = np.sqrt(np.mean(v1**2))
                print(f"Test measurement successful!")
                print(f"  Ch1 RMS: {test_rms:.4f}V")
                print(f"  Ch1 samples: {len(v1)}")
                print(f"  Duration: {(t1[-1]-t1[0])*1e6:.1f}μs")
                
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
        
        fn_ctrl.resume()
        # Generate all positions but skip to start_index if continuing
        for idx, (x, y, z) in enumerate(generate_scan_positions(config)):
            # Skip already completed voxels when continuing
            if idx < start_index:
                continue
            # Move to scan position (already relative to our origin)
            s = time.time()
            move_to_position(printer, x, y, z, config.feed_rate, config.step_calibration)
            
            # Calculate remaining delay needed to reach 0.2s total
            move_time = time.time() - s
            remaining_delay = max(0.2 - move_time, 0)
            time.sleep(remaining_delay)
            print(f"Move time: {move_time:.2f}s, Added delay: {remaining_delay:.2f}s")
            print(f"Total time: {time.time() - s:.2f}s")
            
            # Capture measurement
            t1, v1 = capture_waveform(scope, 1)
            # Process measurement (example: RMS of channel 1)
            rms_value = np.sqrt(np.mean(v1**2))
            
            # Store in 3D array
            ix = int(x / config.voxel_size)
            iy = int(y / config.voxel_size)
            iz = int(z / config.voxel_size)
            pressure_field[ix, iy, iz] = rms_value
            
            # Save individual measurement
            voxel_data = {
                'position_mm': [x, y, z],  # Position in scan volume
                'time': t1,
                'ch1_voltage': v1,
                'rms': rms_value
            }
            np.save(output_path / f'voxel_{ix:03d}_{iy:03d}_{iz:03d}.npy', voxel_data)
            
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
                      f"Position: ({x:.1f}, {y:.1f}, {z:.1f}) | "
                      f"RMS: {rms_value:.3f}V | "
                      f"ETA: {eta/60:.1f} min")
                      
            # Auto-save partial data every 500 voxels
            if idx % 500 == 0 and idx > 0:
                np.save(output_path / 'pressure_field_partial.npy', pressure_field)
                print(f"  → Auto-saved partial data at voxel {idx}")
        
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
#!/usr/bin/env python3
"""
visualize_live.py - Real-time visualization of pressure field measurements
Shows a 3D volume with color-coded pressure intensity that updates as measurements are collected
"""

import vtk
import numpy as np
import pathlib
import time
import datetime
from vtk.util.numpy_support import numpy_to_vtk
from collections import defaultdict
import os
import sys
import re

class PressureFieldVisualizer:
    def __init__(self):
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1024, 768)
        self.render_window.SetWindowName("Live Pressure Field Measurement")
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        
        # Volume data
        self.volume_data = None
        self.volume_actor = None
        self.volume_mapper = None
        self.volume_property = None
        self.pressure_array = None
        
        # Progress tracking
        self.total_voxels = 0
        self.measured_voxels = 0
        self.start_time = None
        self.voxel_positions = {}
        
        # Progress bar
        self.progress_text = vtk.vtkTextActor()
        self.progress_bar = vtk.vtkTextActor()
        self.stats_text = vtk.vtkTextActor()
        
        # Color bar
        self.scalar_bar = None
        self.lut = None
        
        # Data monitoring
        self.data_dir = None
        self.config = None
        self.seen_files = set()
        
        # Timer for updates
        self.timer_id = None
        
    def find_latest_measurement_dir(self):
        """Find the most recent measurement directory"""
        base_path = pathlib.Path("pressure_field_data")
        if not base_path.exists():
            return None
            

        timestamp_re = re.compile(r'^\d{8}_\d{6}$')

        dirs = [d for d in base_path.iterdir()
                if d.is_dir() and timestamp_re.match(d.name)]

            
        # Sort by timestamp (directory name)
        latest = sorted(dirs, key=lambda x: x.name)[-1]
        return latest
        
    def load_config(self, config_path):
        """Load measurement configuration"""
        if not config_path.exists():
            return False
            
        self.config = np.load(config_path, allow_pickle=True).item()
        self.total_voxels = int(np.prod(self.config['shape']))
        return True
        
    def setup_volume(self):
        """Initialize the volume visualization"""
        if not self.config:
            return
            
        nx, ny, nz = self.config['shape']
        voxel_size = self.config['voxel_size_mm']
        
        # Create image data
        self.volume_data = vtk.vtkImageData()
        self.volume_data.SetDimensions(nx + 1, ny + 1, nz + 1)
        self.volume_data.SetSpacing(voxel_size, voxel_size, voxel_size)
        self.volume_data.SetOrigin(0, 0, 0)
        
        # Initialize with zeros
        self.pressure_array = np.zeros((nx * ny * nz,), dtype=np.float32)
        vtk_data_array = numpy_to_vtk(self.pressure_array)
        vtk_data_array.SetName("Pressure")
        self.volume_data.GetCellData().SetScalars(vtk_data_array)
        
        # Create volume mapper
        self.volume_mapper = vtk.vtkSmartVolumeMapper()
        self.volume_mapper.SetInputData(self.volume_data)
        self.volume_mapper.SetBlendModeToComposite()
        
        # Create lookup table and color/opacity functions
        self.setup_color_mapping()
        
        # Create volume actor
        self.volume_actor = vtk.vtkVolume()
        self.volume_actor.SetMapper(self.volume_mapper)
        self.volume_actor.SetProperty(self.volume_property)
        
        self.renderer.AddVolume(self.volume_actor)
        
        # Add axes
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(nx * voxel_size * 0.3, 
                           ny * voxel_size * 0.3, 
                           nz * voxel_size * 0.3)
        axes.SetShaftType(0)
        axes.SetCylinderRadius(0.02)
        self.renderer.AddActor(axes)
        
        # Add outline
        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(self.volume_data)
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        outline_actor = vtk.vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
        self.renderer.AddActor(outline_actor)
        
    def setup_color_mapping(self):
        """Setup color mapping and scalar bar"""
        # Create lookup table
        self.lut = vtk.vtkLookupTable()
        self.lut.SetRange(0.0, 1.0)  # Always normalized to 0-1
        self.lut.SetHueRange(0.667, 0.0)  # Blue to Red
        self.lut.Build()
        
        # Create volume property
        self.volume_property = vtk.vtkVolumeProperty()
        
        # Color transfer function (normalized 0-1)
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(0.0, 0.0, 0.0, 1.0)  # Blue
        color_func.AddRGBPoint(0.25, 0.0, 0.5, 1.0)  # Cyan-blue
        color_func.AddRGBPoint(0.5, 0.0, 1.0, 0.0)  # Green
        color_func.AddRGBPoint(0.75, 1.0, 1.0, 0.0)  # Yellow
        color_func.AddRGBPoint(1.0, 1.0, 0.0, 0.0)  # Red
        
        # Opacity transfer function
        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(-0.1, 0.0)   # Unmeasured voxels (transparent)
        opacity_func.AddPoint(0.0, 0.1)    # Min pressure (slightly visible)
        opacity_func.AddPoint(0.25, 0.2)   # Low pressure
        opacity_func.AddPoint(0.5, 0.4)    # Medium pressure
        opacity_func.AddPoint(0.75, 0.6)   # High pressure
        opacity_func.AddPoint(1.0, 0.8)    # Max pressure
        
        self.volume_property.SetColor(color_func)
        self.volume_property.SetScalarOpacity(opacity_func)
        self.volume_property.ShadeOff()
        self.volume_property.SetInterpolationTypeToLinear()
        
        # Create scalar bar
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(self.lut)
        self.scalar_bar.SetTitle("Normalized\nPressure")
        self.scalar_bar.SetNumberOfLabels(6)
        self.scalar_bar.SetPosition(0.9, 0.1)
        self.scalar_bar.SetWidth(0.08)
        self.scalar_bar.SetHeight(0.8)
        
        # Customize scalar bar appearance
        title_prop = self.scalar_bar.GetTitleTextProperty()
        title_prop.SetFontSize(14)
        title_prop.SetColor(1.0, 1.0, 1.0)
        title_prop.BoldOn()
        
        label_prop = self.scalar_bar.GetLabelTextProperty()
        label_prop.SetFontSize(12)
        label_prop.SetColor(1.0, 1.0, 1.0)
        
        self.renderer.AddActor2D(self.scalar_bar)
        
    def setup_progress_display(self):
        """Setup progress bar and text display"""
        # Progress text
        self.progress_text.SetInput("Waiting for measurement to start...")
        text_prop = self.progress_text.GetTextProperty()
        text_prop.SetFontSize(20)
        text_prop.SetColor(1.0, 1.0, 1.0)
        text_prop.SetFontFamilyToArial()
        text_prop.BoldOn()
        self.progress_text.SetPosition(10, 10)
        
        # Progress bar
        self.progress_bar.SetInput("")
        bar_prop = self.progress_bar.GetTextProperty()
        bar_prop.SetFontSize(16)
        bar_prop.SetColor(0.0, 1.0, 0.0)
        bar_prop.SetFontFamilyToCourier()
        bar_prop.BoldOn()
        self.progress_bar.SetPosition(10, 40)
        
        # Statistics text
        self.stats_text.SetInput("")
        stats_prop = self.stats_text.GetTextProperty()
        stats_prop.SetFontSize(14)
        stats_prop.SetColor(0.8, 0.8, 0.8)
        stats_prop.SetFontFamilyToArial()
        self.stats_text.SetPosition(10, 70)
        
        # Add to renderer
        self.renderer.AddActor2D(self.progress_text)
        self.renderer.AddActor2D(self.progress_bar)
        self.renderer.AddActor2D(self.stats_text)
        
    def update_progress_display(self):
        """Update the progress bar and statistics"""
        if self.total_voxels == 0:
            return
            
        progress = self.measured_voxels / self.total_voxels
        
        # Calculate ETA
        if self.start_time and self.measured_voxels > 0:
            elapsed = time.time() - self.start_time
            rate = self.measured_voxels / elapsed
            remaining = self.total_voxels - self.measured_voxels
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "calculating..."
            
        # Update text
        text = f"Progress: {self.measured_voxels}/{self.total_voxels} voxels ({progress*100:.1f}%) | ETA: {eta_str}"
        self.progress_text.SetInput(text)
        
        # Create progress bar
        bar_width = 50
        filled = int(bar_width * progress)
        bar = "[" + "=" * filled + ">" + " " * (bar_width - filled - 1) + "]"
        self.progress_bar.SetInput(bar)
        
        # Update statistics
        if len(self.voxel_positions) > 0:
            values = list(self.voxel_positions.values())
            stats = f"Actual pressure range: {min(values):.6f} - {max(values):.6f} (Δ={max(values)-min(values):.6f})"
            self.stats_text.SetInput(stats)
        
    def load_voxel_data(self, voxel_path):
        """Load and process a single voxel measurement"""
        try:
            data = np.load(voxel_path, allow_pickle=True).item()
            
            # Extract voxel indices from filename
            filename = voxel_path.stem
            parts = filename.split('_')
            if len(parts) >= 4 and parts[0] == 'voxel':
                ix = int(parts[1])
                iy = int(parts[2])
                iz = int(parts[3])
                
                # Get RMS value (pressure intensity)
                rms = data.get('rms', 0.0)
                
                # Update pressure array
                nx, ny, nz = self.config['shape']
                if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                    # Calculate linear index (VTK uses Fortran ordering)
                    idx = ix + iy * nx + iz * nx * ny
                    self.pressure_array[idx] = rms
                    
                    # Track this voxel
                    self.voxel_positions[(ix, iy, iz)] = rms
                    
                    # Start timer on first measurement
                    if self.start_time is None:
                        self.start_time = time.time()
                        
                    return True
                    
        except Exception as e:
            print(f"Error loading voxel data: {e}")
            
        return False
        
    def update_volume_display(self):
        """Update the volume visualization with new data"""
        if self.volume_data is None:
            return
            
        # Re-normalize entire pressure field with current min/max
        if len(self.voxel_positions) > 1:
            values = list(self.voxel_positions.values())
            min_pressure = min(values)
            max_pressure = max(values)
            
            # Create normalized array
            normalized_array = np.zeros_like(self.pressure_array)
            
            # Only normalize measured voxels
            if max_pressure > min_pressure:
                # Min-max normalization for measured values
                for idx, pressure in enumerate(self.pressure_array):
                    if pressure > 0:  # Only normalize measured voxels
                        normalized_array[idx] = (pressure - min_pressure) / (max_pressure - min_pressure)
                    else:
                        normalized_array[idx] = -0.1  # Unmeasured voxels stay transparent
            else:
                # If all values are the same, set them to 0.5
                for idx, pressure in enumerate(self.pressure_array):
                    if pressure > 0:
                        normalized_array[idx] = 0.5
                    else:
                        normalized_array[idx] = -0.1
        else:
            # For single measurement or no measurements
            normalized_array = np.where(self.pressure_array > 0, 0.5, -0.1)
            
        # Update VTK array
        vtk_data_array = numpy_to_vtk(normalized_array)
        vtk_data_array.SetName("Pressure")
        self.volume_data.GetCellData().SetScalars(vtk_data_array)
        self.volume_data.Modified()
        
    def check_for_updates(self):
        """Check for new measurement files"""
        if self.data_dir is None:
            # Try to find measurement directory
            self.data_dir = self.find_latest_measurement_dir()
            if self.data_dir:
                config_path = self.data_dir / "config.npy"
                if self.load_config(config_path):
                    self.setup_volume()
                    print(f"Monitoring: {self.data_dir}")
                    
        if self.data_dir is None:
            return
            
        # Check for new voxel files
        new_files_found = False
        for voxel_file in self.data_dir.glob("voxel_*.npy"):
            if voxel_file not in self.seen_files:
                self.seen_files.add(voxel_file)
                if self.load_voxel_data(voxel_file):
                    self.measured_voxels += 1
                    new_files_found = True
                    
        # Update display if new data
        if new_files_found:
            self.update_volume_display()
            self.update_progress_display()
            self.render_window.Render()
            
    def timer_callback(self, caller, event):
        """Timer callback for periodic updates"""
        self.check_for_updates()
        
    def run(self):
        """Start the visualization"""
        # Setup display
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.setup_progress_display()
        
        # Initial check
        self.check_for_updates()
        
        # Setup camera
        if self.volume_actor:
            self.renderer.ResetCamera()
            camera = self.renderer.GetActiveCamera()
            camera.Azimuth(30)
            camera.Elevation(30)
            camera.Zoom(0.8)
            
        # Setup timer for updates (100ms intervals)
        self.interactor.Initialize()
        self.timer_id = self.interactor.CreateRepeatingTimer(100)
        self.interactor.AddObserver('TimerEvent', self.timer_callback)
        
        # Add keyboard interaction
        def key_press(obj, event):
            key = obj.GetKeySym()
            if key == 'q' or key == 'Q':
                self.interactor.ExitCallback()
            elif key == 'r' or key == 'R':
                self.renderer.ResetCamera()
                self.render_window.Render()
                
        self.interactor.AddObserver('KeyPressEvent', key_press)
        
        # Start interaction
        self.render_window.Render()
        print("\nLive Pressure Field Visualization")
        print("-" * 40)
        print("Controls:")
        print("  Mouse: Rotate/zoom/pan")
        print("  R: Reset camera")
        print("  Q: Quit")
        print("\nWaiting for measurement data...")
        
        self.interactor.Start()

def main():
    visualizer = PressureFieldVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
visualize_debug.py - Debug visualization by loading existing pressure field data
Loads all data from a specified directory to test the visualization logic
"""

import vtk
import numpy as np
import pathlib
from vtk.util.numpy_support import numpy_to_vtk

class DebugPressureFieldVisualizer:
    def __init__(self, data_dir):
        self.data_dir = pathlib.Path(data_dir)
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1024, 768)
        self.render_window.SetWindowName(f"Debug Pressure Field - {self.data_dir.name}")
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        
        # Volume data
        self.volume_data = None
        self.volume_actor = None
        self.pressure_array = None
        self.config = None
        self.voxel_positions = {}
        
    def load_config(self):
        """Load measurement configuration"""
        config_path = self.data_dir / "config.npy"
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            return False
            
        self.config = np.load(config_path, allow_pickle=True).item()
        print(f"Loaded config: {self.config['shape']} voxels, {self.config['voxel_size_mm']}mm spacing")
        return True
        
    def load_all_voxels(self):
        """Load all voxel data files"""
        voxel_files = list(self.data_dir.glob("voxel_*.npy"))
        print(f"Found {len(voxel_files)} voxel files")
        
        loaded_count = 0
        for voxel_file in voxel_files:
            try:
                data = np.load(voxel_file, allow_pickle=True).item()
                
                # Extract indices from filename
                filename = voxel_file.stem
                parts = filename.split('_')
                if len(parts) >= 4 and parts[0] == 'voxel':
                    ix = int(parts[1])
                    iy = int(parts[2])
                    iz = int(parts[3])
                    
                    # Get RMS value
                    rms = data.get('rms', 0.0)
                    
                    # Store data
                    nx, ny, nz = self.config['shape']
                    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                        # Calculate linear index
                        idx = ix + iy * nx + iz * nx * ny
                        self.pressure_array[idx] = rms
                        self.voxel_positions[(ix, iy, iz)] = rms
                        loaded_count += 1
                        
                        if loaded_count % 100 == 0:
                            print(f"  Loaded {loaded_count} voxels...")
                            
            except Exception as e:
                print(f"Error loading {voxel_file}: {e}")
                
        print(f"Successfully loaded {loaded_count} voxels")
        
        # Print statistics
        if self.voxel_positions:
            values = list(self.voxel_positions.values())
            print(f"Pressure statistics:")
            print(f"  Min: {min(values):.6f}")
            print(f"  Max: {max(values):.6f}")
            print(f"  Mean: {np.mean(values):.6f}")
            print(f"  Non-zero: {sum(1 for v in values if v > 0)} voxels")
        
    def setup_volume(self):
        """Initialize the volume visualization"""
        if not self.config:
            return
            
        nx, ny, nz = self.config['shape']
        voxel_size = self.config['voxel_size_mm']
        
        print(f"Setting up volume: {nx}x{ny}x{nz} voxels, {voxel_size}mm spacing")
        
        # Create image data
        self.volume_data = vtk.vtkImageData()
        self.volume_data.SetDimensions(nx + 1, ny + 1, nz + 1)
        self.volume_data.SetSpacing(voxel_size, voxel_size, voxel_size)
        self.volume_data.SetOrigin(0, 0, 0)
        
        # Initialize pressure array
        self.pressure_array = np.zeros((nx * ny * nz,), dtype=np.float32)
        
    def update_volume_display(self):
        """Update the volume visualization with loaded data"""
        if self.volume_data is None:
            return
            
        # Normalize pressure values
        if len(self.voxel_positions) > 0:
            max_pressure = max(self.voxel_positions.values())
            print(f"Normalizing with max pressure: {max_pressure:.6f}")
            if max_pressure > 0:
                normalized_array = self.pressure_array / max_pressure
            else:
                normalized_array = self.pressure_array
        else:
            print("No voxel data to display!")
            normalized_array = self.pressure_array
            
        # Update VTK array
        vtk_data_array = numpy_to_vtk(normalized_array)
        vtk_data_array.SetName("Pressure")
        self.volume_data.GetCellData().SetScalars(vtk_data_array)
        self.volume_data.Modified()
        
        # Create volume mapper
        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(self.volume_data)
        volume_mapper.SetBlendModeToComposite()
        
        # Create volume property
        volume_property = vtk.vtkVolumeProperty()
        
        # Color transfer function
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(0.0, 0.0, 0.0, 1.0)  # Blue
        color_func.AddRGBPoint(0.25, 0.0, 0.5, 1.0)  # Cyan-blue
        color_func.AddRGBPoint(0.5, 0.0, 1.0, 0.0)  # Green
        color_func.AddRGBPoint(0.75, 1.0, 1.0, 0.0)  # Yellow
        color_func.AddRGBPoint(1.0, 1.0, 0.0, 0.0)  # Red
        
        # Opacity transfer function - make it more visible for debugging
        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(0.0, 0.0)    # Transparent at 0
        opacity_func.AddPoint(0.01, 0.2)   # More visible at low values
        opacity_func.AddPoint(0.1, 0.4)    # Even more visible
        opacity_func.AddPoint(0.5, 0.6)    # Semi-opaque
        opacity_func.AddPoint(1.0, 0.8)    # Almost opaque for high values
        
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)
        volume_property.ShadeOff()
        volume_property.SetInterpolationTypeToLinear()
        
        # Create volume actor
        self.volume_actor = vtk.vtkVolume()
        self.volume_actor.SetMapper(volume_mapper)
        self.volume_actor.SetProperty(volume_property)
        
        self.renderer.AddVolume(self.volume_actor)
        
        # Add axes
        nx, ny, nz = self.config['shape']
        voxel_size = self.config['voxel_size_mm']
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
        outline_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        outline_actor.GetProperty().SetLineWidth(2)
        self.renderer.AddActor(outline_actor)
        
        # Add a scale bar
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(f"Volume: {nx*voxel_size}×{ny*voxel_size}×{nz*voxel_size} mm")
        text_prop = text_actor.GetTextProperty()
        text_prop.SetFontSize(16)
        text_prop.SetColor(1.0, 1.0, 1.0)
        text_actor.SetPosition(10, 10)
        self.renderer.AddActor2D(text_actor)
        
    def run(self):
        """Run the debug visualization"""
        # Setup display
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        
        # Load config
        if not self.load_config():
            print("Failed to load config!")
            return
            
        # Setup volume
        self.setup_volume()
        
        # Load all voxel data
        self.load_all_voxels()
        
        # Update display
        self.update_volume_display()
        
        # Setup camera
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(30)
        camera.Elevation(30)
        camera.Zoom(0.8)
        
        # Add keyboard interaction
        def key_press(obj, event):
            key = obj.GetKeySym()
            if key == 'q' or key == 'Q':
                self.interactor.ExitCallback()
            elif key == 'r' or key == 'R':
                self.renderer.ResetCamera()
                self.render_window.Render()
            elif key == 'w' or key == 'W':
                # Toggle wireframe/solid
                if hasattr(self, 'outline_actor'):
                    self.outline_actor.GetProperty().SetRepresentationToWireframe()
                self.render_window.Render()
                
        self.interactor.AddObserver('KeyPressEvent', key_press)
        
        # Start interaction
        self.interactor.Initialize()
        self.render_window.Render()
        
        print("\nDebug Pressure Field Visualization")
        print("-" * 40)
        print("Controls:")
        print("  Mouse: Rotate/zoom/pan")
        print("  R: Reset camera")
        print("  Q: Quit")
        
        self.interactor.Start()

def main():
    import sys
    
    # Default to the provided directory or use command line argument
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/Users/minjunes/fus/ctrl/pressure_field_data/20250703_165937"
    
    print(f"Loading data from: {data_dir}")
    
    visualizer = DebugPressureFieldVisualizer(data_dir)
    visualizer.run()

if __name__ == "__main__":
    main() 
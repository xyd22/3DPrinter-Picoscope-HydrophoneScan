#!/usr/bin/env python3
"""
visualize.py - Enhanced pressure field visualization with focal point emphasis
Implements dynamic normalization and opacity scaling for clear focal point visualization
"""

import vtk
import numpy as np
import pathlib
from vtk.util.numpy_support import numpy_to_vtk

class PressureFieldVisualizer:
    def __init__(self, data_dir):
        self.data_dir = pathlib.Path(data_dir)
        self.config = None
        self.pressure_field = None
        
        # VTK components
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1024, 768)
        self.render_window.SetWindowName("Pressure Field Visualization - Focal Point Enhanced")
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        
    def load_data(self):
        """Load configuration and pressure field data"""
        # Load config
        config_path = self.data_dir / "config.npy"
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            return False
            
        self.config = np.load(config_path, allow_pickle=True).item()
        print(f"Config loaded: {self.config}")
        
        # Load pressure field
        field_path = self.data_dir / "pressure_field.npy"
        if field_path.exists():
            self.pressure_field = np.load(field_path)
            print(f"Loaded pressure field shape: {self.pressure_field.shape}")
        else:
            print("No pressure_field.npy found, reconstructing from voxels...")
            # Reconstruct from individual voxel files
            nx, ny, nz = self.config['shape']
            self.pressure_field = np.zeros((nx, ny, nz))
            
            for voxel_file in self.data_dir.glob("voxel_*.npy"):
                try:
                    # Extract indices
                    parts = voxel_file.stem.split('_')
                    if len(parts) >= 4:
                        ix, iy, iz = int(parts[1]), int(parts[2]), int(parts[3])
                        data = np.load(voxel_file, allow_pickle=True).item()
                        self.pressure_field[ix, iy, iz] = data.get('rms', 0.0)
                except:
                    pass
                    
        # Print statistics
        print(f"\nPressure field statistics:")
        print(f"  Shape: {self.pressure_field.shape}")
        print(f"  Min: {self.pressure_field.min():.6f}")
        print(f"  Max: {self.pressure_field.max():.6f}")
        print(f"  Mean: {self.pressure_field.mean():.6f}")
        print(f"  Non-zero: {np.count_nonzero(self.pressure_field)} voxels")
        
        # Find focal point (maximum pressure location)
        max_idx = np.unravel_index(np.argmax(self.pressure_field), self.pressure_field.shape)
        print(f"  Focal point at voxel: {max_idx} with pressure: {self.pressure_field[max_idx]:.6f}")
        
        return True
        
    def create_volume(self):
        """Create VTK volume with enhanced opacity for focal point visualization"""
        nx, ny, nz = self.pressure_field.shape
        voxel_size = self.config['voxel_size_mm']
        
        # Create image data
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(nx + 1, ny + 1, nz + 1)
        image_data.SetSpacing(voxel_size, voxel_size, voxel_size)
        image_data.SetOrigin(0, 0, 0)
        
        # Normalize pressure field
        flat_field = self.pressure_field.flatten('F')  # Fortran order for VTK
        
        if self.pressure_field.max() > self.pressure_field.min():
            # Dynamic normalization based on actual min/max
            min_val = self.pressure_field[self.pressure_field > 0].min() if np.any(self.pressure_field > 0) else 0
            max_val = self.pressure_field.max()
            
            # Normalize measured voxels to 0-1, unmeasured to -0.1
            normalized_field = np.where(
                flat_field > 0,
                (flat_field - min_val) / (max_val - min_val),
                -0.1
            )
        else:
            normalized_field = np.where(flat_field > 0, 0.5, -0.1)
            
        # Convert to VTK array
        vtk_data = numpy_to_vtk(normalized_field.astype(np.float32))
        vtk_data.SetName("Pressure")
        image_data.GetCellData().SetScalars(vtk_data)
        
        # Create volume mapper
        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(image_data)
        volume_mapper.SetBlendModeToComposite()
        
        # Create volume property with enhanced focal point visibility
        volume_property = self.create_enhanced_volume_property()
        
        # Create volume actor
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        self.renderer.AddVolume(volume)
        
        return image_data
        
    def create_enhanced_volume_property(self):
        """Create volume property with enhanced opacity for focal point visualization"""
        volume_property = vtk.vtkVolumeProperty()
        
        # Color transfer function (blue to red through rainbow)
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(0.0, 0.0, 0.0, 1.0)   # Blue
        color_func.AddRGBPoint(0.2, 0.0, 0.5, 1.0)   # Cyan-blue
        color_func.AddRGBPoint(0.4, 0.0, 1.0, 0.5)   # Green-cyan
        color_func.AddRGBPoint(0.6, 0.5, 1.0, 0.0)   # Green-yellow
        color_func.AddRGBPoint(0.8, 1.0, 0.8, 0.0)   # Yellow
        color_func.AddRGBPoint(0.9, 1.0, 0.4, 0.0)   # Orange
        color_func.AddRGBPoint(1.0, 1.0, 0.0, 0.0)   # Red
        
        # Enhanced opacity transfer function for focal point visibility
        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(-0.1, 0.0)   # Unmeasured voxels (fully transparent)
        opacity_func.AddPoint(0.0, 0.02)   # Minimum pressure (barely visible)
        opacity_func.AddPoint(0.1, 0.30)   # Very low pressure
        opacity_func.AddPoint(0.3, 0.3)    # Low pressure (mostly transparent)
        opacity_func.AddPoint(0.7, 0.3)    # Medium pressure
        opacity_func.AddPoint(0.7, 0.5)    # Medium-high pressure
        opacity_func.AddPoint(0.85, 0.65)   # High pressure (focal region)
        opacity_func.AddPoint(0.95, 0.80)   # Very high pressure (focal core)
        opacity_func.AddPoint(1.0, 0.95)   # Maximum pressure (focal center)
        
        # Apply a gradient opacity for better edge detection
        gradient_opacity = vtk.vtkPiecewiseFunction()
        gradient_opacity.AddPoint(0.0, 0.0)
        gradient_opacity.AddPoint(0.1, 0.1)
        gradient_opacity.AddPoint(0.3, 0.3)
        gradient_opacity.AddPoint(0.5, 0.5)
        gradient_opacity.AddPoint(1.0, 0.8)
        
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)
        volume_property.SetGradientOpacity(gradient_opacity)
        volume_property.ShadeOn()  # Enable shading for better 3D perception
        volume_property.SetInterpolationTypeToLinear()
        volume_property.SetAmbient(0.2)
        volume_property.SetDiffuse(0.7)
        volume_property.SetSpecular(0.3)
        volume_property.SetSpecularPower(20)
        
        return volume_property
        
    def add_visualization_elements(self, image_data):
        """Add axes, outline, scalar bar, and text information"""
        nx, ny, nz = self.pressure_field.shape
        voxel_size = self.config['voxel_size_mm']
        
        # Add coordinate axes
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(nx * voxel_size * 0.3, 
                           ny * voxel_size * 0.3, 
                           nz * voxel_size * 0.3)
        axes.SetShaftType(0)
        axes.SetCylinderRadius(0.02)
        self.renderer.AddActor(axes)
        
        # Add bounding box
        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(image_data)
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        outline_actor = vtk.vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
        outline_actor.GetProperty().SetLineWidth(2)
        self.renderer.AddActor(outline_actor)
        
        # Create scalar bar
        lut = vtk.vtkLookupTable()
        lut.SetRange(0.0, 1.0)
        lut.SetHueRange(0.667, 0.0)  # Blue to Red
        lut.SetAlphaRange(0.1, 0.95)  # Match opacity function
        lut.Build()
        
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)
        scalar_bar.SetTitle("Normalized\nPressure")
        scalar_bar.SetNumberOfLabels(6)
        scalar_bar.SetPosition(0.9, 0.1)
        scalar_bar.SetWidth(0.08)
        scalar_bar.SetHeight(0.6)
        
        # Customize scalar bar appearance
        title_prop = scalar_bar.GetTitleTextProperty()
        title_prop.SetFontSize(14)
        title_prop.SetColor(1.0, 1.0, 1.0)
        title_prop.BoldOn()
        
        label_prop = scalar_bar.GetLabelTextProperty()
        label_prop.SetFontSize(12)
        label_prop.SetColor(1.0, 1.0, 1.0)
        
        self.renderer.AddActor2D(scalar_bar)
        
        # Add text information
        text_actor = vtk.vtkTextActor()
        info_text = (f"Volume: {nx}×{ny}×{nz} voxels ({voxel_size}mm)\n"
                    f"Pressure range: {self.pressure_field.min():.6f} - {self.pressure_field.max():.6f}\n"
                    f"Non-zero voxels: {np.count_nonzero(self.pressure_field)}")
        text_actor.SetInput(info_text)
        text_prop = text_actor.GetTextProperty()
        text_prop.SetFontSize(16)
        text_prop.SetColor(1, 1, 1)
        text_prop.SetFontFamilyToArial()
        text_actor.SetPosition(10, 10)
        self.renderer.AddActor2D(text_actor)
        
        # Add focal point marker
        max_idx = np.unravel_index(np.argmax(self.pressure_field), self.pressure_field.shape)
        focal_text = vtk.vtkTextActor()
        focal_text.SetInput(f"Focal point: {max_idx}")
        focal_prop = focal_text.GetTextProperty()
        focal_prop.SetFontSize(14)
        focal_prop.SetColor(1.0, 0.8, 0.0)
        focal_prop.SetFontFamilyToArial()
        focal_prop.BoldOn()
        focal_text.SetPosition(10, 80)
        self.renderer.AddActor2D(focal_text)
        
    def run(self):
        """Run the visualization"""
        # Load data
        if not self.load_data():
            return
            
        # Create volume
        image_data = self.create_volume()
        
        # Add visualization elements
        self.add_visualization_elements(image_data)
        
        # Setup camera
        self.renderer.SetBackground(0.05, 0.05, 0.05)
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(45)
        camera.Elevation(30)
        camera.Zoom(0.8)
        
        # Keyboard controls
        def key_press(obj, event):
            key = obj.GetKeySym()
            if key == 'q' or key == 'Q':
                self.interactor.ExitCallback()
            elif key == 'r' or key == 'R':
                self.renderer.ResetCamera()
                self.render_window.Render()
            elif key == 'f' or key == 'F':
                # Focus on focal point
                max_idx = np.unravel_index(np.argmax(self.pressure_field), self.pressure_field.shape)
                focal_pos = [max_idx[i] * self.config['voxel_size_mm'] for i in range(3)]
                camera.SetFocalPoint(focal_pos)
                camera.SetPosition(focal_pos[0] + 100, focal_pos[1] + 100, focal_pos[2] + 100)
                self.render_window.Render()
                
        self.interactor.AddObserver('KeyPressEvent', key_press)
        
        # Print instructions
        print("\nPressure Field Visualization - Enhanced Focal Point")
        print("-" * 50)
        print("Controls:")
        print("  Mouse: Rotate/zoom/pan")
        print("  R: Reset camera")
        print("  F: Focus on focal point")
        print("  Q: Quit")
        print("\nOpacity scaled to emphasize high-pressure focal regions")
        
        # Start interaction
        self.interactor.Initialize()
        self.render_window.Render()
        self.interactor.Start()

def main():
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        # Use default or most recent directory
        data_dir = "/Users/minjunes/fus/ctrl/pressure_field_data/20250703_165937"
        
    print(f"Visualizing data from: {data_dir}")
    
    visualizer = PressureFieldVisualizer(data_dir)
    visualizer.run()

if __name__ == "__main__":
    main() 
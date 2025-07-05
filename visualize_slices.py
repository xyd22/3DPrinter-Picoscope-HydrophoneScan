#!/usr/bin/env python3
"""
visualize_slices.py - Slice-based visualization of pressure field data
Shows the pressure field as interactive 2D slices through the volume
"""

import vtk
import numpy as np
import pathlib
from vtk.util.numpy_support import numpy_to_vtk

class SlicePressureFieldVisualizer:
    def __init__(self, data_dir):
        self.data_dir = pathlib.Path(data_dir)
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1024, 768)
        self.render_window.SetWindowName(f"Slice Visualization - {self.data_dir.name}")
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        
        # Data
        self.volume_data = None
        self.pressure_field = None
        self.config = None
        
        # Slices
        self.x_slice = None
        self.y_slice = None
        self.z_slice = None
        self.current_x = 0
        self.current_y = 0
        self.current_z = 0
        
    def load_data(self):
        """Load configuration and pressure field data"""
        # Load config
        config_path = self.data_dir / "config.npy"
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            return False
            
        self.config = np.load(config_path, allow_pickle=True).item()
        nx, ny, nz = self.config['shape']
        
        # Try to load complete pressure field first
        field_path = self.data_dir / "pressure_field.npy"
        if field_path.exists():
            self.pressure_field = np.load(field_path)
        else:
            # Reconstruct from voxel files
            self.pressure_field = np.zeros((nx, ny, nz))
            count = 0
            for voxel_file in self.data_dir.glob("voxel_*.npy"):
                parts = voxel_file.stem.split('_')
                if len(parts) >= 4:
                    ix, iy, iz = int(parts[1]), int(parts[2]), int(parts[3])
                    data = np.load(voxel_file, allow_pickle=True).item()
                    self.pressure_field[ix, iy, iz] = data.get('rms', 0.0)
                    count += 1
            print(f"Reconstructed from {count} voxel files")
            
        # Print statistics
        print(f"Pressure field shape: {self.pressure_field.shape}")
        print(f"Value range: {self.pressure_field.min():.6f} to {self.pressure_field.max():.6f}")
        print(f"Range span: {self.pressure_field.max() - self.pressure_field.min():.6f}")
        
        return True
        
    def create_volume_data(self):
        """Create VTK volume data"""
        nx, ny, nz = self.pressure_field.shape
        voxel_size = self.config['voxel_size_mm']
        
        # Create image data
        self.volume_data = vtk.vtkImageData()
        self.volume_data.SetDimensions(nx, ny, nz)
        self.volume_data.SetSpacing(voxel_size, voxel_size, voxel_size)
        self.volume_data.SetOrigin(0, 0, 0)
        
        # Normalize data using min-max
        min_val = self.pressure_field.min()
        max_val = self.pressure_field.max()
        
        if max_val > min_val:
            normalized = (self.pressure_field - min_val) / (max_val - min_val)
        else:
            normalized = np.ones_like(self.pressure_field) * 0.5
            
        # Convert to VTK array (note: VTK expects point data, not cell data for slicing)
        flat_data = normalized.flatten('F')  # Fortran order
        vtk_data = numpy_to_vtk(flat_data)
        vtk_data.SetName("Pressure")
        self.volume_data.GetPointData().SetScalars(vtk_data)
        
    def create_slices(self):
        """Create three orthogonal slices"""
        nx, ny, nz = self.pressure_field.shape
        voxel_size = self.config['voxel_size_mm']
        
        # Color lookup table
        lut = vtk.vtkLookupTable()
        lut.SetRange(0, 1)
        lut.SetHueRange(0.667, 0.0)  # Blue to Red
        lut.Build()
        
        # X slice (YZ plane)
        self.x_slice = vtk.vtkImageActor()
        self.x_slice.GetMapper().SetInputData(self.volume_data)
        self.x_slice.SetDisplayExtent(nx//2, nx//2, 0, ny-1, 0, nz-1)
        self.x_slice.GetProperty().SetColorWindow(1.0)
        self.x_slice.GetProperty().SetColorLevel(0.5)
        self.x_slice.GetProperty().SetLookupTable(lut)
        self.x_slice.SetOpacity(0.8)
        self.current_x = nx // 2
        
        # Y slice (XZ plane)
        self.y_slice = vtk.vtkImageActor()
        self.y_slice.GetMapper().SetInputData(self.volume_data)
        self.y_slice.SetDisplayExtent(0, nx-1, ny//2, ny//2, 0, nz-1)
        self.y_slice.GetProperty().SetColorWindow(1.0)
        self.y_slice.GetProperty().SetColorLevel(0.5)
        self.y_slice.GetProperty().SetLookupTable(lut)
        self.y_slice.SetOpacity(0.8)
        self.current_y = ny // 2
        
        # Z slice (XY plane)
        self.z_slice = vtk.vtkImageActor()
        self.z_slice.GetMapper().SetInputData(self.volume_data)
        self.z_slice.SetDisplayExtent(0, nx-1, 0, ny-1, nz//2, nz//2)
        self.z_slice.GetProperty().SetColorWindow(1.0)
        self.z_slice.GetProperty().SetColorLevel(0.5)
        self.z_slice.GetProperty().SetLookupTable(lut)
        self.z_slice.SetOpacity(0.8)
        self.current_z = nz // 2
        
        # Add to renderer
        self.renderer.AddActor(self.x_slice)
        self.renderer.AddActor(self.y_slice)
        self.renderer.AddActor(self.z_slice)
        
        # Add color bar
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)
        scalar_bar.SetTitle("Normalized Pressure")
        scalar_bar.SetNumberOfLabels(5)
        self.renderer.AddActor2D(scalar_bar)
        
    def add_annotations(self):
        """Add volume outline and axes"""
        nx, ny, nz = self.pressure_field.shape
        voxel_size = self.config['voxel_size_mm']
        
        # Outline
        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(self.volume_data)
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        outline_actor = vtk.vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(1, 1, 1)
        outline_actor.GetProperty().SetLineWidth(2)
        self.renderer.AddActor(outline_actor)
        
        # Axes
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(nx * voxel_size * 0.3,
                           ny * voxel_size * 0.3,
                           nz * voxel_size * 0.3)
        self.renderer.AddActor(axes)
        
        # Text info
        text_actor = vtk.vtkTextActor()
        info = (f"Volume: {nx}×{ny}×{nz} voxels | "
                f"Range: {self.pressure_field.min():.6f} - {self.pressure_field.max():.6f} | "
                f"Use arrow keys to move slices")
        text_actor.SetInput(info)
        text_prop = text_actor.GetTextProperty()
        text_prop.SetFontSize(14)
        text_prop.SetColor(1, 1, 1)
        text_actor.SetPosition(10, 10)
        self.renderer.AddActor2D(text_actor)
        
    def update_slice_positions(self):
        """Update slice display extents"""
        nx, ny, nz = self.pressure_field.shape
        
        # Clamp values
        self.current_x = max(0, min(self.current_x, nx-1))
        self.current_y = max(0, min(self.current_y, ny-1))
        self.current_z = max(0, min(self.current_z, nz-1))
        
        # Update display extents
        self.x_slice.SetDisplayExtent(self.current_x, self.current_x, 0, ny-1, 0, nz-1)
        self.y_slice.SetDisplayExtent(0, nx-1, self.current_y, self.current_y, 0, nz-1)
        self.z_slice.SetDisplayExtent(0, nx-1, 0, ny-1, self.current_z, self.current_z)
        
        self.render_window.Render()
        
    def run(self):
        """Run the visualization"""
        # Load data
        if not self.load_data():
            return
            
        # Create visualization
        self.create_volume_data()
        self.create_slices()
        self.add_annotations()
        
        # Setup renderer
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(30)
        camera.Elevation(30)
        
        # Keyboard interaction
        def key_press(obj, event):
            key = obj.GetKeySym()
            if key == 'q' or key == 'Q':
                self.interactor.ExitCallback()
            elif key == 'r' or key == 'R':
                self.renderer.ResetCamera()
                self.render_window.Render()
            elif key == 'Left':
                self.current_x -= 1
                self.update_slice_positions()
            elif key == 'Right':
                self.current_x += 1
                self.update_slice_positions()
            elif key == 'Down':
                self.current_y -= 1
                self.update_slice_positions()
            elif key == 'Up':
                self.current_y += 1
                self.update_slice_positions()
            elif key == 'Prior' or key == 'Page_Up':  # Page Up
                self.current_z += 1
                self.update_slice_positions()
            elif key == 'Next' or key == 'Page_Down':  # Page Down
                self.current_z -= 1
                self.update_slice_positions()
            elif key == 'x':
                self.x_slice.SetVisibility(not self.x_slice.GetVisibility())
                self.render_window.Render()
            elif key == 'y':
                self.y_slice.SetVisibility(not self.y_slice.GetVisibility())
                self.render_window.Render()
            elif key == 'z':
                self.z_slice.SetVisibility(not self.z_slice.GetVisibility())
                self.render_window.Render()
                
        self.interactor.AddObserver('KeyPressEvent', key_press)
        
        # Start
        self.interactor.Initialize()
        self.render_window.Render()
        
        print("\nSlice Visualization Controls:")
        print("-" * 40)
        print("Mouse: Rotate/zoom/pan")
        print("Left/Right arrows: Move X slice")
        print("Up/Down arrows: Move Y slice")
        print("Page Up/Down: Move Z slice")
        print("X/Y/Z: Toggle slice visibility")
        print("R: Reset camera")
        print("Q: Quit")
        
        self.interactor.Start()

def main():
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/Users/minjunes/fus/ctrl/pressure_field_data/20250703_165937"
        
    visualizer = SlicePressureFieldVisualizer(data_dir)
    visualizer.run()

if __name__ == "__main__":
    main() 
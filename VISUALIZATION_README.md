# Pressure Field Visualization Tools

This package provides multiple visualization tools for viewing pressure field measurements from `measure.py`.

## Installation

Install the required dependencies:

```bash
pip install vtk numpy
```

## Visualization Options

### 1. Live Visualization (Real-time Updates)

Monitors for new measurements and updates in real-time:

```bash
python visualize_live.py
```

**Features:**
- Auto-detects the latest measurement directory
- Updates every 100ms as new voxels are measured
- Shows progress bar with ETA
- Uses min-max normalization for narrow pressure ranges

### 2. Debug Visualization (Post-measurement)

For viewing completed measurements with detailed statistics:

```bash
python visualize_debug.py [path/to/data/directory]
# or with default path:
python visualize_debug.py
```

**Features:**
- Loads all voxel data at once
- Prints detailed statistics
- More visible opacity settings for debugging

### 3. Direct Field Visualization

Simple visualization of the complete pressure field:

```bash
python visualize_field_direct.py [path/to/data/directory]
```

**Features:**
- Loads pressure_field.npy directly
- Can reconstruct from individual voxel files if needed
- Shows min/max values and statistics

### 4. Slice Visualization

Interactive 2D slice views through the 3D volume:

```bash
python visualize_slices.py [path/to/data/directory]
```

**Features:**
- Three orthogonal slices (X, Y, Z planes)
- Arrow keys to move slices
- X/Y/Z keys to toggle slice visibility
- Best for examining narrow pressure ranges

## Usage Workflow

1. **For live monitoring during measurement:**
   ```bash
   # Terminal 1: Start visualization first
   python visualize_live.py
   
   # Terminal 2: Run measurement
   python measure.py
   ```

2. **For examining completed measurements:**
   ```bash
   # Use slice view for detailed examination
   python visualize_slices.py pressure_field_data/20250703_165937
   
   # Or use debug view for statistics
   python visualize_debug.py pressure_field_data/20250703_165937
   ```

## Troubleshooting

### Nothing visible in volume rendering?

This often happens when pressure values have a very narrow range (e.g., 1.024 to 1.028). The visualizations use min-max normalization to handle this, but you can also:

1. Use the slice visualization for better visibility
2. Check data statistics with `inspect_data.py`:
   ```bash
   python inspect_data.py pressure_field_data/20250703_165937
   ```

### Controls for All Visualizations

- **Mouse**: Click and drag to rotate, scroll to zoom, middle-click to pan
- **R**: Reset camera to default view
- **Q**: Quit the visualization

## Data Format

The visualization tools expect:
- `config.npy`: Contains volume dimensions, voxel size, and other metadata
- `voxel_XXX_YYY_ZZZ.npy`: Individual voxel measurements with RMS values
- `pressure_field.npy` (optional): Complete 3D array of pressure values

Each voxel file contains:
- `position_mm`: [x, y, z] position in millimeters
- `rms`: RMS pressure value
- `ch1_voltage`: Raw voltage measurements
- `time`: Time array for the measurement 
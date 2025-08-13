import numpy as np

file_path = r'pressure_field_data/20250812_145520/pressure_field_partial.npy'

pressure_field = np.load(file_path)

print(pressure_field.shape)
coords = np.argwhere(pressure_field == 0.0)
print(coords)
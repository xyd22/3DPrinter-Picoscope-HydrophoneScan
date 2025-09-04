import numpy as np

pressure_filed = np.load('./pressure_field_data/20250902_203004/pressure_field.npy')
focus_index = np.max(pressure_filed, axis = 0) > 10
focus_pressure_field = pressure_filed[:, focus_index]
max_index = np.argmax(focus_pressure_field, axis = 0)
print(focus_index)
# print(max_index)

X = np.linspace(1, max_index.shape[0], max_index.shape[0]).reshape(-1, 1)

slope, intercept = np.polyfit(X.flatten(), max_index, 1)

print(slope, intercept) # -0.46081277213352684 ~ -0.5

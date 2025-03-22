import numpy as np

# Parameters
T = 60         # Number of frames
J = 17         # Joints (MotionBERT expects 17)
C = 3          # 3D coordinates (x, y, z)

# Create random dummy skeleton data
dummy_data = np.random.rand(1, T, J, C).astype(np.float32)

# Save to .npy file
np.save("custom_input.npy", dummy_data)
print(f"✅ Dummy input saved with shape {dummy_data.shape} to custom_input.npy")

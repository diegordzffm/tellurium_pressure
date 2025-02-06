import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Simulated training data (example k-points and energy bands)
kpoints = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [1/3, 1/3, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.5]
])

band_energies = np.array([
    [-1.2, -0.8, 0.0, 0.5, 1.0],
    [-1.0, -0.5, 0.2, 0.8, 1.5],
    [-0.9, -0.4, 0.3, 1.0, 1.8],
    [-1.1, -0.6, 0.1, 0.7, 1.3]
]).T  # Transpose to align with k-points

# Define Neural Network Model
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(3,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(band_energies.shape[1])  # Output layer with same number of bands
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train the model
history = model.fit(kpoints, band_energies, epochs=200, verbose=1)

# Predict band energies for given k-points
predicted_energies = model.predict(kpoints)

# Plot results
plt.figure(figsize=(7, 5))
for i in range(band_energies.shape[1]):
    plt.plot(range(len(kpoints)), band_energies[:, i], color="blue", linestyle="solid", label=f"Actual Band {i+1}")
    plt.plot(range(len(kpoints)), predicted_energies[:, i], color="red", linestyle="dashed", label=f"Predicted Band {i+1}")

plt.xticks(range(len(kpoints)), [r"$\Gamma$", "M", "K", r"$\Gamma$", "A"])
plt.ylabel("Energy (eV)")
plt.xlabel("Wave Vector")
plt.title("Neural Network Band Structure Prediction")
plt.grid(True)
plt.legend()
plt.show()

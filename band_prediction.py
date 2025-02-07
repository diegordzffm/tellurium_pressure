import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


kpoints = np.array([
num_kpoints = 50
high_symmetry_points = np.array([

    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [1/3, 1/3, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.5]
])

kpoints = np.concatenate([
    np.linspace(high_symmetry_points[i], high_symmetry_points[i + 1], num_kpoints)
    for i in range(len(high_symmetry_points) - 1)
])

def tight_binding_model(k):
    t = 1.0  # Hopping parameter
    a = 1.0  # Lattice constant
    k_dot_a = np.linalg.norm(k) * a
    return -2 * t * (np.cos(k_dot_a) + np.cos(k_dot_a / 2))  # Simplified dispersion

band_energies = np.array([tight_binding_model(k) for k in kpoints]).reshape(-1, 1)

def fourier_features(kpoints, num_features=10):
    kpoints = np.asarray(kpoints)
    freq = np.linspace(1, 10, num_features)
    sin_features = np.sin(np.outer(kpoints[:, 0], freq))
    cos_features = np.cos(np.outer(kpoints[:, 1], freq))
    return np.hstack([sin_features, cos_features])

kpoints_transformed = fourier_features(kpoints, num_features=15)

model = keras.Sequential([
    layers.Input(shape=(kpoints_transformed.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01, decay_steps=100, decay_rate=0.96
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

history = model.fit(kpoints_transformed, band_energies, epochs=500, verbose=1, batch_size=16)

predicted_energies = model.predict(kpoints_transformed)

plt.figure(figsize=(8, 5))
plt.plot(range(len(kpoints)), band_energies, color="blue", linestyle="solid", label="Tight-Binding Model")
plt.plot(range(len(kpoints)), predicted_energies, color="red", linestyle="dashed", label="Neural Network Prediction")
plt.xticks(np.linspace(0, len(kpoints), len(high_symmetry_points)), [r"$\Gamma$", "M", "K", r"$\Gamma$", "A"])
plt.ylabel("Energy (eV)")
plt.xlabel("Wave Vector")
plt.title("Machine Learning Band Structure Prediction vs. Tight-Binding")
plt.grid(True)
plt.legend()
plt.show()

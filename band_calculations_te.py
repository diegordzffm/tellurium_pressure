from ase.build import bulk
from ase.calculators.espresso import Espresso
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import pymc3 as pm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


a = 4.45  

a = 4.45  #

c = 5.93  # c-axis length in Angstroms
structure = bulk("Te", crystalstructure="hexagonal", a=a, c=c)

pseudopotentials = {"Te": "Te.pbe-n-rrkjus_psl.1.0.0.UPF"}  
k_grid = (12, 12, 12)  

input_data = {
    "control": {
        "calculation": "scf",
        "outdir": "qe_output",
        "pseudo_dir": "./pseudos",
        "prefix": "Te",
        "verbosity": "high",
        "tprnfor": True,
        "tstress": True,
    },
    "system": {
        "ecutwfc": 60,
        "ecutrho": 240,
        "occupations": "smearing",
        "smearing": "gaussian",
        "degauss": 0.005,
        "nbnd": 40,
        "ibrav": 4,
        "celldm(1)": a / 0.529,
        "celldm(3)": c / a,
        "nspin": 2,
        "lspinorb": True,
        "noncolin": True,
        "lda_plus_u": True,
        "hubbard_u(1)": 2.0
    },
    "electrons": {
        "conv_thr": 1e-8,
        "mixing_beta": 0.4,
        "diagonalization": "david",
    },
    "ions": {
        "ion_dynamics": "bfgs"
    }
}

calc = Espresso(pseudopotentials=pseudopotentials, input_data=input_data, kpts=k_grid)
structure.set_calculator(calc)
calc.calculate(structure)

kpoints = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [1/3, 1/3, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.5],
])

subprocess.run("pw.x < band_structure.in > band_structure.out", shell=True)

band_energies = []
with open("band_structure.out", "r") as f:
    lines = f.readlines()
    for line in lines:
        if "bands (ev)" in line:
            band_energies.append([float(x) for x in lines[lines.index(line) + 1].split()])

band_energies = np.array(band_energies).T

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=band_energies.flatten())
    trace = pm.sample(1000, return_inferencedata=False)

model_nn = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(len(kpoints),)),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])
model_nn.compile(optimizer="adam", loss="mse")
model_nn.fit(kpoints, band_energies.mean(axis=0), epochs=200, verbose=0)
predicted_energies = model_nn.predict(kpoints)

plt.figure(figsize=(7, 5))
for band in band_energies:
    plt.plot(range(len(kpoints)), band, color="darkblue", linewidth=1)
plt.plot(range(len(kpoints)), predicted_energies, color="red", linestyle="dashed", label="NN Prediction")
plt.xticks(range(len(kpoints)), [r"$\Gamma$", "M", "K", r"$\Gamma$", "A"])
plt.ylabel("Energy (eV)")
plt.xlabel("Wave Vector")
plt.title("Advanced Band Structure of Tellurium (SOC + U) with Bayesian and NN Predictions")
plt.grid(True)
plt.legend()
plt.show()

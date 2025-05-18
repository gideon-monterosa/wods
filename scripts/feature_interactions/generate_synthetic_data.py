import numpy as np
import pandas as pd
import os

from feature_interaction.synthetic_data_generator import SyntheticDataGenerator

os.makedirs("./data/", exist_ok=True)

# Initialisiere den Datengenerator
data_generator = SyntheticDataGenerator()

# 1. Datensätze ohne Noise und irrelevante Features für verschiedene Größen
sample_sizes = [200, 1000, 5000, 25000]
for sample_size in sample_sizes:
    X, y = data_generator.generate_complex_synthetic_dataset(
        n_samples=sample_size, irrelevant_ratio=0, noise_level=0.0
    )

    feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    filepath = f"./data/complex_dataset_n{sample_size}_noise0_irrelevant0.csv"
    df.to_csv(filepath, index=False)
    print(f"Datensatz gespeichert: {filepath}")

# 2. Datensätze mit Noise, ohne irrelevante Features (Größe 1000)
noise_levels = [0.1, 0.2]
for noise_level in noise_levels:
    X, y = data_generator.generate_complex_synthetic_dataset(
        n_samples=1000, irrelevant_ratio=0, noise_level=noise_level
    )
    feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # Konvertiere Noise-Level zu ganzzahligen Werten (0.1 → 10, 0.2 → 20)
    noise_int = int(noise_level * 100)
    filepath = f"./data/complex_dataset_n1000_noise{noise_int}_irrelevant0.csv"
    df.to_csv(filepath, index=False)
    print(f"Datensatz gespeichert: {filepath}")

# 3. Datensätze ohne Noise, mit irrelevanten Features (Größe 1000)
irrelevant_ratios = [1, 2, 5]
for irrelevant_ratio in irrelevant_ratios:
    X, y = data_generator.generate_complex_synthetic_dataset(
        n_samples=1000, irrelevant_ratio=irrelevant_ratio, noise_level=0.0
    )

    feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    filepath = f"./data/complex_dataset_n1000_noise0_irrelevant{irrelevant_ratio}.csv"
    df.to_csv(filepath, index=False)
    print(f"Datensatz gespeichert: {filepath}")

# 4. Datensatz mit Noise und irrelevanten Features (Größe 1000, ratio 2, noise 0.1)
X, y = data_generator.generate_complex_synthetic_dataset(
    n_samples=1000, irrelevant_ratio=2, noise_level=0.1
)

feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

filepath = f"./data/complex_dataset_n1000_noise10_irrelevant2.csv"
df.to_csv(filepath, index=False)
print(f"Datensatz gespeichert: {filepath}")

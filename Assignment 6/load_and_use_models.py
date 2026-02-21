"""
Script to demonstrate loading and using the trained CTGAN and TVAE models.
"""

import pandas as pd
from ctgan import CTGAN, TVAE

print("=" * 60)
print("LOADING AND USING TRAINED MODELS")
print("=" * 60)

# Load CTGAN model
print("\n[1/4] Loading CTGAN model...")
ctgan = CTGAN.load('ctgan_model.pkl')
print("✓ CTGAN model loaded successfully")

# Generate new samples from CTGAN
print("\n[2/4] Generating 100 new samples from CTGAN...")
new_ctgan_samples = ctgan.sample(100)
print(f"✓ Generated {len(new_ctgan_samples)} samples")
print("\nSample data from CTGAN:")
print(new_ctgan_samples.head())
print(f"\nColumns: {list(new_ctgan_samples.columns)}")

# Load TVAE model
print("\n[3/4] Loading TVAE model...")
tvae = TVAE.load('tvae_model.pkl')
print("✓ TVAE model loaded successfully")

# Generate new samples from TVAE
print("\n[4/4] Generating 100 new samples from TVAE...")
new_tvae_samples = tvae.sample(100)
print(f"✓ Generated {len(new_tvae_samples)} samples")
print("\nSample data from TVAE:")
print(new_tvae_samples.head())

# Summary statistics
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\nBoth models can be loaded and used to generate new synthetic data!")
print("- CTGAN: Conditional Tabular GAN")
print("- TVAE: Tabular Variational Autoencoder")
print("\nTo generate more samples, use: model.sample(n_samples)")
print("=" * 60)

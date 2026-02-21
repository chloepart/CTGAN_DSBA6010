"""
Script to train CTGAN and TVAE models on the demo adult dataset.
"""

import pandas as pd
from ctgan import CTGAN, TVAE, load_demo

# Load the demo adult dataset
print("Loading demo adult dataset...")
real_data = load_demo()
print(f"Dataset loaded: {real_data.shape[0]} rows, {real_data.shape[1]} columns")
print(f"Columns: {list(real_data.columns)}\n")

# Define discrete columns for the adult dataset
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

print("=" * 60)
print("TRAINING CTGAN MODEL")
print("=" * 60)

# Initialize and train CTGAN
# Using 50 epochs for faster training (can be increased for better quality)
ctgan = CTGAN(epochs=50, verbose=True)
print("\nTraining CTGAN with 50 epochs...")
ctgan.fit(real_data, discrete_columns)
print("\nCTGAN training completed!")

# Generate synthetic samples from CTGAN
print("\nGenerating 1000 synthetic samples from CTGAN...")
ctgan_synthetic_data = ctgan.sample(1000)
print(f"Synthetic data generated: {ctgan_synthetic_data.shape}")

# Save CTGAN model
print("\nSaving CTGAN model...")
ctgan.save('ctgan_model.pkl')
print("CTGAN model saved to: ctgan_model.pkl")

# Save CTGAN synthetic data
ctgan_synthetic_data.to_csv('ctgan_synthetic_data.csv', index=False)
print("CTGAN synthetic data saved to: ctgan_synthetic_data.csv\n")

print("=" * 60)
print("TRAINING TVAE MODEL")
print("=" * 60)

# Initialize and train TVAE
# Using 50 epochs for faster training (can be increased for better quality)
tvae = TVAE(epochs=50, verbose=True)
print("\nTraining TVAE with 50 epochs...")
tvae.fit(real_data, discrete_columns)
print("\nTVAE training completed!")

# Generate synthetic samples from TVAE
print("\nGenerating 1000 synthetic samples from TVAE...")
tvae_synthetic_data = tvae.sample(1000)
print(f"Synthetic data generated: {tvae_synthetic_data.shape}")

# Save TVAE model
print("\nSaving TVAE model...")
tvae.save('tvae_model.pkl')
print("TVAE model saved to: tvae_model.pkl")

# Save TVAE synthetic data
tvae_synthetic_data.to_csv('tvae_synthetic_data.csv', index=False)
print("TVAE synthetic data saved to: tvae_synthetic_data.csv\n")

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("\nSummary:")
print("- CTGAN model: ctgan_model.pkl")
print("- CTGAN synthetic data: ctgan_synthetic_data.csv")
print("- TVAE model: tvae_model.pkl")
print("- TVAE synthetic data: tvae_synthetic_data.csv")

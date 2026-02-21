"""
Quick demo script to train CTGAN and TVAE models on the demo adult dataset.
Using reduced epochs for demonstration purposes.
"""

import pandas as pd
from ctgan import CTGAN, TVAE, load_demo

print("=" * 60)
print("CTGAN & TVAE TRAINING DEMO")
print("=" * 60)

# Load the demo adult dataset
print("\n[1/6] Loading demo adult dataset...")
real_data = load_demo()
print(f"✓ Dataset loaded: {real_data.shape[0]} rows, {real_data.shape[1]} columns")

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

# Train CTGAN
print("\n[2/6] Training CTGAN (10 epochs for quick demo)...")
ctgan = CTGAN(epochs=10, verbose=True)
ctgan.fit(real_data, discrete_columns)
print("✓ CTGAN training completed!")

# Save CTGAN model
print("\n[3/6] Saving CTGAN model and generating samples...")
ctgan.save('ctgan_model.pkl')
ctgan_synthetic_data = ctgan.sample(1000)
ctgan_synthetic_data.to_csv('ctgan_synthetic_data.csv', index=False)
print(f"✓ CTGAN model saved (ctgan_model.pkl)")
print(f"✓ Generated 1000 synthetic samples (ctgan_synthetic_data.csv)")

# Train TVAE
print("\n[4/6] Training TVAE (10 epochs for quick demo)...")
tvae = TVAE(epochs=10, verbose=True)
tvae.fit(real_data, discrete_columns)
print("✓ TVAE training completed!")

# Save TVAE model
print("\n[5/6] Saving TVAE model and generating samples...")
tvae.save('tvae_model.pkl')
tvae_synthetic_data = tvae.sample(1000)
tvae_synthetic_data.to_csv('tvae_synthetic_data.csv', index=False)
print(f"✓ TVAE model saved (tvae_model.pkl)")
print(f"✓ Generated 1000 synthetic samples (tvae_synthetic_data.csv)")

# Display summary
print("\n[6/6] Training Complete!")
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("\nModels saved:")
print("  • ctgan_model.pkl - CTGAN trained model")
print("  • tvae_model.pkl  - TVAE trained model")
print("\nSynthetic data generated:")
print("  • ctgan_synthetic_data.csv - 1000 samples from CTGAN")
print("  • tvae_synthetic_data.csv  - 1000 samples from TVAE")
print("\nNote: For production use, increase epochs (e.g., 300) for better quality.")
print("=" * 60)

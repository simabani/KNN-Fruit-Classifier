# src/generate_data.py

import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples per fruit
num_samples = 50

# Feature distributions per fruit
def generate_fruit_data(label, shape_range, color_range):
    shape = np.random.uniform(shape_range[0], shape_range[1], num_samples)
    color_score = np.random.uniform(color_range[0], color_range[1], num_samples)
    return pd.DataFrame({
        'shape': shape,
        'color_score': color_score,
        'label': label
    })

# Generate data for each fruit
apple_data = generate_fruit_data('apple', shape_range=(6.5, 8.0), color_range=(0.8, 1.0))
banana_data = generate_fruit_data('banana', shape_range=(5.0, 6.5), color_range=(0.5, 0.7))
orange_data = generate_fruit_data('orange', shape_range=(7.0, 8.5), color_range=(0.75, 0.95))

# Combine all into one DataFrame
all_data = pd.concat([apple_data, banana_data, orange_data], ignore_index=True)

# Ensure output directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV
all_data.to_csv('data/fruits.csv', index=False)

print("✅ Dummy fruit dataset generated and saved to data/fruits.csv")

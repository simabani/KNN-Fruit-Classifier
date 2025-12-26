# src/predict_sample.py

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv('data/fruits.csv')

# Prepare features and labels
X = data[['shape', 'color_score']]
y = data['label']

# Train a simple K-NN model (retraining for simplicity)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Define new samples to classify
new_samples = pd.DataFrame([
    {'shape': 7.2, 'color_score': 0.82},
    {'shape': 5.8, 'color_score': 0.60},
    {'shape': 8.1, 'color_score': 0.91}
])

# Predict the labels
predictions = knn.predict(new_samples)

# Output predictions
for i, sample in new_samples.iterrows():
    print(f"Sample {i+1}: shape={sample['shape']}, color_score={sample['color_score']} → predicted: {predictions[i]}")


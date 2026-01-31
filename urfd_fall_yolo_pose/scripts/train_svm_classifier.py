"""
Train Linear SVM Classifier on URFD Ground Truth Features
Mimics the paper's methodology to find optimal decision boundary.
"""
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json

# Load Ground Truth CSV
print("Loading Ground Truth features...")
cols = ['seq', 'frame', 'label', 'HW_Ratio', 'MM_Ratio', 'Occ', 'Std', 'HHmax', 'H', 'D', 'P40']
df_falls = pd.read_csv('../data/00_raw_download/extracted_features/urfall-cam0-falls.csv', 
                       header=None, names=cols)
df_adls = pd.read_csv('../data/00_raw_download/extracted_features/urfall-cam0-adls.csv', 
                      header=None, names=cols)

# Combine datasets
df_falls['source'] = 'fall'
df_adls['source'] = 'adl'
df = pd.concat([df_falls, df_adls], ignore_index=True)

# Filter: Only use lying frames (label=1) vs normal frames (label=-1)
# Exclude transition frames (label=0)
df = df[df['label'] != 0].copy()

# Create binary labels: 1=Fall, 0=ADL based on source
df['is_fall'] = (df['source'] == 'fall').astype(int)

# Select 4 key features from paper
# Note: We need to convert to YOLO-pose equivalents
# HW_Ratio (h/w) -> inverse for w/h
# HHmax -> height_ratio (same)
# D -> floor_distance (needs normalization)
# Std -> MaxStdXZ (spatial spread)

df['wh_ratio'] = 1.0 / (df['HW_Ratio'] + 1e-6)  # w/h
df['height_ratio'] = df['HHmax']  # h/hmax
df['floor_distance_proxy'] = 1.0 - (df['D'] / 1500.0)  # Normalize D (max ~1500mm)
df['spatial_std'] = df['Std'] / 300.0  # Normalize Std

# Feature matrix
feature_cols = ['wh_ratio', 'height_ratio', 'floor_distance_proxy', 'spatial_std']
X = df[feature_cols].values
y = df['is_fall'].values

print(f"\nDataset: {len(X)} samples")
print(f"  Falls: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
print(f"  ADLs: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")

# Split by sequence (not random) to avoid data leakage
# Use 80% sequences for train, 20% for test
sequences = df['seq'].unique()
np.random.seed(42)
np.random.shuffle(sequences)
split_idx = int(len(sequences) * 0.8)
train_seqs = sequences[:split_idx]
test_seqs = sequences[split_idx:]

train_mask = df['seq'].isin(train_seqs)
test_mask = df['seq'].isin(test_seqs)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\nTrain: {len(X_train)} samples, Test: {len(X_test)} samples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear SVM with Grid Search
print("\nTraining Linear SVM...")
param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'class_weight': ['balanced', None]
}

svm = LinearSVC(max_iter=10000, random_state=42, dual=False)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', verbose=1)
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV F1-score: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)

print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['ADL', 'Fall']))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]} (ADL correct), FP={cm[0,1]} (ADL as Fall)")
print(f"  FN={cm[1,0]} (Fall missed), TP={cm[1,1]} (Fall detected)")

# Extract SVM coefficients
weights = best_svm.coef_[0]
intercept = best_svm.intercept_[0]

print("\n" + "="*60)
print("SVM DECISION FUNCTION")
print("="*60)
print("Standardized weights:")
for i, col in enumerate(feature_cols):
    print(f"  {col}: {weights[i]:.4f}")
print(f"  Intercept: {intercept:.4f}")

# Convert to original scale
# decision = w1*x1 + w2*x2 + w3*x3 + w4*x4 + b
# After standardization: decision = w1*(x1-mean1)/std1 + ...
# Original: decision = (w1/std1)*x1 - (w1*mean1/std1) + ...
original_weights = weights / scaler.scale_
original_intercept = intercept - np.sum(weights * scaler.mean_ / scaler.scale_)

print("\nOriginal scale weights (for rules.py):")
for i, col in enumerate(feature_cols):
    print(f"  {col}: {original_weights[i]:.6f}")
print(f"  Intercept: {original_intercept:.6f}")

# Save model
print("\nSaving model...")
with open('outputs/svm_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_svm,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'weights': weights,
        'intercept': intercept,
        'original_weights': original_weights,
        'original_intercept': original_intercept
    }, f)

# Save config for rules.py
config = {
    'svm_weights': {
        'wh_ratio': float(original_weights[0]),
        'height_ratio': float(original_weights[1]),
        'floor_distance': float(original_weights[2]),
        'spatial_std': float(original_weights[3])
    },
    'svm_intercept': float(original_intercept),
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist()
}

with open('outputs/svm_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\nModel saved to outputs/svm_model.pkl")
print("Config saved to outputs/svm_config.json")
print("\nNext: Update rules.py to use these learned weights!")

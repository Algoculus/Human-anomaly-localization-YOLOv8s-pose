"""
Train Linear SVM with ONLY 3 features that YOLO can provide:
- wh_ratio (w/h from bbox)
- height_ratio (h/hmax from bbox)
- floor_distance (cy/img_h from bbox)

Excludes MaxStdXZ since it requires depth point cloud.
"""
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json

print("Loading Ground Truth features...")
cols = ['seq', 'frame', 'label', 'HW_Ratio', 'MM_Ratio', 'Occ', 'Std', 'HHmax', 'H', 'D', 'P40']
df_falls = pd.read_csv('../data/00_raw_download/extracted_features/urfall-cam0-falls.csv', 
                       header=None, names=cols)
df_adls = pd.read_csv('../data/00_raw_download/extracted_features/urfall-cam0-adls.csv', 
                      header=None, names=cols)

df_falls['source'] = 'fall'
df_adls['source'] = 'adl'
df = pd.concat([df_falls, df_adls], ignore_index=True)

# Filter: Use all valid frames (lying=1, normal=-1)
# Exclude transition frames (label=0)
df = df[df['label'] != 0].copy()

# Create binary labels: 
# 1 = "Fall Lying" (The state we want to detect)
# 0 = Everything else (Standing, ADL Lying, ADL Standing)
df['is_fall'] = 0
mask_fall_lying = (df['source'] == 'fall') & (df['label'] == 1)
df.loc[mask_fall_lying, 'is_fall'] = 1

print(f"Class 1 (Fall Lying): {df['is_fall'].sum()}")
print(f"Class 0 (Others): {len(df) - df['is_fall'].sum()}")

# ONLY 3 features that YOLO can compute
# Note: YOLO hw_ratio = w/h. Data HW_Ratio = h/w.
# We use YOLO format (w/h) for consistency.
df['wh_ratio'] = 1.0 / (df['HW_Ratio'] + 1e-6)
df['height_ratio'] = df['HHmax']
df['floor_distance_proxy'] = 1.0 - (df['D'] / 1500.0) # Proxy for cy/img_h

feature_cols = ['wh_ratio', 'height_ratio', 'floor_distance_proxy']
X = df[feature_cols].values
y = df['is_fall'].values

print(f"\nDataset: {len(X)} samples, {sum(y)} Falls, {len(y)-sum(y)} ADLs")

# Sequence-based split
sequences = df['seq'].unique()
np.random.seed(42)
np.random.shuffle(sequences)
split_idx = int(len(sequences) * 0.8)
train_seqs, test_seqs = sequences[:split_idx], sequences[split_idx:]

train_mask = df['seq'].isin(train_seqs)
test_mask = df['seq'].isin(test_seqs)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
print("\nTraining Linear SVM (3 features)...")
param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'class_weight': ['balanced', None]
}

svm = LinearSVC(max_iter=10000, random_state=42, dual=False)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', verbose=1)
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV F1: {grid_search.best_score_:.4f}")

# Evaluate
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)

print("\n" + "="*60)
print("TEST SET EVALUATION (3 Features)")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['ADL', 'Fall']))
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

# Extract weights
weights = best_svm.coef_[0]
intercept = best_svm.intercept_[0]

print("\n" + "="*60)
print("SVM WEIGHTS (Standardized)")
print("="*60)
for i, col in enumerate(feature_cols):
    print(f"  {col}: {weights[i]:.4f}")
print(f"  Intercept: {intercept:.4f}")

# Original scale
original_weights = weights / scaler.scale_
original_intercept = intercept - np.sum(weights * scaler.mean_ / scaler.scale_)

print("\nOriginal Scale Weights:")
for i, col in enumerate(feature_cols):
    print(f"  {col}: {original_weights[i]:.6f}")
print(f"  Intercept: {original_intercept:.6f}")

# Save
config = {
    'svm_weights': {
        'wh_ratio': float(original_weights[0]),
        'height_ratio': float(original_weights[1]),
        'floor_distance': float(original_weights[2])
    },
    'svm_intercept': float(original_intercept),
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'num_features': 3
}

with open('outputs/svm_config_3feat.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\nSaved to outputs/svm_config_3feat.json")

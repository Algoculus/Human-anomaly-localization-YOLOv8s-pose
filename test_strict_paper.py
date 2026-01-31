
import sys
import os
sys.path.insert(0, os.getcwd())
from src.urfd.rules import FallScorer

# Mock config for strict cam0 mode
config = {
    "weights": {"impact": 0.2, "shape": 0.6, "pose": 0.2},
    "paper_features": {
        "hw_ratio_thres": 1.2,
        "height_ratio_thres": 0.55,
        "floor_distance_thres": 0.75,
        "variance_thres": 0.04,
        "enable_pose_features": False  # STRICT MODE
    },
    "thresholds": {}
}

scorer = FallScorer(config)

print(f"Scorer Strict Mode (enable_pose_features): {scorer.enable_pose_features}")

# Test Case 1: CLEAR FALL (Lying, Low, Floor, High Variance)
fall_feat = {
    "bbox": (100, 300, 400, 450), # 300x150 -> hw=2.0
    "hw_ratio": 2.0,
    "height_ratio": 0.4,
    "floor_distance": 0.85,
    "position_variance": 0.06,
    "dy_peak": 8.0, 
    "impact_signature": 0.8 # Ignored in strict mode
}

res_fall = scorer.compute_fall_confidence(fall_feat)
print(f"\nTest 1 (FALL):")
print(f"  Confidence: {res_fall.confidence:.4f}")
print(f"  Contributions: {[f'{c.name}={c.score:.2f}' for c in res_fall.contributions]}")

# Test Case 2: CLEAR ADL (Standing, Normal, Low Variance)
adl_feat = {
    "bbox": (100, 100, 200, 400), # 100x300 -> hw=0.33
    "hw_ratio": 0.33,
    "height_ratio": 0.95,
    "floor_distance": 0.4,
    "position_variance": 0.01,
}

res_adl = scorer.compute_fall_confidence(adl_feat)
print(f"\nTest 2 (ADL):")
print(f"  Confidence: {res_adl.confidence:.4f}")

# Test Case 3: BORDERLINE (Sitting? hw=1.0, height=0.6, floor=0.6)
# Should NOT trigger if strict
border_feat = {
    "bbox": (100, 200, 250, 350), # 150x150 -> hw=1.0
    "hw_ratio": 1.0,
    "height_ratio": 0.65,
    "floor_distance": 0.60,
    "position_variance": 0.02
}

res_border = scorer.compute_fall_confidence(border_feat)
print(f"\nTest 3 (BORDERLINE):")
print(f"  Confidence: {res_border.confidence:.4f}")

assert res_fall.confidence > 0.8, "Fall confidence too low"
assert res_adl.confidence < 0.2, "ADL confidence too high"
assert res_border.confidence < 0.5, "Borderline case should be negative"

print("\n✅ Strict Paper Logic Verified!")

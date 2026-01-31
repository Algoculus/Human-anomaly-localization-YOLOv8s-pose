"""
Paper Features Loader - URFD Pre-Extracted Features

Loads the pre-computed features from the URFD dataset CSV files:
- urfall-cam0-falls.csv
- urfall-cam0-adls.csv

CSV Columns (based on URFD paper):
[0] seq_name: e.g., "fall-01", "adl-01"
[1] frame_idx: Frame number
[2] label: -1 (unknown), 0 (normal), 1 (fall)
[3] HeightWidthRatio: h/w ratio
[4] MajorMinorRatio: Ellipse axis ratio
[5] BoundingBoxOccupancyRatio: Person area / bbox area
[6] MaxStdXZ: Max standard deviation in XZ plane (KEY FEATURE)
[7] HHmaxRatio: Current height / Max height
[8] H: Bounding box height
[9] D: Depth distance
[10] MaxStdXZ (repeated or alternative measure)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class PaperFeaturesLoader:
    """Loader for URFD paper's pre-extracted features."""
    
    COLUMN_NAMES = [
        'seq_name', 'frame_idx', 'label',
        'HeightWidthRatio', 'MajorMinorRatio', 'BoundingBoxOccupancyRatio',
        'MaxStdXZ', 'HHmaxRatio', 'H', 'D', 'MaxStdXZ_alt'
    ]
    
    def __init__(self, data_root: Path):
        """
        Initialize loader.
        
        Args:
            data_root: Path to data directory (contains 00_raw_download)
        """
        self.data_root = Path(data_root)
        self.features_dir = self.data_root / "00_raw_download" / "extracted_features"
        
        self.fall_features: Optional[pd.DataFrame] = None
        self.adl_features: Optional[pd.DataFrame] = None
        self._loaded = False
        
    def load(self) -> bool:
        """Load both CSV files."""
        falls_path = self.features_dir / "urfall-cam0-falls.csv"
        adls_path = self.features_dir / "urfall-cam0-adls.csv"
        
        if not falls_path.exists() or not adls_path.exists():
            print(f"Warning: Paper features not found at {self.features_dir}")
            return False
        
        try:
            self.fall_features = pd.read_csv(falls_path, header=None, names=self.COLUMN_NAMES)
            self.adl_features = pd.read_csv(adls_path, header=None, names=self.COLUMN_NAMES)
            self._loaded = True
            print(f"Loaded paper features: {len(self.fall_features)} fall frames, {len(self.adl_features)} ADL frames")
            return True
        except Exception as e:
            print(f"Error loading paper features: {e}")
            return False
    
    def get_sequence_features(self, seq_name: str) -> Optional[pd.DataFrame]:
        """
        Get all features for a sequence.
        
        Args:
            seq_name: e.g., "fall-01", "adl-01"
            
        Returns:
            DataFrame with features for all frames, or None if not found
        """
        if not self._loaded:
            self.load()
        
        if seq_name.startswith('fall'):
            df = self.fall_features
        else:
            df = self.adl_features
            
        if df is None:
            return None
            
        seq_df = df[df['seq_name'] == seq_name].copy()
        return seq_df if len(seq_df) > 0 else None
    
    def get_frame_features(self, seq_name: str, frame_idx: int) -> Optional[Dict]:
        """
        Get features for a specific frame.
        
        Args:
            seq_name: Sequence name
            frame_idx: Frame index
            
        Returns:
            Dict with paper features, or None if not found
        """
        seq_df = self.get_sequence_features(seq_name)
        if seq_df is None:
            return None
        
        # Find closest frame (paper CSV may have different frame numbering)
        frame_df = seq_df[seq_df['frame_idx'] == frame_idx]
        
        if len(frame_df) == 0:
            # Try approximate match
            closest_idx = (seq_df['frame_idx'] - frame_idx).abs().idxmin()
            frame_df = seq_df.loc[[closest_idx]]
        
        if len(frame_df) == 0:
            return None
            
        row = frame_df.iloc[0]
        return {
            'label': int(row['label']),
            'HeightWidthRatio': float(row['HeightWidthRatio']),
            'MajorMinorRatio': float(row['MajorMinorRatio']),
            'BoundingBoxOccupancyRatio': float(row['BoundingBoxOccupancyRatio']),
            'MaxStdXZ': float(row['MaxStdXZ']),
            'HHmaxRatio': float(row['HHmaxRatio']),
            'H': float(row['H']),
            'D': float(row['D'])
        }
    
    def get_max_std_xz(self, seq_name: str, frame_idx: int) -> float:
        """
        Get MaxStdXZ for a frame (paper's key fall indicator).
        
        Returns 0.0 if not available.
        """
        features = self.get_frame_features(seq_name, frame_idx)
        if features is None:
            return 0.0
        return features.get('MaxStdXZ', 0.0)


class SyncLabelsLoader:
    """Loader for frame-level ground truth from sync_csv."""
    
    def __init__(self, data_root: Path):
        """
        Args:
            data_root: Path to data directory
        """
        self.data_root = Path(data_root)
        self.fall_sync_dir = self.data_root / "00_raw_download" / "falls" / "sync_csv"
        self.adl_sync_dir = self.data_root / "00_raw_download" / "adl" / "sync_csv"
        
    def load_sequence_labels(self, seq_name: str) -> Dict[int, int]:
        """
        Load frame-level labels for a sequence.
        
        Args:
            seq_name: e.g., "fall-01"
            
        Returns:
            Dict {frame_idx: label} where label is 0 (normal) or 1 (fall)
        """
        if seq_name.startswith('fall'):
            sync_dir = self.fall_sync_dir
            filename = f"{seq_name}-data.csv"
        else:
            sync_dir = self.adl_sync_dir
            filename = f"{seq_name}-data.csv"
            
        sync_path = sync_dir / filename
        
        if not sync_path.exists():
            # For ADL sequences, all frames are normal
            return {}
        
        try:
            # sync_csv format: frame_idx, timestamp, label
            df = pd.read_csv(sync_path, header=None, names=['frame_idx', 'timestamp', 'label'])
            
            # Convert to dict
            labels = {}
            for _, row in df.iterrows():
                frame_idx = int(row['frame_idx'])
                # Label: 1 if fall, 0 if normal
                # In sync_csv, label > 0 means fall
                label = 1 if float(row['label']) > 0.5 else 0
                labels[frame_idx] = label
                
            return labels
        except Exception as e:
            print(f"Error loading sync labels for {seq_name}: {e}")
            return {}
    
    def get_fall_start_frame(self, seq_name: str) -> int:
        """
        Get the frame where fall starts (first label=1).
        
        Returns -1 if no fall or not found.
        """
        labels = self.load_sequence_labels(seq_name)
        for frame_idx in sorted(labels.keys()):
            if labels[frame_idx] == 1:
                return frame_idx
        return -1


def load_all_paper_features(data_root: Path) -> pd.DataFrame:
    """
    Load all paper features as a single DataFrame for training.
    
    Returns:
        DataFrame with columns: seq_name, frame_idx, label, features...
    """
    loader = PaperFeaturesLoader(data_root)
    loader.load()
    
    all_features = pd.concat([
        loader.fall_features,
        loader.adl_features
    ], ignore_index=True)
    
    return all_features

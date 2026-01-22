"""
URFall Dataset Loader

Comprehensive data loading utilities for the UR Fall Detection Dataset.
Handles RGB frames, depth images, sync data, accelerometer data, and ground truth labels.

Ground Truth Labels (from annotation CSV):
- label = -1: not lying (negative class)
- label = 1: lying (positive class)
- label = 0: transition frame (EXCLUDE from evaluation)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Union

import numpy as np
import pandas as pd
import cv2
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import Config, get_config


# =========================
# DATA CLASSES
# =========================

@dataclass
class FrameData:
    """Data for a single frame."""

    sequence_id: str
    frame_number: int  # 1-indexed
    timestamp_ms: Optional[float] = None

    rgb_path: Optional[Path] = None
    depth_path: Optional[Path] = None

    label: Optional[int] = None  # -1, 0, 1
    depth_features: Optional[Dict[str, float]] = None

    # Accelerometer (aligned to frame timestamp)
    acc_sv_total: Optional[float] = None
    acc_x: Optional[float] = None
    acc_y: Optional[float] = None
    acc_z: Optional[float] = None

    # cache
    _rgb_image: Optional[np.ndarray] = field(default=None, repr=False)
    _depth_image: Optional[np.ndarray] = field(default=None, repr=False)

    def load_rgb(self) -> Optional[np.ndarray]:
        """Load RGB image from disk (BGR via OpenCV)."""
        if self._rgb_image is not None:
            return self._rgb_image
        if self.rgb_path is None or not self.rgb_path.exists():
            return None
        img = cv2.imread(str(self.rgb_path))
        self._rgb_image = img
        return img

    def load_depth(self, depth_scale: float = 6000.0) -> Optional[np.ndarray]:
        """
        Load and rescale depth image from disk.

        Depth PNG16 => d_mm = (C * pixel_value) / 65535
        """
        if self._depth_image is not None:
            return self._depth_image
        if self.depth_path is None or not self.depth_path.exists():
            return None

        depth_raw = cv2.imread(str(self.depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            return None

        self._depth_image = (depth_scale * depth_raw.astype(np.float32)) / 65535.0
        return self._depth_image

    def clear_cache(self):
        self._rgb_image = None
        self._depth_image = None

    @property
    def is_valid_for_eval(self) -> bool:
        return self.label is not None and self.label != 0


@dataclass
class CameraData:
    camera_id: str  # cam0, cam1
    frames: List[FrameData] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def get_frame(self, frame_number: int) -> Optional[FrameData]:
        # Faster lookup could be dict, but list OK for dataset size
        for f in self.frames:
            if f.frame_number == frame_number:
                return f
        return None


@dataclass
class SequenceData:
    sequence_id: str
    sequence_type: str  # fall / adl
    sequence_number: int

    cam0: Optional[CameraData] = None
    cam1: Optional[CameraData] = None

    sync_data: Optional[pd.DataFrame] = None
    acc_data: Optional[pd.DataFrame] = None
    annotations: Optional[pd.DataFrame] = None

    @property
    def has_cam1(self) -> bool:
        return self.cam1 is not None and self.cam1.num_frames > 0

    @property
    def num_frames(self) -> int:
        return self.cam0.num_frames if self.cam0 else 0

    def get_label_distribution(self) -> Dict[int, int]:
        if self.annotations is None:
            return {}
        return self.annotations["label"].value_counts().to_dict()


# =========================
# DATASET WRAPPER
# =========================

class URFallDataset:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.paths = self.config.paths
        self.dataset_config = self.config.dataset

        self.fall_sequences: List[SequenceData] = []
        self.adl_sequences: List[SequenceData] = []

        self._falls_annotations: Optional[pd.DataFrame] = None
        self._adls_annotations: Optional[pd.DataFrame] = None

        logger.info(f"URFall Dataset initialized with data root: {self.paths.data_root}")

    # ---------- Annotations ----------

    def _load_annotations(self):
        falls_csv = self.paths.annotations / "urfall-cam0-falls.csv"
        adls_csv = self.paths.annotations / "urfall-cam0-adls.csv"

        col_names = [
            "sequence", "frame_number", "label",
            "HeightWidthRatio", "MajorMinorRatio", "BoundingBoxOccupancy",
            "MaxStdXZ", "HHmaxRatio", "H", "D", "P40"
        ]

        if falls_csv.exists():
            self._falls_annotations = pd.read_csv(falls_csv, header=None, names=col_names)
            logger.info(f"Loaded {len(self._falls_annotations)} fall annotations")
        else:
            logger.warning(f"Falls annotation file not found: {falls_csv}")

        if adls_csv.exists():
            self._adls_annotations = pd.read_csv(adls_csv, header=None, names=col_names)
            logger.info(f"Loaded {len(self._adls_annotations)} ADL annotations")
        else:
            logger.warning(f"ADL annotation file not found: {adls_csv}")

    def _get_sequence_annotations(self, sequence_id: str) -> Optional[pd.DataFrame]:
        if sequence_id.startswith("fall") and self._falls_annotations is not None:
            return self._falls_annotations[self._falls_annotations["sequence"] == sequence_id].copy()
        if sequence_id.startswith("adl") and self._adls_annotations is not None:
            return self._adls_annotations[self._adls_annotations["sequence"] == sequence_id].copy()
        return None

    # ---------- Sync + ACC ----------

    def _load_sync_data(self, sequence_id: str, sequence_type: str) -> Optional[pd.DataFrame]:
        """
        Sync data:
        - falls: frame_number, time_ms, sv_total
        - adl:  frame_number, time_ms (sometimes no sv_total)
        """
        if sequence_type == "fall":
            sync_dir = self.paths.extracted_frames / "falls" / sequence_id / "sync"
            if not sync_dir.exists():
                sync_dir = self.paths.raw_download / "falls" / "sync_csv"
            sync_file = sync_dir / f"{sequence_id}-data.csv"
        else:
            sync_dir = self.paths.extracted_frames / "adl" / sequence_id / "sync"
            if not sync_dir.exists():
                sync_dir = self.paths.raw_download / "adl" / "sync_csv"
            sync_file = sync_dir / f"{sequence_id}-data.csv"

        if not sync_file.exists():
            logger.debug(f"Sync file not found: {sync_file}")
            return None

        try:
            df = pd.read_csv(sync_file, header=None)
            if df.shape[1] == 2:
                df.columns = ["frame_number", "time_ms"]
            elif df.shape[1] == 3:
                df.columns = ["frame_number", "time_ms", "sv_total"]
            else:
                logger.warning(f"Unexpected sync format in {sync_file}: {df.shape[1]} columns")
                return None

            # Ensure numeric
            df["frame_number"] = df["frame_number"].astype(int)
            df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
            if "sv_total" in df.columns:
                df["sv_total"] = pd.to_numeric(df["sv_total"], errors="coerce")

            df = df.dropna(subset=["time_ms"])
            return df
        except Exception as e:
            logger.error(f"Error loading sync data {sync_file}: {e}")
            return None

    def _load_acc_data(self, sequence_id: str, sequence_type: str) -> Optional[pd.DataFrame]:
        """
        ACC format: time_ms, sv_total, ax, ay, az
        """
        if sequence_type == "fall":
            acc_dir = self.paths.extracted_frames / "falls" / sequence_id / "acc"
            if not acc_dir.exists():
                acc_dir = self.paths.raw_download / "falls" / "acc_csv"
            acc_file = acc_dir / f"{sequence_id}-acc.csv"
        else:
            acc_dir = self.paths.extracted_frames / "adl" / sequence_id / "acc"
            if not acc_dir.exists():
                acc_dir = self.paths.raw_download / "adl" / "acc_csv"
            acc_file = acc_dir / f"{sequence_id}-acc.csv"

        if not acc_file.exists():
            logger.debug(f"ACC file not found: {acc_file}")
            return None

        try:
            df = pd.read_csv(acc_file, header=None, names=["time_ms", "sv_total", "ax", "ay", "az"])
            df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
            for c in ["sv_total", "ax", "ay", "az"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["time_ms"])
            return df
        except Exception as e:
            logger.error(f"Error loading acc data {acc_file}: {e}")
            return None

    def _interpolate_acc_to_frames(
        self,
        sync_df: Optional[pd.DataFrame],
        acc_df: Optional[pd.DataFrame],
        tolerance_ms: float = 80.0
    ) -> Dict[int, Dict[str, float]]:
        """
        Align accelerometer to frame timestamps by nearest neighbor with tolerance.
        Returns: frame_number -> acc dict
        """
        if sync_df is None or acc_df is None or sync_df.empty or acc_df.empty:
            return {}

        acc_times = acc_df["time_ms"].values.astype(float)
        result: Dict[int, Dict[str, float]] = {}

        for _, row in sync_df.iterrows():
            frame_num = int(row["frame_number"])
            t = float(row["time_ms"])

            idx = int(np.argmin(np.abs(acc_times - t)))
            if abs(acc_times[idx] - t) > tolerance_ms:
                # too far -> skip to avoid wrong alignment
                continue

            r = acc_df.iloc[idx]
            result[frame_num] = {
                "sv_total": float(r["sv_total"]) if pd.notna(r["sv_total"]) else None,
                "ax": float(r["ax"]) if pd.notna(r["ax"]) else None,
                "ay": float(r["ay"]) if pd.notna(r["ay"]) else None,
                "az": float(r["az"]) if pd.notna(r["az"]) else None,
            }

        return result

    # ---------- Frame discovery ----------

    def _find_rgb_frames(self, sequence_id: str, sequence_type: str, camera: str) -> List[Tuple[int, Path]]:
        if sequence_type == "fall":
            base_dir = self.paths.extracted_frames / "falls" / sequence_id
        else:
            base_dir = self.paths.extracted_frames / "adl" / sequence_id

        rgb_dir = base_dir / camera / "rgb"
        if not rgb_dir.exists():
            logger.debug(f"RGB directory not found: {rgb_dir}")
            return []

        subdirs = [p for p in rgb_dir.iterdir() if p.is_dir()]
        if subdirs:
            rgb_dir = subdirs[0]

        frames: List[Tuple[int, Path]] = []
        pattern = re.compile(rf"{sequence_id}-{camera}-rgb-(\d+)\.png", re.IGNORECASE)

        for p in rgb_dir.glob("*.png"):
            m = pattern.match(p.name)
            if m:
                frames.append((int(m.group(1)), p))

        frames.sort(key=lambda x: x[0])
        return frames

    def _find_depth_frames(self, sequence_id: str, sequence_type: str, camera: str) -> Dict[int, Path]:
        if sequence_type == "fall":
            base_dir = self.paths.extracted_frames / "falls" / sequence_id
        else:
            base_dir = self.paths.extracted_frames / "adl" / sequence_id

        depth_dir = base_dir / camera / "depth_png16"
        if not depth_dir.exists():
            return {}

        subdirs = [p for p in depth_dir.iterdir() if p.is_dir()]
        if subdirs:
            depth_dir = subdirs[0]

        pattern = re.compile(rf"{sequence_id}-{camera}-d-(\d+)\.png", re.IGNORECASE)
        depth_frames: Dict[int, Path] = {}

        for p in depth_dir.glob("*.png"):
            m = pattern.match(p.name)
            if m:
                depth_frames[int(m.group(1))] = p

        return depth_frames

    # ---------- Sequence loader ----------

    def load_sequence(self, sequence_id: str) -> Optional[SequenceData]:
        m = re.match(r"(fall|adl)-(\d+)", sequence_id)
        if not m:
            logger.error(f"Invalid sequence ID: {sequence_id}")
            return None

        sequence_type = m.group(1)
        sequence_number = int(m.group(2))

        if self._falls_annotations is None and self._adls_annotations is None:
            self._load_annotations()

        annotations = self._get_sequence_annotations(sequence_id)
        sync_data = self._load_sync_data(sequence_id, sequence_type)
        acc_data = self._load_acc_data(sequence_id, sequence_type)

        acc_by_frame = self._interpolate_acc_to_frames(sync_data, acc_data, tolerance_ms=80.0)

        # Sync lookup
        sync_lookup: Dict[int, Dict[str, Union[float, None]]] = {}
        if sync_data is not None and not sync_data.empty:
            for _, row in sync_data.iterrows():
                fn = int(row["frame_number"])
                sync_lookup[fn] = {
                    "time_ms": float(row["time_ms"]),
                    "sv_total": float(row["sv_total"]) if ("sv_total" in sync_data.columns and pd.notna(row.get("sv_total"))) else None
                }

        # Annotation lookup
        ann_lookup: Dict[int, Dict] = {}
        if annotations is not None and not annotations.empty:
            for _, row in annotations.iterrows():
                fn = int(row["frame_number"])
                ann_lookup[fn] = {
                    "label": int(row["label"]),
                    "depth_features": {
                        "HeightWidthRatio": float(row["HeightWidthRatio"]) if pd.notna(row.get("HeightWidthRatio")) else None,
                        "MajorMinorRatio": float(row["MajorMinorRatio"]) if pd.notna(row.get("MajorMinorRatio")) else None,
                        "BoundingBoxOccupancy": float(row["BoundingBoxOccupancy"]) if pd.notna(row.get("BoundingBoxOccupancy")) else None,
                        "MaxStdXZ": float(row["MaxStdXZ"]) if pd.notna(row.get("MaxStdXZ")) else None,
                        "HHmaxRatio": float(row["HHmaxRatio"]) if pd.notna(row.get("HHmaxRatio")) else None,
                        "H": float(row["H"]) if pd.notna(row.get("H")) else None,
                        "D": float(row["D"]) if pd.notna(row.get("D")) else None,
                        "P40": float(row["P40"]) if pd.notna(row.get("P40")) else None,
                    }
                }

        # cam0
        cam0_rgb = self._find_rgb_frames(sequence_id, sequence_type, "cam0")
        cam0_depth = self._find_depth_frames(sequence_id, sequence_type, "cam0")

        cam0_frames: List[FrameData] = []
        for fn, rgb_path in cam0_rgb:
            sync_info = sync_lookup.get(fn, {})
            ann_info = ann_lookup.get(fn, {})
            acc_info = acc_by_frame.get(fn, {})

            # Prefer acc sv_total if available; fallback to sync sv_total
            sv_total = acc_info.get("sv_total", None)
            if sv_total is None:
                sv_total = sync_info.get("sv_total", None)

            cam0_frames.append(
                FrameData(
                    sequence_id=sequence_id,
                    frame_number=fn,
                    timestamp_ms=sync_info.get("time_ms"),
                    rgb_path=rgb_path,
                    depth_path=cam0_depth.get(fn),
                    label=ann_info.get("label"),
                    depth_features=ann_info.get("depth_features"),
                    acc_sv_total=sv_total,
                    acc_x=acc_info.get("ax"),
                    acc_y=acc_info.get("ay"),
                    acc_z=acc_info.get("az"),
                )
            )

        cam0 = CameraData(camera_id="cam0", frames=cam0_frames)

        # cam1 (falls only)
        cam1 = None
        if sequence_type == "fall":
            cam1_rgb = self._find_rgb_frames(sequence_id, sequence_type, "cam1")
            cam1_depth = self._find_depth_frames(sequence_id, sequence_type, "cam1")
            if cam1_rgb:
                cam1_frames: List[FrameData] = []
                for fn, rgb_path in cam1_rgb:
                    sync_info = sync_lookup.get(fn, {})
                    acc_info = acc_by_frame.get(fn, {})

                    sv_total = acc_info.get("sv_total", None)
                    if sv_total is None:
                        sv_total = sync_info.get("sv_total", None)

                    cam1_frames.append(
                        FrameData(
                            sequence_id=sequence_id,
                            frame_number=fn,
                            timestamp_ms=sync_info.get("time_ms"),
                            rgb_path=rgb_path,
                            depth_path=cam1_depth.get(fn),
                            label=None,  # cam1 doesn't have labels
                            depth_features=None,
                            acc_sv_total=sv_total,
                            acc_x=acc_info.get("ax"),
                            acc_y=acc_info.get("ay"),
                            acc_z=acc_info.get("az"),
                        )
                    )
                cam1 = CameraData(camera_id="cam1", frames=cam1_frames)

        seq = SequenceData(
            sequence_id=sequence_id,
            sequence_type=sequence_type,
            sequence_number=sequence_number,
            cam0=cam0,
            cam1=cam1,
            sync_data=sync_data,
            acc_data=acc_data,
            annotations=annotations,
        )

        logger.debug(
            f"Loaded {sequence_id}: {cam0.num_frames} cam0 frames"
            + (f", {cam1.num_frames} cam1 frames" if cam1 else "")
        )
        return seq

    # ---------- Bulk loading ----------

    def load_all(self, load_falls: bool = True, load_adls: bool = True, use_cache: bool = False):
        """
        NOTE: default use_cache=False to avoid pickle issues across environments.
        """
        logger.info("Loading all URFall sequences...")

        self._load_annotations()

        if load_falls:
            self.fall_sequences = []
            for i in range(1, self.dataset_config.num_fall_sequences + 1):
                sid = f"fall-{i:02d}"
                seq = self.load_sequence(sid)
                if seq is not None:
                    self.fall_sequences.append(seq)
            logger.info(f"Loaded {len(self.fall_sequences)} fall sequences")

        if load_adls:
            self.adl_sequences = []
            for i in range(1, self.dataset_config.num_adl_sequences + 1):
                sid = f"adl-{i:02d}"
                seq = self.load_sequence(sid)
                if seq is not None:
                    self.adl_sequences.append(seq)
            logger.info(f"Loaded {len(self.adl_sequences)} ADL sequences")

    @property
    def all_sequences(self) -> List[SequenceData]:
        return self.fall_sequences + self.adl_sequences

    def get_train_val_test_split(self, random_seed: int = 42) -> Tuple[List[SequenceData], List[SequenceData], List[SequenceData]]:
        np.random.seed(random_seed)

        train_ratio = self.dataset_config.train_ratio
        val_ratio = self.dataset_config.val_ratio

        def split_list(seqs: List[SequenceData]) -> Tuple[List[SequenceData], List[SequenceData], List[SequenceData]]:
            n = len(seqs)
            idx = np.random.permutation(n)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            train_idx = idx[:train_end]
            val_idx = idx[train_end:val_end]
            test_idx = idx[val_end:]
            return ([seqs[i] for i in train_idx], [seqs[i] for i in val_idx], [seqs[i] for i in test_idx])

        fall_train, fall_val, fall_test = split_list(self.fall_sequences)
        adl_train, adl_val, adl_test = split_list(self.adl_sequences)

        train = fall_train + adl_train
        val = fall_val + adl_val
        test = fall_test + adl_test

        logger.info(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")
        return train, val, test

    def get_all_frames_for_eval(self, camera: str = "cam0") -> Iterator[FrameData]:
        for seq in self.all_sequences:
            cam = seq.cam0 if camera == "cam0" else seq.cam1
            if cam is None:
                continue
            for f in cam.frames:
                if f.is_valid_for_eval:
                    yield f

    def get_statistics(self) -> Dict:
        stats = {
            "num_fall_sequences": len(self.fall_sequences),
            "num_adl_sequences": len(self.adl_sequences),
            "total_sequences": len(self.all_sequences),
            "fall_frames_cam0": sum(s.cam0.num_frames for s in self.fall_sequences if s.cam0),
            "fall_frames_cam1": sum(s.cam1.num_frames for s in self.fall_sequences if s.cam1),
            "adl_frames_cam0": sum(s.cam0.num_frames for s in self.adl_sequences if s.cam0),
            "label_distribution": {"not_lying": 0, "lying": 0, "transition": 0},
        }

        for s in self.all_sequences:
            if s.cam0:
                for f in s.cam0.frames:
                    if f.label == -1:
                        stats["label_distribution"]["not_lying"] += 1
                    elif f.label == 1:
                        stats["label_distribution"]["lying"] += 1
                    elif f.label == 0:
                        stats["label_distribution"]["transition"] += 1

        return stats


# =========================
# Convenience functions
# =========================

def load_sequence(sequence_id: str, config: Optional[Config] = None) -> Optional[SequenceData]:
    dataset = URFallDataset(config)
    dataset._load_annotations()
    return dataset.load_sequence(sequence_id)


def load_all_sequences(config: Optional[Config] = None) -> URFallDataset:
    dataset = URFallDataset(config)
    dataset.load_all()
    return dataset


def get_train_val_test_split(config: Optional[Config] = None, random_seed: int = 42) -> Tuple[List, List, List]:
    dataset = load_all_sequences(config)
    return dataset.get_train_val_test_split(random_seed)

if __name__ == "__main__":
    # Test the data loader
    from loguru import logger
    import sys
    
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    
    print("Testing URFall Data Loader...")
    print("=" * 50)
    
    # Load a single fall sequence
    seq = load_sequence("fall-01")
    if seq:
        print(f"\nLoaded: {seq.sequence_id}")
        print(f"  Type: {seq.sequence_type}")
        print(f"  Cam0 frames: {seq.cam0.num_frames if seq.cam0 else 0}")
        print(f"  Cam1 frames: {seq.cam1.num_frames if seq.cam1 else 0}")
        print(f"  Label distribution: {seq.get_label_distribution()}")
        
        # Check first frame
        if seq.cam0 and seq.cam0.frames:
            frame = seq.cam0.frames[0]
            print(f"\n  First frame:")
            print(f"    Frame number: {frame.frame_number}")
            print(f"    Timestamp: {frame.timestamp_ms} ms")
            print(f"    Label: {frame.label}")
            print(f"    RGB path: {frame.rgb_path}")
            print(f"    Depth path: {frame.depth_path}")
            print(f"    Acc SV_total: {frame.acc_sv_total}")
    
    # Load an ADL sequence
    seq_adl = load_sequence("adl-01")
    if seq_adl:
        print(f"\nLoaded: {seq_adl.sequence_id}")
        print(f"  Type: {seq_adl.sequence_type}")
        print(f"  Cam0 frames: {seq_adl.cam0.num_frames if seq_adl.cam0 else 0}")
        print(f"  Label distribution: {seq_adl.get_label_distribution()}")
    
    print("\n" + "=" * 50)
    print("Data loader test complete!")

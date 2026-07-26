#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IARC Viewer & Extractor v1.0
Standalone viewer and programmatic API for .iarc archives.

STANDALONE USAGE:
    python iarc_viewer.py
    python iarc_viewer.py path/to/archive.iarc
    python iarc_viewer.py path/to/archive.iarc --extract-all output_folder/
    python iarc_viewer.py path/to/archive.iarc --extract 42 output.png
    python iarc_viewer.py path/to/archive.iarc --list

PROGRAMMATIC USAGE:
    from iarc_viewer import IARCReader

    reader = IARCReader("archive.iarc", rife_dir="path/to/RIFE")
    reader.open()

    # List frames
    for f in reader.list_frames():
        print(f.index, f.name, f.frame_type)

    # Extract single frame as numpy BGR
    img = reader.get_frame(42)

    # Extract to file
    reader.extract_frame(42, "output.png")

    # Extract range
    reader.extract_range(0, 100, "output_folder/")

    # Extract all
    reader.extract_all("output_folder/")

    # Get archive info
    info = reader.info()
    print(info["total_frames"], info["reduction_percent"])

    reader.close()

GUI USAGE FROM OTHER APP:
    from iarc_viewer import IARCViewerApp

    app = IARCViewerApp(archive_path="archive.iarc")
    app.mainloop()

from iarc_viewer import IARCReader

# Context manager
with IARCReader("archive.iarc") as reader:
    print(reader.info())
    img = reader.get_frame(42)          # BGR numpy
    img = reader.get_frame_rgb(42)      # RGB numpy
    img = reader.get_frame_pil(42)      # PIL Image
    reader.extract_frame(42, "out.png")
    reader.extract_all("output/")

# Or manual open/close
reader = IARCReader("archive.iarc")
reader.open()
frames = reader.list_frames()
reader.close()

# Launch GUI from code
from iarc_viewer import IARCViewerApp
app = IARCViewerApp(archive_path="archive.iarc")
app.mainloop()

"""

import os
import sys
import io
import json
import struct
import hashlib
import time
import argparse
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import IntEnum
from collections import OrderedDict

import numpy as np
import cv2

# ============================================================================
# CONSTANTS
# ============================================================================

ARCHIVE_MAGIC = b"IARC"
ARCHIVE_FORMAT_VERSION = 1

SUPPORTED_OUTPUT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

DEFAULT_RIFE_DIR = r"E:\ai\und\assets\pretrained_models\RIFE"


class FrameType(IntEnum):
    KEYFRAME = 0
    INTERPOLATED = 1
    RESIDUAL = 2
    FORCED_KEYFRAME = 3
    DELETED = 4


FRAME_TYPE_LABELS = {
    FrameType.KEYFRAME: "K",
    FrameType.INTERPOLATED: "I",
    FrameType.RESIDUAL: "R",
    FrameType.FORCED_KEYFRAME: "C",
    FrameType.DELETED: "D",
}

FRAME_TYPE_NAMES = {
    FrameType.KEYFRAME: "Keyframe",
    FrameType.INTERPOLATED: "Interpolated",
    FrameType.RESIDUAL: "Interpolated + Residual",
    FrameType.FORCED_KEYFRAME: "Forced Keyframe",
    FrameType.DELETED: "Deleted",
}

# UI Colors (light theme)
COLOR_BG = "#F5F5F5"
COLOR_BG_WHITE = "#FFFFFF"
COLOR_BG_PANEL = "#E8EAF6"
COLOR_BG_HEADER = "#C5CAE9"
COLOR_TEXT = "#212121"
COLOR_TEXT_SEC = "#616161"
COLOR_ACCENT = "#3F51B5"
COLOR_SUCCESS = "#4CAF50"
COLOR_WARNING = "#FF9800"
COLOR_DANGER = "#F44336"
COLOR_INFO = "#2196F3"
COLOR_BTN_TEXT = "#FFFFFF"
COLOR_KEYFRAME = "#4CAF50"
COLOR_INTERPOLATED = "#2196F3"
COLOR_RESIDUAL = "#FF9800"
COLOR_FORCED = "#9C27B0"
COLOR_DELETED = "#F44336"

FRAME_TYPE_COLORS = {
    FrameType.KEYFRAME: COLOR_KEYFRAME,
    FrameType.INTERPOLATED: COLOR_INTERPOLATED,
    FrameType.RESIDUAL: COLOR_RESIDUAL,
    FrameType.FORCED_KEYFRAME: COLOR_FORCED,
    FrameType.DELETED: COLOR_DELETED,
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FrameEntry:
    """Single frame entry in the archive index."""
    index: int = 0
    name: str = ""
    frame_type: int = FrameType.KEYFRAME
    width: int = 0
    height: int = 0
    data_offset: int = 0
    data_size: int = 0
    residual_offset: int = 0
    residual_size: int = 0
    parent_keyframe_a: int = -1
    parent_keyframe_b: int = -1
    interpolation_timestep: float = 0.0
    gop_id: int = 0
    face_score: float = 1.0
    body_score: float = 1.0
    similarity_score: float = 1.0
    motion_score: float = 0.0
    scene_cut: bool = False
    identity_hash: str = ""
    is_deleted: bool = False
    checksum: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FrameEntry":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class ArchiveHeader:
    """Archive file header."""
    magic: bytes = ARCHIVE_MAGIC
    version: int = ARCHIVE_FORMAT_VERSION
    total_frames: int = 0
    keyframe_count: int = 0
    interpolated_count: int = 0
    residual_count: int = 0
    forced_keyframe_count: int = 0
    deleted_count: int = 0
    gop_count: int = 0
    gop_size: int = 12
    original_width: int = 0
    original_height: int = 0
    compression_codec: str = "webp"
    compression_quality: int = 92
    archive_downscale: bool = False
    downscale_factor: float = 1.0
    use_residuals: bool = True
    residual_strength: str = "medium"
    face_safe: bool = True
    body_safe: bool = True
    identity_check: bool = True
    depth_aware: bool = False
    similarity_threshold: float = 0.92
    face_threshold: float = 0.95
    body_threshold: float = 0.90
    rife_model: str = ""
    raft_model: str = ""
    upscaler_model: str = ""
    created_timestamp: str = ""
    build_time_seconds: float = 0.0
    original_total_bytes: int = 0
    archive_total_bytes: int = 0
    reduction_percent: float = 0.0
    index_offset: int = 0
    index_size: int = 0
    data_start_offset: int = 0
    checksum: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ArchiveHeader":
        d["magic"] = d.get("magic", "IARC").encode("ascii")
        valid = {k: v for k, v in d.items()
                 if k in cls.__dataclass_fields__}
        return cls(**valid)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def human_readable_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def webp_bytes_to_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("WebP decoding failed")
    return img


# ============================================================================
# RIFE ENGINE — Standalone, minimal, GPU-safe
# ============================================================================

class RIFEEngine:
    """
    Minimal RIFE interpolation engine.
    Lazy-loads model on first use.
    FP16 safe — converts model and inputs together.
    """

    def __init__(self, rife_dir: str = DEFAULT_RIFE_DIR,
                 device: str = "auto", fp16: bool = True):
        self.rife_dir = rife_dir
        self.fp16 = fp16
        self._model = None
        self._lock = threading.Lock()

        import torch
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Disable FP16 on CPU
        if self.device == "cpu":
            self.fp16 = False

    def _load(self):
        """Load RIFE model lazily."""
        import torch

        if self.rife_dir not in sys.path:
            sys.path.insert(0, self.rife_dir)

        try:
            from model.RIFE_HDv3 import Model as RIFEModel
        except ImportError:
            try:
                from RIFE_HDv3 import Model as RIFEModel
            except ImportError:
                from model.RIFE import Model as RIFEModel

        model = RIFEModel()
        model.load_model(self.rife_dir, -1)
        model.eval()

        if self.device == "cuda":
            model.device()
            if self.fp16:
                model.flownet = model.flownet.half()

        return model

    def get_model(self):
        with self._lock:
            if self._model is None:
                self._model = self._load()
            return self._model

    def interpolate(self, img1: np.ndarray, img2: np.ndarray,
                    timestep: float = 0.5) -> np.ndarray:
        """
        Interpolate between two BGR images.
        Returns BGR numpy uint8 array.
        """
        import torch

        model = self.get_model()
        use_fp16 = self.fp16 and self.device == "cuda"

        def _to_tensor(img: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(
                img.copy()
            ).permute(2, 0, 1).float() / 255.0
            t = t.unsqueeze(0)
            if self.device == "cuda":
                t = t.cuda()
            if use_fp16:
                t = t.half()
            return t

        # h, w = img1.shape[:2]
        # pad_h = ((h - 1) // 64 + 1) * 64
        # pad_w = ((w - 1) // 64 + 1) * 64
        # need_pad = (h != pad_h or w != pad_w)
        #
        # if need_pad:
        #     i1 = cv2.resize(img1, (pad_w, pad_h),
        #                     interpolation=cv2.INTER_LINEAR)
        #     i2 = cv2.resize(img2, (pad_w, pad_h),
        #                     interpolation=cv2.INTER_LINEAR)
        # else:
        #     i1, i2 = img1, img2

        # h1, w1 = img1.shape[:2]
        # h2, w2 = img2.shape[:2]
        #
        # if h2 != h1 or w2 != w1:
        #     img2 = cv2.resize(img2, (w1, h1),
        #                       interpolation=cv2.INTER_LINEAR)
        #
        # h, w = h1, w1

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        mismatched = (h1 != h2 or w1 != w2)

        if mismatched:
            canvas_h = max(h1, h2)
            canvas_w = max(w1, w2)

            def _letterbox(img, target_h, target_w):
                ih, iw = img.shape[:2]
                scale = min(target_w / iw, target_h / ih)
                new_w = int(iw * scale)
                new_h = int(ih * scale)
                resized = cv2.resize(
                    img, (new_w, new_h),
                    interpolation=cv2.INTER_LANCZOS4
                )
                canvas = np.zeros(
                    (target_h, target_w, 3), dtype=np.uint8
                )
                y_off = (target_h - new_h) // 2
                x_off = (target_w - new_w) // 2
                canvas[y_off:y_off + new_h,
                       x_off:x_off + new_w] = resized
                return canvas, (x_off, y_off, new_w, new_h)

            img1, box1 = _letterbox(img1, canvas_h, canvas_w)
            img2, box2 = _letterbox(img2, canvas_h, canvas_w)
            h, w = canvas_h, canvas_w
        else:
            h, w = h1, w1
        pad_h = ((h - 1) // 64 + 1) * 64
        pad_w = ((w - 1) // 64 + 1) * 64
        need_pad = (h != pad_h or w != pad_w)

        if need_pad:
            i1 = cv2.resize(img1, (pad_w, pad_h),
                             interpolation=cv2.INTER_LINEAR)
            i2 = cv2.resize(img2, (pad_w, pad_h),
                             interpolation=cv2.INTER_LINEAR)
        else:
            i1, i2 = img1, img2

        t1 = _to_tensor(i1)
        t2 = _to_tensor(i2)

        with torch.no_grad():
            try:
                mid = model.inference(t1, t2, timestep=timestep)
            except TypeError:
                mid = model.inference(t1, t2)

        result = (
            mid[0].float().clamp(0, 1)
            .cpu().permute(1, 2, 0)
            .numpy() * 255
        ).astype(np.uint8)

        # if need_pad:
        #     result = cv2.resize(result, (w, h),
        #                         interpolation=cv2.INTER_LINEAR)
        # if need_pad:
        #     result = cv2.resize(result, (w1, h1),
        #                         interpolation=cv2.INTER_LINEAR)
        if need_pad:
            result = cv2.resize(result, (w, h),
                                interpolation=cv2.INTER_LINEAR)

        if mismatched:
            x_off, y_off, new_w, new_h = box1
            result = result[y_off:y_off + new_h,
                            x_off:x_off + new_w]
            result = cv2.resize(result, (w1, h1),
                                interpolation=cv2.INTER_LANCZOS4)

        return result

    def unload(self):
        with self._lock:
            self._model = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# IARC READER — Core API (no GUI dependency)
# ============================================================================

class IARCReader:
    """
    Read-only API for .iarc archives.
    Supports listing, random access extraction, range extraction.
    Thread-safe for concurrent frame reads.

    Usage:
        reader = IARCReader("archive.iarc")
        reader.open()

        frames = reader.list_frames()
        img = reader.get_frame(42)
        reader.extract_frame(42, "output.png")
        reader.extract_all("output_folder/")

        info = reader.info()
        reader.close()
    """

    def __init__(self, archive_path: str,
                 rife_dir: str = DEFAULT_RIFE_DIR,
                 device: str = "auto",
                 fp16: bool = True,
                 cache_size: int = 50):
        """
        Args:
            archive_path: Path to .iarc file
            rife_dir: Path to RIFE model directory
            device: "auto", "cuda", or "cpu"
            fp16: Use FP16 for RIFE inference
            cache_size: Number of decoded frames to cache in memory
        """
        self.archive_path = archive_path
        self.rife_dir = rife_dir
        self._device = device
        self._fp16 = fp16
        self._cache_size = cache_size

        self._header: Optional[ArchiveHeader] = None
        self._frames: List[FrameEntry] = []
        self._file_handle = None
        self._data_start: int = 0
        self._rife: Optional[RIFEEngine] = None
        self._decode_cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._is_open = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self):
        """Open the archive and parse header + index."""
        if self._is_open:
            self.close()

        with open(self.archive_path, "rb") as fp:
            magic = fp.read(4)
            if magic != ARCHIVE_MAGIC:
                raise ValueError(
                    f"Not a valid .iarc archive (magic: {magic})"
                )

            version = struct.unpack("<I", fp.read(4))[0]
            header_size = struct.unpack("<Q", fp.read(8))[0]
            index_size = struct.unpack("<Q", fp.read(8))[0]

            header_json = fp.read(header_size).rstrip()
            index_json = fp.read(index_size)

            self._data_start = fp.tell()

        header_dict = json.loads(header_json)
        self._header = ArchiveHeader.from_dict(header_dict)

        index_list = json.loads(index_json)
        self._frames = [FrameEntry.from_dict(d) for d in index_list]

        self._file_handle = open(self.archive_path, "rb")
        self._is_open = True

        return self

    def close(self):
        """Close the archive and free resources."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

        self._decode_cache.clear()

        if self._rife:
            self._rife.unload()
            self._rife = None

        self._is_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    # RIFE lazy loader
    # ------------------------------------------------------------------

    def _get_rife(self) -> RIFEEngine:
        if self._rife is None:
            self._rife = RIFEEngine(
                rife_dir=self.rife_dir,
                device=self._device,
                fp16=self._fp16
            )
        return self._rife

    # ------------------------------------------------------------------
    # Info / List
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """Get archive information as a dictionary."""
        self._check_open()
        h = self._header
        return {
            "archive_path": self.archive_path,
            "total_frames": h.total_frames,
            "keyframe_count": h.keyframe_count,
            "forced_keyframe_count": h.forced_keyframe_count,
            "interpolated_count": h.interpolated_count,
            "residual_count": h.residual_count,
            "deleted_count": h.deleted_count,
            "gop_count": h.gop_count,
            "gop_size": h.gop_size,
            "original_width": h.original_width,
            "original_height": h.original_height,
            "codec": h.compression_codec,
            "quality": h.compression_quality,
            "archive_downscale": h.archive_downscale,
            "downscale_factor": h.downscale_factor,
            "original_total_bytes": h.original_total_bytes,
            "archive_total_bytes": h.archive_total_bytes,
            "reduction_percent": h.reduction_percent,
            "created": h.created_timestamp,
            "archive_size_human": human_readable_size(
                h.archive_total_bytes
            ),
            "original_size_human": human_readable_size(
                h.original_total_bytes
            ),
        }

    def list_frames(self) -> List[FrameEntry]:
        """List all frames in the archive."""
        self._check_open()
        return self._frames.copy()

    def get_frame_entry(self, index: int) -> FrameEntry:
        """Get metadata for a specific frame."""
        self._check_open()
        self._check_index(index)
        return self._frames[index]

    def frame_count(self) -> int:
        """Get total frame count."""
        self._check_open()
        return len(self._frames)

    def get_header(self) -> ArchiveHeader:
        """Get archive header."""
        self._check_open()
        return self._header

    # ------------------------------------------------------------------
    # Frame extraction — core API
    # ------------------------------------------------------------------

    def get_frame(self, index: int,
                  use_cache: bool = True) -> np.ndarray:
        """
        Decode and return a single frame as BGR numpy array.
        For keyframes: direct WebP decode.
        For interpolated: RIFE reconstruction from parent keyframes.
        For residual: RIFE + residual correction.

        Args:
            index: Frame index (0-based)
            use_cache: Use LRU decode cache (default True)

        Returns:
            BGR numpy uint8 array at original resolution
        """
        self._check_open()
        self._check_index(index)

        # Cache check
        if use_cache and index in self._decode_cache:
            # Move to end (LRU)
            self._decode_cache.move_to_end(index)
            return self._decode_cache[index].copy()

        # img = self._decode_frame(index)
        img = self._decode_frame(index, _depth=0)

        # Cache store
        if use_cache:
            self._decode_cache[index] = img.copy()
            while len(self._decode_cache) > self._cache_size:
                self._decode_cache.popitem(last=False)

        return img

    def get_frame_rgb(self, index: int) -> np.ndarray:
        """Get frame as RGB numpy array (for PIL/matplotlib)."""
        bgr = self.get_frame(index)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_frame_pil(self, index: int):
        """Get frame as PIL Image."""
        from PIL import Image
        rgb = self.get_frame_rgb(index)
        return Image.fromarray(rgb)

    def extract_frame(self, index: int, output_path: str):
        """
        Extract a single frame and save to disk.

        Args:
            index: Frame index
            output_path: Output file path (.png, .jpg, .webp, etc.)
        """
        img = self.get_frame(index)
        ext = os.path.splitext(output_path)[1].lower()

        if ext == ".webp":
            cv2.imwrite(output_path, img,
                        [cv2.IMWRITE_WEBP_QUALITY, 95])
        elif ext in (".jpg", ".jpeg"):
            cv2.imwrite(output_path, img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(output_path, img)

    def extract_range(self, start: int, end: int,
                      output_folder: str,
                      ext: str = ".png",
                      callback=None) -> int:
        """
        Extract a range of frames to a folder.

        Args:
            start: Start index (inclusive)
            end: End index (inclusive)
            output_folder: Output directory
            ext: Output file extension
            callback: Optional fn(current, total, frame_name) for progress

        Returns:
            Number of frames extracted
        """
        self._check_open()
        os.makedirs(output_folder, exist_ok=True)

        count = 0
        total = end - start + 1

        for i in range(start, end + 1):
            if i < 0 or i >= len(self._frames):
                continue

            f = self._frames[i]
            if f.is_deleted:
                continue

            name = os.path.splitext(f.name)[0] + ext
            out_path = os.path.join(output_folder, name)

            try:
                self.extract_frame(i, out_path)
                count += 1
            except Exception as e:
                print(f"Warning: Failed to extract frame {i}: {e}")

            if callback:
                callback(count, total, f.name)

        return count

    def extract_all(self, output_folder: str,
                    ext: str = ".png",
                    callback=None) -> int:
        """
        Extract all frames to a folder.

        Args:
            output_folder: Output directory
            ext: Output file extension
            callback: Optional fn(current, total, frame_name)

        Returns:
            Number of frames extracted
        """
        return self.extract_range(
            0, len(self._frames) - 1,
            output_folder, ext, callback
        )

    def extract_keyframes_only(self, output_folder: str,
                               ext: str = ".png",
                               callback=None) -> int:
        """
        Extract only keyframes (no RIFE needed, very fast).

        Returns:
            Number of frames extracted
        """
        self._check_open()
        os.makedirs(output_folder, exist_ok=True)

        kf_indices = [
            f.index for f in self._frames
            if f.frame_type in (
                FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME
            ) and not f.is_deleted
        ]

        count = 0
        total = len(kf_indices)

        for pos, i in enumerate(kf_indices):
            f = self._frames[i]
            name = os.path.splitext(f.name)[0] + ext
            out_path = os.path.join(output_folder, name)

            try:
                self.extract_frame(i, out_path)
                count += 1
            except Exception as e:
                print(f"Warning: Failed to extract keyframe {i}: {e}")

            if callback:
                callback(count, total, f.name)

        return count

    # ------------------------------------------------------------------
    # Search / Filter
    # ------------------------------------------------------------------

    def search(self, query: str) -> List[FrameEntry]:
        """Search frames by name (partial match)."""
        self._check_open()
        q = query.lower()
        return [f for f in self._frames if q in f.name.lower()]

    def filter_by_type(self, frame_type: int) -> List[FrameEntry]:
        """Filter frames by type."""
        self._check_open()
        return [f for f in self._frames if f.frame_type == frame_type]

    def get_keyframe_indices(self) -> List[int]:
        """Get indices of all keyframes."""
        self._check_open()
        return [
            f.index for f in self._frames
            if f.frame_type in (
                FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME
            )
        ]

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def verify(self) -> Dict[str, Any]:
        """Verify archive integrity."""
        self._check_open()
        results = {
            "valid": True,
            "total_frames": len(self._frames),
            "corrupted_frames": [],
            "missing_data": [],
        }

        for f in self._frames:
            if f.is_deleted:
                continue
            if f.frame_type in (
                FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME
            ):
                if f.data_size <= 0:
                    results["missing_data"].append(f.index)
                    results["valid"] = False

        return results


    def compact(self, output_path: str,
                callback=None) -> str:
        """
        Compact archive to output_path, removing deleted frames.

        Args:
            output_path: Path for compacted archive
            callback: Optional fn(message, current, total)

        Returns:
            Path to compacted archive
        """
        self._check_open()

        header = self._header
        old_frames = self._frames
        total = len(old_frames)

        def _cb(msg, cur, tot):
            if callback:
                callback(msg, cur, tot)

        # Build index mapping
        old_to_new = {}
        new_idx = 0
        for f in old_frames:
            if not f.is_deleted:
                old_to_new[f.index] = new_idx
                new_idx += 1

        # Collect live data
        live_frames = []
        data_blobs = {}
        residual_blobs = {}
        new_idx = 0

        for i, f in enumerate(old_frames):
            if f.is_deleted:
                continue

            if f.frame_type in (
                FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME
            ) and f.data_size > 0:
                with self._lock:
                    self._file_handle.seek(
                        self._data_start + f.data_offset
                    )
                    data_blobs[new_idx] = self._file_handle.read(
                        f.data_size
                    )

            if f.frame_type == FrameType.RESIDUAL \
               and f.residual_size > 0:
                with self._lock:
                    self._file_handle.seek(
                        self._data_start + f.residual_offset
                    )
                    residual_blobs[new_idx] = self._file_handle.read(
                        f.residual_size
                    )

            nf = FrameEntry(
                index=new_idx,
                name=f.name,
                frame_type=f.frame_type,
                width=f.width,
                height=f.height,
                data_size=f.data_size,
                residual_size=f.residual_size,
                parent_keyframe_a=old_to_new.get(
                    f.parent_keyframe_a, -1
                ),
                parent_keyframe_b=old_to_new.get(
                    f.parent_keyframe_b, -1
                ),
                interpolation_timestep=f.interpolation_timestep,
                gop_id=f.gop_id,
                face_score=f.face_score,
                body_score=f.body_score,
                similarity_score=f.similarity_score,
                motion_score=f.motion_score,
                scene_cut=f.scene_cut,
                is_deleted=False,
                checksum=f.checksum,
            )

            if nf.frame_type in (
                FrameType.INTERPOLATED, FrameType.RESIDUAL
            ):
                if nf.parent_keyframe_a == -1 or \
                   nf.parent_keyframe_b == -1:
                    try:
                        img = self._decode_frame(f.index)
                        blob = cv2.imencode(
                            ".webp", img,
                            [cv2.IMWRITE_WEBP_QUALITY,
                             header.compression_quality]
                        )[1].tobytes()
                        data_blobs[new_idx] = blob
                        nf.frame_type = FrameType.KEYFRAME
                        nf.data_size = len(blob)
                        nf.residual_size = 0
                        nf.parent_keyframe_a = -1
                        nf.parent_keyframe_b = -1
                        if new_idx in residual_blobs:
                            del residual_blobs[new_idx]
                    except Exception:
                        pass

            live_frames.append(nf)
            new_idx += 1

            if i % 500 == 0:
                _cb("Reading frames...", i + 1, total)

        # Compute offsets
        offset = 0
        for i, f in enumerate(live_frames):
            if i in data_blobs:
                f.data_offset = offset
                f.data_size = len(data_blobs[i])
                offset += f.data_size
            else:
                f.data_offset = 0
                f.data_size = 0
            if i in residual_blobs:
                f.residual_offset = offset
                f.residual_size = len(residual_blobs[i])
                offset += f.residual_size
            else:
                f.residual_offset = 0
                f.residual_size = 0

        # Write
        import struct as st

        new_header = ArchiveHeader(
            total_frames=len(live_frames),
            keyframe_count=sum(
                1 for f in live_frames
                if f.frame_type == FrameType.KEYFRAME
            ),
            interpolated_count=sum(
                1 for f in live_frames
                if f.frame_type == FrameType.INTERPOLATED
            ),
            residual_count=sum(
                1 for f in live_frames
                if f.frame_type == FrameType.RESIDUAL
            ),
            forced_keyframe_count=sum(
                1 for f in live_frames
                if f.frame_type == FrameType.FORCED_KEYFRAME
            ),
            original_width=header.original_width,
            original_height=header.original_height,
            compression_codec=header.compression_codec,
            compression_quality=header.compression_quality,
            gop_size=header.gop_size,
            created_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            original_total_bytes=header.original_total_bytes,
        )

        h_json = json.dumps(
            new_header.to_dict(), separators=(",", ":")
        ).encode("utf-8")
        i_json = json.dumps(
            [f.to_dict() for f in live_frames],
            separators=(",", ":")
        ).encode("utf-8")

        with open(output_path, "wb") as fp:
            fp.write(ARCHIVE_MAGIC)
            fp.write(st.pack("<I", ARCHIVE_FORMAT_VERSION))
            fp.write(st.pack("<Q", len(h_json)))
            fp.write(st.pack("<Q", len(i_json)))
            fp.write(h_json)
            fp.write(i_json)
            for i in range(len(live_frames)):
                if i in data_blobs:
                    fp.write(data_blobs[i])
                if i in residual_blobs:
                    fp.write(residual_blobs[i])
            fp.write(b"\x00" * 64)
            arc_size = fp.tell()

        with open(output_path, "r+b") as fp:
            fp.seek(0)
            all_d = fp.read(arc_size - 64)
            cs = hashlib.sha256(all_d).hexdigest()
            fp.seek(arc_size - 64)
            fp.write(cs.encode("ascii")[:64].ljust(64, b"\x00"))

        _cb("Compact complete", total, total)
        return output_path

    # ------------------------------------------------------------------
    # Internal decode logic
    # ------------------------------------------------------------------

    def _decode_frame(self, index: int,
                      _depth: int = 0) -> np.ndarray:
        """Internal: decode a single frame. Handles recursion safely."""
        if _depth > 10:
            raise RuntimeError(
                f"Recursion depth exceeded decoding frame {index}. "
                f"Archive may have circular parent references."
            )

        f = self._frames[index]

        if f.is_deleted:
            raise RuntimeError(f"Frame {index} is deleted")

        h = self._header

        if f.frame_type in (FrameType.KEYFRAME,
                            FrameType.FORCED_KEYFRAME):
            if f.data_size <= 0:
                raise RuntimeError(
                    f"Keyframe {index} has no stored data"
                )

            with self._lock:
                self._file_handle.seek(
                    self._data_start + f.data_offset
                )
                data = self._file_handle.read(f.data_size)

            img = webp_bytes_to_image(data)

            if h.archive_downscale and h.downscale_factor < 1.0:
                img = cv2.resize(
                    img,
                    (h.original_width, h.original_height),
                    interpolation=cv2.INTER_LANCZOS4
                )

            return img

        elif f.frame_type in (FrameType.INTERPOLATED,
                              FrameType.RESIDUAL):
            pa = f.parent_keyframe_a
            pb = f.parent_keyframe_b

            # Safety: parents must not be self
            if pa == index:
                pa = self._resolve_to_keyframe(
                    index - 1 if index > 0 else 0
                )
            if pb == index:
                pb = self._resolve_to_keyframe(
                    index + 1
                    if index < len(self._frames) - 1
                    else index - 1
                )
            if pa == index and pb == index:
                for kf in self._frames:
                    if kf.index != index and \
                            kf.frame_type in (FrameType.KEYFRAME,
                                              FrameType.FORCED_KEYFRAME) \
                            and not kf.is_deleted and kf.data_size > 0:
                        pa = kf.index
                        pb = kf.index
                        break

            # Resolve to actual keyframes
            pa = self._resolve_to_keyframe(pa)
            pb = self._resolve_to_keyframe(pb)

            # Use get_frame for caching, but pass depth
            img_a = self._get_frame_with_depth(pa, _depth + 1)
            img_b = self._get_frame_with_depth(pb, _depth + 1)

            rife = self._get_rife()
            interpolated = rife.interpolate(
                img_a, img_b, f.interpolation_timestep
            )

            if f.frame_type == FrameType.RESIDUAL \
                    and f.residual_size > 0:
                with self._lock:
                    self._file_handle.seek(
                        self._data_start + f.residual_offset
                    )
                    res_data = self._file_handle.read(
                        f.residual_size
                    )

                residual = webp_bytes_to_image(res_data)

                if residual.shape != interpolated.shape:
                    residual = cv2.resize(
                        residual,
                        (interpolated.shape[1],
                         interpolated.shape[0])
                    )

                interpolated = cv2.add(interpolated, residual)

            return interpolated

        raise RuntimeError(f"Unknown frame type: {f.frame_type}")

    def _resolve_to_keyframe(self, index: int) -> int:
        """Find nearest actual keyframe if index points to
        an interpolated frame."""
        if index < 0 or index >= len(self._frames):
            return index

        f = self._frames[index]
        if f.frame_type in (FrameType.KEYFRAME,
                            FrameType.FORCED_KEYFRAME):
            return index

        for offset in range(1, len(self._frames)):
            back = index - offset
            if back >= 0:
                bf = self._frames[back]
                if bf.frame_type in (FrameType.KEYFRAME,
                                     FrameType.FORCED_KEYFRAME) \
                        and not bf.is_deleted and bf.data_size > 0:
                    return back

            fwd = index + offset
            if fwd < len(self._frames):
                ff = self._frames[fwd]
                if ff.frame_type in (FrameType.KEYFRAME,
                                     FrameType.FORCED_KEYFRAME) \
                        and not ff.is_deleted and ff.data_size > 0:
                    return fwd

            if back < 0 and fwd >= len(self._frames):
                break

        return index

    def _get_frame_with_depth(self, index: int,
                              _depth: int) -> np.ndarray:
        """Get frame with depth tracking for recursion safety."""
        # Check cache first
        if index in self._decode_cache:
            self._decode_cache.move_to_end(index)
            return self._decode_cache[index].copy()

        img = self._decode_frame(index, _depth=_depth)

        # Cache it
        self._decode_cache[index] = img.copy()
        while len(self._decode_cache) > self._cache_size:
            self._decode_cache.popitem(last=False)

        return img

    def _check_open(self):
        if not self._is_open:
            raise RuntimeError(
                "Archive not open. Call .open() first."
            )

    def _check_index(self, index: int):
        if index < 0 or index >= len(self._frames):
            raise IndexError(
                f"Frame index {index} out of range "
                f"(0-{len(self._frames) - 1})"
            )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self):
        """Clear the decoded frame cache."""
        self._decode_cache.clear()

    def set_cache_size(self, size: int):
        """Set maximum cache size."""
        self._cache_size = size
        while len(self._decode_cache) > self._cache_size:
            self._decode_cache.popitem(last=False)

# ============================================================================
# GUI VIEWER — Standalone Tkinter application
# ============================================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


class ToolTip:
    """Singleton tooltip — only one visible at a time."""

    _active = None

    def __init__(self, widget, text: str, delay: int = 500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_window = None
        self.scheduled = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")
        widget.bind("<MouseWheel>", self._on_leave, add="+")

    def _on_enter(self, event=None):
        if ToolTip._active and ToolTip._active is not self:
            ToolTip._active._hide()
            ToolTip._active._cancel()
        self._cancel()
        self.scheduled = self.widget.after(self.delay, self._show)

    def _on_leave(self, event=None):
        self._cancel()
        self._hide()

    def _cancel(self):
        if self.scheduled:
            try:
                self.widget.after_cancel(self.scheduled)
            except Exception:
                pass
            self.scheduled = None

    def _show(self):
        if self.tip_window:
            return
        try:
            if not self.widget.winfo_exists():
                return
            if not self.widget.winfo_viewable():
                return
        except Exception:
            return

        ToolTip._active = self
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_attributes("-topmost", True)

        frame = tk.Frame(tw, bg="#FFFDE7", relief="solid", bd=1)
        frame.pack()
        tk.Label(
            frame, text=self.text, justify="left",
            bg="#FFFDE7", fg="#333333", font=("Segoe UI", 9),
            padx=8, pady=4, wraplength=350
        ).pack()

        tw.update_idletasks()
        try:
            sw = self.widget.winfo_screenwidth()
            sh = self.widget.winfo_screenheight()
        except Exception:
            sw, sh = 1920, 1080

        tw_w = tw.winfo_width()
        tw_h = tw.winfo_height()
        if x + tw_w > sw:
            x = sw - tw_w - 10
        if y + tw_h > sh:
            y = self.widget.winfo_rooty() - tw_h - 5

        tw.wm_geometry(f"+{x}+{y}")

        try:
            self.widget.after(4000, self._hide)
        except Exception:
            pass

    def _hide(self):
        if self.tip_window:
            try:
                self.tip_window.destroy()
            except Exception:
                pass
            self.tip_window = None
        if ToolTip._active is self:
            ToolTip._active = None


def tip(widget, text: str):
    """Attach tooltip to widget."""
    return ToolTip(widget, text)


class ImageCanvas(tk.Canvas):
    """Image preview canvas with zoom and pan."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent, bg=COLOR_BG_PANEL,
            highlightthickness=1, highlightbackground="#BDBDBD",
            **kwargs
        )
        self._image = None
        self._photo = None
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._drag_start = None

        self.bind("<MouseWheel>", self._on_scroll)
        self.bind("<ButtonPress-1>", self._on_drag_start)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<Double-Button-1>", lambda e: self.fit())
        self.bind("<Configure>", lambda e: self._render())

    def set_image(self, bgr: np.ndarray):
        if bgr is None:
            self.clear()
            return
        self._image = bgr
        self._render()

    def clear(self):
        self._image = None
        self._photo = None
        self.delete("all")

    def fit(self):
        if self._image is None:
            return
        h, w = self._image.shape[:2]
        cw = self.winfo_width()
        ch = self.winfo_height()
        self._scale = min(cw / w, ch / h) * 0.95
        self._offset_x = 0
        self._offset_y = 0
        self._render()

    def zoom_100(self):
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._render()

    def _render(self):
        if self._image is None:
            return
        self.delete("all")

        h, w = self._image.shape[:2]
        nw = max(1, int(w * self._scale))
        nh = max(1, int(h * self._scale))

        resized = cv2.resize(self._image, (nw, nh),
                             interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(pil_img)

        cw = self.winfo_width()
        ch = self.winfo_height()
        x = (cw - nw) // 2 + self._offset_x
        y = (ch - nh) // 2 + self._offset_y
        self.create_image(x, y, anchor="nw", image=self._photo)

    def _on_scroll(self, event):
        if self._image is None:
            return
        factor = 1.1 if event.delta > 0 else 0.9
        self._scale = max(0.05, min(20.0, self._scale * factor))
        self._render()

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if self._drag_start:
            self._offset_x += event.x - self._drag_start[0]
            self._offset_y += event.y - self._drag_start[1]
            self._drag_start = (event.x, event.y)
            self._render()


class IARCViewerApp(tk.Tk):
    """
    Standalone .iarc archive viewer and extractor.

    Usage:
        app = IARCViewerApp()
        app.mainloop()

        # Or with pre-loaded archive:
        app = IARCViewerApp(archive_path="file.iarc")
        app.mainloop()
    """

    def __init__(self, archive_path: str = None,
                 rife_dir: str = DEFAULT_RIFE_DIR,
                 **kwargs):
        super().__init__(**kwargs)

        self.title("IARC Viewer")
        self.state("zoomed")
        self.configure(bg=COLOR_BG)
        self.minsize(1100, 700)

        self._reader: Optional[IARCReader] = None
        self._frames: List[FrameEntry] = []
        self._rife_dir = rife_dir
        self._decode_time = 0.0

        self._setup_styles()
        self._build_toolbar()
        self._build_main()
        self._build_statusbar()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if archive_path and os.path.exists(archive_path):
            self.after(200, lambda: self._open_archive(archive_path))

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=COLOR_BG, foreground=COLOR_TEXT,
                        font=("Segoe UI", 10))
        style.configure("TNotebook.Tab", padding=[12, 6],
                        font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", background=COLOR_BG_WHITE,
                        foreground=COLOR_TEXT,
                        fieldbackground=COLOR_BG_WHITE,
                        font=("Consolas", 10))
        style.configure("Treeview.Heading",
                        font=("Segoe UI", 10, "bold"),
                        background=COLOR_BG_HEADER)
        style.configure("TProgressbar", troughcolor=COLOR_BG_PANEL,
                        background=COLOR_ACCENT, thickness=20)

    def _btn(self, parent, text, color, command, tooltip_text=""):
        active = {
            COLOR_ACCENT: "#303F9F", COLOR_SUCCESS: "#388E3C",
            COLOR_WARNING: "#F57C00", COLOR_DANGER: "#D32F2F",
            COLOR_INFO: "#1976D2",
        }.get(color, "#555555")

        fg = COLOR_BTN_TEXT if color != COLOR_WARNING else "#212121"

        btn = tk.Button(
            parent, text=text, bg=color, fg=fg,
            activebackground=active, activeforeground=fg,
            font=("Segoe UI", 10, "bold"), relief="flat",
            cursor="hand2", padx=14, pady=5, command=command
        )
        btn.bind("<Enter>", lambda e: btn.config(bg=active))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))

        if tooltip_text:
            tip(btn, tooltip_text)

        return btn

    def _build_toolbar(self):
        tb = tk.Frame(self, bg=COLOR_BG_HEADER)
        tb.pack(fill="x")

        left = tk.Frame(tb, bg=COLOR_BG_HEADER)
        left.pack(side="left", padx=10, pady=5)

        self._btn(
            left, "📁 Open Archive", COLOR_ACCENT,
            self._browse_open,
            "Open an .iarc archive file"
        ).pack(side="left", padx=3)

        self._btn(
            left, "🔄 Reload", COLOR_INFO,
            self._reload,
            "Reload the currently open archive"
        ).pack(side="left", padx=3)

        ttk.Separator(left, orient="vertical").pack(
            side="left", fill="y", padx=8
        )

        self._btn(
            left, "📤 Extract Selected", COLOR_WARNING,
            self._extract_selected,
            "Extract selected frame(s) to disk"
        ).pack(side="left", padx=3)

        self._btn(
            left, "📤 Extract Range", COLOR_WARNING,
            self._extract_range_dialog,
            "Extract a range of frames by index"
        ).pack(side="left", padx=3)

        self._btn(
            left, "📤 Extract All", COLOR_WARNING,
            self._extract_all,
            "Extract all frames to a folder (uses GPU for interpolated)"
        ).pack(side="left", padx=3)

        self._btn(
            left, "📤 KFs Only", COLOR_INFO,
            self._extract_kf_only,
            "Export only keyframes (fast, no GPU)"
        ).pack(side="left", padx=3)

        ttk.Separator(left, orient="vertical").pack(
            side="left", fill="y", padx=8
        )

        self._btn(
            left, "✅ Verify", COLOR_SUCCESS,
            self._verify,
            "Verify archive integrity"
        ).pack(side="left", padx=3)

        # Right — GPU status
        right = tk.Frame(tb, bg=COLOR_BG_HEADER)
        right.pack(side="right", padx=10, pady=5)

        self._gpu_label = tk.Label(
            right, text="GPU: ...", bg=COLOR_BG_HEADER,
            fg=COLOR_TEXT_SEC, font=("Segoe UI", 9)
        )
        self._gpu_label.pack(side="right")
        tip(self._gpu_label, "GPU/CUDA status for RIFE operations")

        self.after(300, self._update_gpu)

    def _build_main(self):
        main = tk.Frame(self, bg=COLOR_BG)
        main.pack(fill="both", expand=True, padx=10, pady=5)

        # Left — frame list + search
        left = tk.LabelFrame(
            main, text="Frame Index", font=("Segoe UI", 10, "bold"),
            bg=COLOR_BG, fg=COLOR_TEXT, padx=5, pady=5
        )
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        tip(left, "All frames in the archive. Click to select. "
                  "Double-click to decode and preview.")

        # Search bar
        search_frame = tk.Frame(left, bg=COLOR_BG)
        search_frame.pack(fill="x", pady=(0, 5))

        tk.Label(
            search_frame, text="Search:", bg=COLOR_BG,
            fg=COLOR_TEXT, font=("Segoe UI", 10)
        ).pack(side="left")

        self._search_var = tk.StringVar()
        search_entry = tk.Entry(
            search_frame, textvariable=self._search_var,
            font=("Segoe UI", 10), width=15, relief="solid", bd=1
        )
        search_entry.pack(side="left", padx=5)
        tip(search_entry, "Search by frame name or index")
        self._search_var.trace_add("write", self._on_search)

        tk.Label(
            search_frame, text="Filter:", bg=COLOR_BG,
            fg=COLOR_TEXT, font=("Segoe UI", 10)
        ).pack(side="left", padx=(10, 0))

        self._filter_var = tk.StringVar(value="All")
        filter_cb = ttk.Combobox(
            search_frame,
            values=["All", "Keyframe", "Interpolated",
                    "Residual", "Forced KF", "Deleted"],
            textvariable=self._filter_var,
            width=12, state="readonly", font=("Segoe UI", 10)
        )
        filter_cb.pack(side="left", padx=5)
        filter_cb.bind("<<ComboboxSelected>>", self._on_filter)
        tip(filter_cb, "Filter by frame storage type")

        tk.Label(
            search_frame, text="Go to:", bg=COLOR_BG,
            fg=COLOR_TEXT, font=("Segoe UI", 10)
        ).pack(side="left", padx=(10, 0))

        self._goto_var = tk.StringVar()
        goto_entry = tk.Entry(
            search_frame, textvariable=self._goto_var,
            font=("Segoe UI", 10), width=8, relief="solid", bd=1
        )
        goto_entry.pack(side="left", padx=5)
        goto_entry.bind("<Return>", self._on_goto)
        tip(goto_entry, "Jump to a specific frame index")

        self._btn(
            search_frame, "Go", COLOR_INFO, self._on_goto,
            "Jump to frame index"
        ).pack(side="left", padx=2)

        self._count_var = tk.StringVar(value="0 frames")
        tk.Label(
            search_frame, textvariable=self._count_var,
            bg=COLOR_BG, fg=COLOR_TEXT_SEC, font=("Segoe UI", 9)
        ).pack(side="right", padx=5)

        # Treeview
        tree_frame = tk.Frame(left, bg=COLOR_BG)
        tree_frame.pack(fill="both", expand=True)

        cols = ("idx", "name", "type", "size",
                "sim", "face", "body", "status")
        self._tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings",
            selectmode="extended", height=30
        )

        col_defs = {
            "idx": ("#", 55, "center"),
            "name": ("Name", 200, "w"),
            "type": ("Type", 90, "center"),
            "size": ("Size", 85, "e"),
            "sim": ("Sim", 65, "center"),
            "face": ("Face", 65, "center"),
            "body": ("Body", 65, "center"),
            "status": ("Status", 70, "center"),
        }
        for cid, (heading, width, anchor) in col_defs.items():
            self._tree.heading(cid, text=heading,
                               command=lambda c=cid: self._sort(c))
            self._tree.column(cid, width=width, anchor=anchor,
                              minwidth=40)

        self._tree.tag_configure("keyframe", foreground=COLOR_KEYFRAME)
        self._tree.tag_configure("interpolated",
                                 foreground=COLOR_INTERPOLATED)
        self._tree.tag_configure("residual", foreground=COLOR_RESIDUAL)
        self._tree.tag_configure("forced", foreground=COLOR_FORCED)
        self._tree.tag_configure("deleted", foreground=COLOR_DELETED)
        self._tree.tag_configure("even", background=COLOR_BG_WHITE)
        self._tree.tag_configure("odd", background="#F5F5F5")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        self._tree.bind("<<TreeviewSelect>>", self._on_select)
        self._tree.bind("<Double-1>", self._on_double_click)

        self._sort_col = "idx"
        self._sort_rev = False
        self._filtered: List[int] = []

        # Center — preview
        center = tk.LabelFrame(
            main, text="Preview", font=("Segoe UI", 10, "bold"),
            bg=COLOR_BG, fg=COLOR_TEXT, padx=5, pady=5
        )
        center.pack(side="left", fill="both", expand=True, padx=5)
        center.config(width=500)

        self._canvas = ImageCanvas(center)
        self._canvas.pack(fill="both", expand=True)
        tip(self._canvas, "Scroll to zoom, drag to pan, "
                          "double-click to fit")

        ctrl = tk.Frame(center, bg=COLOR_BG)
        ctrl.pack(fill="x", pady=5)

        self._btn(ctrl, "◀ Prev", "#9E9E9E", self._prev,
                  "Previous frame").pack(side="left", padx=2)
        self._btn(ctrl, "Next ▶", "#9E9E9E", self._next,
                  "Next frame").pack(side="left", padx=2)
        self._btn(ctrl, "🔍 Decode", COLOR_ACCENT, self._decode_current,
                  "Decode and display selected frame").pack(
            side="left", padx=2
        )
        self._btn(ctrl, "Fit", COLOR_INFO,
                  lambda: self._canvas.fit(),
                  "Fit to preview area").pack(side="left", padx=2)
        self._btn(ctrl, "100%", COLOR_INFO,
                  lambda: self._canvas.zoom_100(),
                  "View at actual size").pack(side="left", padx=2)

        self._decode_var = tk.StringVar(value="")
        tk.Label(
            ctrl, textvariable=self._decode_var, bg=COLOR_BG,
            fg=COLOR_TEXT_SEC, font=("Segoe UI", 9)
        ).pack(side="right", padx=10)

        # Right — info panel
        right_panel = tk.LabelFrame(
            main, text="Info", font=("Segoe UI", 10, "bold"),
            bg=COLOR_BG, fg=COLOR_TEXT, padx=5, pady=5
        )
        right_panel.pack(side="right", fill="y", padx=(5, 0))
        right_panel.config(width=270)
        right_panel.pack_propagate(False)

        arch_lf = tk.LabelFrame(
            right_panel, text="Archive Info",
            font=("Segoe UI", 9, "bold"),
            bg=COLOR_BG, fg=COLOR_TEXT, padx=5, pady=5
        )
        arch_lf.pack(fill="x", pady=5)
        tip(arch_lf, "Summary of the loaded archive")

        self._arch_text = tk.Text(
            arch_lf, height=9, font=("Consolas", 9),
            bg=COLOR_BG_WHITE, fg=COLOR_TEXT, relief="solid",
            bd=1, wrap="word"
        )
        self._arch_text.pack(fill="x")
        self._arch_text.config(state="disabled")

        frame_lf = tk.LabelFrame(
            right_panel, text="Frame Detail",
            font=("Segoe UI", 9, "bold"),
            bg=COLOR_BG, fg=COLOR_TEXT, padx=5, pady=5
        )
        frame_lf.pack(fill="x", pady=5)
        tip(frame_lf, "Details of the selected frame")

        self._frame_text = tk.Text(
            frame_lf, height=14, font=("Consolas", 9),
            bg=COLOR_BG_WHITE, fg=COLOR_TEXT, relief="solid",
            bd=1, wrap="word"
        )
        self._frame_text.pack(fill="x")
        self._frame_text.config(state="disabled")

    def _build_statusbar(self):
        sb = tk.Frame(self, bg=COLOR_BG_PANEL)
        sb.pack(fill="x", side="bottom")

        self._status_var = tk.StringVar(value="Ready")
        tk.Label(
            sb, textvariable=self._status_var, bg=COLOR_BG_PANEL,
            fg=COLOR_TEXT, font=("Segoe UI", 9), anchor="w"
        ).pack(side="left", padx=10, pady=4)

        self._arch_status_var = tk.StringVar(value="No archive loaded")
        tk.Label(
            sb, textvariable=self._arch_status_var, bg=COLOR_BG_PANEL,
            fg=COLOR_TEXT_SEC, font=("Segoe UI", 9), anchor="e"
        ).pack(side="right", padx=10, pady=4)

    # ------------------------------------------------------------------
    # Open / Reload
    # ------------------------------------------------------------------

    def _browse_open(self):
        path = filedialog.askopenfilename(
            title="Open Archive",
            filetypes=[("IARC Archives", "*.iarc"), ("All", "*.*")]
        )
        if path:
            self._open_archive(path)

    def _open_archive(self, path: str):
        self._status("Opening archive...")
        try:
            if self._reader:
                self._reader.close()

            self._reader = IARCReader(
                path, rife_dir=self._rife_dir
            )
            self._reader.open()
            self._frames = self._reader.list_frames()

            self._populate_tree()
            self._update_arch_info()

            name = os.path.basename(path)
            n = len(self._frames)
            sz = human_readable_size(
                self._reader.get_header().archive_total_bytes
            )
            self._arch_status_var.set(
                f"{name} | {n:,} frames | {sz}"
            )
            self._status(f"Opened: {n:,} frames")

        except Exception as e:
            messagebox.showerror("Open Error", str(e))
            self._status("Open failed")

    def _reload(self):
        if self._reader and self._reader._is_open:
            self._open_archive(self._reader.archive_path)

    # ------------------------------------------------------------------
    # Tree population — batched for 100k+
    # ------------------------------------------------------------------

    def _populate_tree(self):
        self._tree.delete(*self._tree.get_children())

        self._apply_filters()

    def _apply_filters(self):
        search_q = self._search_var.get().strip().lower()
        filt = self._filter_var.get()

        type_map = {
            "Keyframe": FrameType.KEYFRAME,
            "Interpolated": FrameType.INTERPOLATED,
            "Residual": FrameType.RESIDUAL,
            "Forced KF": FrameType.FORCED_KEYFRAME,
            "Deleted": FrameType.DELETED,
        }

        indices = []
        for i, f in enumerate(self._frames):
            if filt != "All":
                target = type_map.get(filt)
                if target is not None and f.frame_type != target:
                    continue
            if search_q:
                if (search_q not in f.name.lower()
                        and search_q not in str(f.index)):
                    continue
            indices.append(i)

        # Sort
        col = self._sort_col
        rev = self._sort_rev

        def key_fn(idx):
            f = self._frames[idx]
            if col == "idx":
                return f.index
            elif col == "name":
                return f.name.lower()
            elif col == "type":
                return f.frame_type
            elif col == "size":
                return f.data_size + f.residual_size
            elif col == "sim":
                return f.similarity_score
            elif col == "face":
                return f.face_score
            elif col == "body":
                return f.body_score
            elif col == "status":
                return int(f.is_deleted)
            return 0

        indices.sort(key=key_fn, reverse=rev)
        self._filtered = indices
        self._count_var.set(f"{len(indices):,} frames")

        # Clear and batch insert
        self._tree.delete(*self._tree.get_children())
        self._batch_pos = 0
        self._insert_batch()

    def _insert_batch(self):
        BATCH = 5000
        start = self._batch_pos
        end = min(start + BATCH, len(self._filtered))

        tag_map = {
            FrameType.KEYFRAME: "keyframe",
            FrameType.INTERPOLATED: "interpolated",
            FrameType.RESIDUAL: "residual",
            FrameType.FORCED_KEYFRAME: "forced",
            FrameType.DELETED: "deleted",
        }

        for pos in range(start, end):
            idx = self._filtered[pos]
            f = self._frames[idx]

            t_tag = tag_map.get(f.frame_type, "")
            s_tag = "even" if pos % 2 == 0 else "odd"

            total_sz = f.data_size + f.residual_size
            sz_str = human_readable_size(total_sz) if total_sz > 0 else "---"
            status = "Deleted" if f.is_deleted else "OK"
            t_label = FRAME_TYPE_LABELS.get(f.frame_type, "?")
            t_name = FRAME_TYPE_NAMES.get(f.frame_type, "?")

            self._tree.insert(
                "", "end", iid=str(f.index),
                values=(
                    f.index, f.name,
                    f"[{t_label}] {t_name}", sz_str,
                    f"{f.similarity_score:.3f}",
                    f"{f.face_score:.3f}",
                    f"{f.body_score:.3f}",
                    status
                ),
                tags=(t_tag, s_tag)
            )

        self._batch_pos = end

        if end < len(self._filtered):
            self.after(1, self._insert_batch)

    def _sort(self, col):
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = False
        self._apply_filters()

    def _on_search(self, *args):
        if hasattr(self, "_search_id"):
            self.after_cancel(self._search_id)
        self._search_id = self.after(300, self._apply_filters)

    def _on_filter(self, event=None):
        self._apply_filters()

    def _on_goto(self, event=None):
        try:
            idx = int(self._goto_var.get().strip())
            iid = str(idx)
            if self._tree.exists(iid):
                self._tree.see(iid)
                self._tree.selection_set(iid)
                self._tree.focus(iid)
                self._show_frame_detail(idx)
        except (ValueError, tk.TclError):
            pass

    # ------------------------------------------------------------------
    # Selection / Decode
    # ------------------------------------------------------------------

    def _on_select(self, event=None):
        sel = self._tree.selection()
        if sel:
            try:
                idx = int(sel[0])
                self._show_frame_detail(idx)
            except (ValueError, IndexError):
                pass

    def _on_double_click(self, event=None):
        sel = self._tree.selection()
        if sel:
            try:
                idx = int(sel[0])
                self._decode_frame(idx)
            except (ValueError, IndexError):
                pass

    def _decode_current(self):
        sel = self._tree.selection()
        if sel:
            try:
                self._decode_frame(int(sel[0]))
            except (ValueError, IndexError):
                pass

    def _decode_frame(self, index: int):
        if not self._reader:
            return

        self._status(f"Decoding frame {index}...")
        self._decode_var.set("Decoding...")
        self.update_idletasks()

        def task():
            t0 = time.time()
            img = self._reader.get_frame(index)
            elapsed = time.time() - t0
            return img, elapsed

        def done(result):
            img, elapsed = result
            self._canvas.set_image(img)
            self._canvas.after(50, self._canvas.fit)
            ms = elapsed * 1000
            self._decode_var.set(f"Decoded in {ms:.0f} ms")
            self._status(f"Frame {index} decoded in {ms:.0f} ms")

        def error(e):
            self._decode_var.set("Decode failed")
            messagebox.showerror("Decode Error", str(e))

        self._run_threaded(task, done, error)

    def _prev(self):
        sel = self._tree.selection()
        if not sel:
            return
        item = self._tree.prev(sel[0])
        if item:
            self._tree.selection_set(item)
            self._tree.see(item)
            self._tree.focus(item)
            try:
                idx = int(item)
                self._show_frame_detail(idx)
                self._decode_frame(idx)
            except ValueError:
                pass

    def _next(self):
        sel = self._tree.selection()
        if not sel:
            return
        item = self._tree.next(sel[0])
        if item:
            self._tree.selection_set(item)
            self._tree.see(item)
            self._tree.focus(item)
            try:
                idx = int(item)
                self._show_frame_detail(idx)
                self._decode_frame(idx)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Info panels
    # ------------------------------------------------------------------

    def _update_arch_info(self):
        if not self._reader:
            return
        info = self._reader.info()
        text = (
            f"Frames:    {info['total_frames']:,}\n"
            f"Keyframes: {info['keyframe_count'] + info['forced_keyframe_count']:,}\n"
            f"Interp:    {info['interpolated_count']:,}\n"
            f"Residual:  {info['residual_count']:,}\n"
            f"Codec:     {info['codec']}\n"
            f"Quality:   {info['quality']}\n"
            f"Size:      {info['archive_size_human']}\n"
            f"Original:  {info['original_size_human']}\n"
            f"Reduced:   {info['reduction_percent']:.1f}%\n"
            f"Res:       {info['original_width']}x{info['original_height']}\n"
            f"Created:   {info['created']}\n"
        )
        self._arch_text.config(state="normal")
        self._arch_text.delete("1.0", "end")
        self._arch_text.insert("1.0", text)
        self._arch_text.config(state="disabled")

    def _show_frame_detail(self, index: int):
        if index < 0 or index >= len(self._frames):
            return
        f = self._frames[index]
        t_name = FRAME_TYPE_NAMES.get(f.frame_type, "Unknown")
        t_label = FRAME_TYPE_LABELS.get(f.frame_type, "?")
        total_sz = f.data_size + f.residual_size

        text = (
            f"Frame:      {f.index}\n"
            f"Name:       {f.name}\n"
            f"Type:       [{t_label}] {t_name}\n"
            f"Size:       {human_readable_size(total_sz)}\n"
            f"GOP:        {f.gop_id}\n"
            f"Similarity: {f.similarity_score:.4f}\n"
            f"Face Score: {f.face_score:.4f}\n"
            f"Body Score: {f.body_score:.4f}\n"
            f"Motion:     {f.motion_score:.4f}\n"
            f"Scene Cut:  {'Yes' if f.scene_cut else 'No'}\n"
            f"Parents:    K({f.parent_keyframe_a})"
            f" → K({f.parent_keyframe_b})\n"
            f"Timestep:   {f.interpolation_timestep:.4f}\n"
            f"Deleted:    {'Yes' if f.is_deleted else 'No'}\n"
        )
        self._frame_text.config(state="normal")
        self._frame_text.delete("1.0", "end")
        self._frame_text.insert("1.0", text)
        self._frame_text.config(state="disabled")

    def _update_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory
                mem_gb = mem / (1024 ** 3)
                self._gpu_label.config(
                    text=f"🟢 {name} ({mem_gb:.1f}GB)",
                    fg=COLOR_SUCCESS
                )
            else:
                self._gpu_label.config(
                    text="🔴 CUDA unavailable", fg=COLOR_DANGER
                )
        except Exception:
            self._gpu_label.config(
                text="🔴 GPU error", fg=COLOR_DANGER
            )

    # ------------------------------------------------------------------
    # Extraction actions
    # ------------------------------------------------------------------

    def _extract_selected(self):
        if not self._reader:
            return
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Extract", "Select frame(s) first.")
            return

        indices = [int(s) for s in sel if s.isdigit()]

        if len(indices) == 1:
            path = filedialog.asksaveasfilename(
                title="Save Frame As", defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("WebP", "*.webp"),
                           ("JPEG", "*.jpg")]
            )
            if path:
                try:
                    self._reader.extract_frame(indices[0], path)
                    messagebox.showinfo("Done", f"Saved to:\n{path}")
                except Exception as e:
                    messagebox.showerror("Error", str(e))
        else:
            folder = filedialog.askdirectory(
                title="Select Output Folder"
            )
            if folder:
                self._extract_indices(indices, folder)

    def _extract_range_dialog(self):
        if not self._reader:
            return

        dlg = tk.Toplevel(self)
        dlg.title("Extract Range")
        dlg.geometry("300x180")
        dlg.transient(self)
        dlg.configure(bg=COLOR_BG)

        tk.Label(dlg, text="Start Index:", bg=COLOR_BG,
                 font=("Segoe UI", 10)).pack(pady=(15, 2))
        start_var = tk.IntVar(value=0)
        tk.Entry(dlg, textvariable=start_var, width=10,
                 font=("Segoe UI", 10)).pack()

        tk.Label(dlg, text="End Index:", bg=COLOR_BG,
                 font=("Segoe UI", 10)).pack(pady=(10, 2))
        end_var = tk.IntVar(value=len(self._frames) - 1)
        tk.Entry(dlg, textvariable=end_var, width=10,
                 font=("Segoe UI", 10)).pack()

        def do_it():
            dlg.destroy()
            folder = filedialog.askdirectory(
                title="Output Folder"
            )
            if folder:
                indices = list(
                    range(start_var.get(), end_var.get() + 1)
                )
                self._extract_indices(indices, folder)

        self._btn(dlg, "Extract", COLOR_SUCCESS, do_it).pack(pady=15)

    def _extract_all(self):
        if not self._reader:
            return
        folder = filedialog.askdirectory(
            title="Output Folder for All Frames"
        )
        if folder:
            indices = list(range(len(self._frames)))
            self._extract_indices(indices, folder)

    def _extract_kf_only(self):
        if not self._reader:
            return
        folder = filedialog.askdirectory(
            title="Output Folder for Keyframes"
        )
        if folder:
            indices = [
                f.index for f in self._frames
                if f.frame_type in (
                    FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME
                ) and not f.is_deleted
            ]
            self._extract_indices(indices, folder)

    def _extract_indices(self, indices: List[int], folder: str):
        """Extract list of frame indices with progress dialog."""
        total = len(indices)

        dlg = tk.Toplevel(self)
        dlg.title("Extracting...")
        dlg.geometry("480x160")
        dlg.transient(self)
        dlg.grab_set()
        dlg.configure(bg=COLOR_BG)

        msg_var = tk.StringVar(value="Starting...")
        tk.Label(dlg, textvariable=msg_var, bg=COLOR_BG,
                 font=("Segoe UI", 10)).pack(pady=(15, 5))

        prog_var = tk.DoubleVar(value=0)
        ttk.Progressbar(
            dlg, variable=prog_var, maximum=100, length=420
        ).pack(padx=30, pady=5)

        pct_var = tk.StringVar(value="0%")
        tk.Label(dlg, textvariable=pct_var, bg=COLOR_BG,
                 font=("Segoe UI", 9)).pack()

        cancelled = [False]

        self._btn(
            dlg, "Cancel", COLOR_DANGER,
            lambda: cancelled.__setitem__(0, True)
        ).pack(pady=10)

        def task():
            os.makedirs(folder, exist_ok=True)
            extracted = 0
            errors = 0

            for pos, idx in enumerate(indices):
                if cancelled[0]:
                    break
                if idx < 0 or idx >= len(self._frames):
                    continue

                f = self._frames[idx]
                if f.is_deleted:
                    continue

                name = os.path.splitext(f.name)[0] + ".png"
                out = os.path.join(folder, name)

                try:
                    self._reader.extract_frame(idx, out)
                    extracted += 1
                except Exception:
                    errors += 1

                pct = ((pos + 1) / total) * 100
                self.after(0, lambda m=f"Extracting: {f.name}",
                           p=pct, c=pos + 1: (
                    msg_var.set(m),
                    prog_var.set(p),
                    pct_var.set(f"{c:,}/{total:,} ({p:.0f}%)")
                ))

            return extracted, errors

        def done(result):
            dlg.grab_release()
            dlg.destroy()
            extracted, errors = result
            msg = f"Extracted {extracted:,} frames to:\n{folder}"
            if errors:
                msg += f"\n\n{errors} frames had errors."
            messagebox.showinfo("Done", msg)
            self._status(f"Extracted {extracted:,} frames")

        def error(e):
            dlg.grab_release()
            dlg.destroy()
            messagebox.showerror("Error", str(e))

        self._run_threaded(task, done, error)

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def _verify(self):
        if not self._reader:
            messagebox.showinfo("Verify", "No archive loaded.")
            return
        results = self._reader.verify()
        if results["valid"]:
            messagebox.showinfo(
                "Integrity", f"✅ Archive valid.\n"
                             f"Frames: {results['total_frames']:,}"
            )
        else:
            messagebox.showwarning(
                "Integrity",
                f"⚠ Issues found.\n"
                f"Missing data: {len(results['missing_data'])}"
            )

    # ------------------------------------------------------------------
    # Threading helper
    # ------------------------------------------------------------------

    def _run_threaded(self, task_fn, on_done, on_error):
        """Run task_fn in background thread, call on_done/on_error on main."""
        def _run():
            try:
                result = task_fn()
                self.after(0, lambda: on_done(result))
            except Exception as e:
                self.after(0, lambda: on_error(e))

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _status(self, msg: str):
        self._status_var.set(msg)
        self.update_idletasks()

    def _on_close(self):
        if self._reader:
            self._reader.close()
        self.destroy()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IARC Viewer & Extractor"
    )
    parser.add_argument(
        "archive", nargs="?", default=None,
        help="Path to .iarc archive file"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all frames and exit"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Show archive info and exit"
    )
    parser.add_argument(
        "--extract", nargs=2, metavar=("INDEX", "OUTPUT"),
        help="Extract single frame: --extract 42 output.png"
    )
    parser.add_argument(
        "--extract-range", nargs=3,
        metavar=("START", "END", "FOLDER"),
        help="Extract range: --extract-range 0 100 output/"
    )
    parser.add_argument(
        "--extract-all", metavar="FOLDER",
        help="Extract all frames to folder"
    )
    parser.add_argument(
        "--extract-keyframes", metavar="FOLDER",
        help="Extract only keyframes to folder"
    )
    parser.add_argument(
        "--rife-dir", default=DEFAULT_RIFE_DIR,
        help="Path to RIFE model directory"
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device"
    )
    parser.add_argument(
        "--no-fp16", action="store_true",
        help="Disable FP16 for RIFE"
    )

    args = parser.parse_args()

    fp16 = not args.no_fp16

    # CLI mode — no GUI
    if args.archive and any([
        args.list, args.info, args.extract,
        args.extract_range, args.extract_all,
        args.extract_keyframes
    ]):
        reader = IARCReader(
            args.archive,
            rife_dir=args.rife_dir,
            device=args.device,
            fp16=fp16
        )
        reader.open()

        if args.info:
            info = reader.info()
            for k, v in info.items():
                print(f"  {k}: {v}")
            reader.close()
            return

        if args.list:
            for f in reader.list_frames():
                t = FRAME_TYPE_LABELS.get(f.frame_type, "?")
                d = "DEL" if f.is_deleted else "OK"
                sz = human_readable_size(f.data_size + f.residual_size)
                print(f"  [{t}] {f.index:>6}  {f.name:<30}  "
                      f"{sz:>10}  {d}")
            reader.close()
            return

        if args.extract:
            idx = int(args.extract[0])
            out = args.extract[1]
            print(f"Extracting frame {idx} → {out}")
            reader.extract_frame(idx, out)
            print("Done.")
            reader.close()
            return

        if args.extract_range:
            s, e, folder = (int(args.extract_range[0]),
                            int(args.extract_range[1]),
                            args.extract_range[2])

            def cb(cur, tot, name):
                print(f"  {cur}/{tot}: {name}")

            print(f"Extracting frames {s}-{e} → {folder}/")
            n = reader.extract_range(s, e, folder, callback=cb)
            print(f"Done. Extracted {n} frames.")
            reader.close()
            return

        if args.extract_all:
            folder = args.extract_all

            def cb(cur, tot, name):
                if cur % 100 == 0 or cur == tot:
                    print(f"  {cur:,}/{tot:,}: {name}")

            print(f"Extracting all frames → {folder}/")
            n = reader.extract_all(folder, callback=cb)
            print(f"Done. Extracted {n:,} frames.")
            reader.close()
            return

        if args.extract_keyframes:
            folder = args.extract_keyframes

            def cb(cur, tot, name):
                if cur % 100 == 0 or cur == tot:
                    print(f"  {cur:,}/{tot:,}: {name}")

            print(f"Extracting keyframes → {folder}/")
            n = reader.extract_keyframes_only(folder, callback=cb)
            print(f"Done. Extracted {n:,} keyframes.")
            reader.close()
            return

    # GUI mode
    app = IARCViewerApp(
        archive_path=args.archive if args.archive else None,
        rife_dir=args.rife_dir
    )
    app.mainloop()


if __name__ == "__main__":
    main()

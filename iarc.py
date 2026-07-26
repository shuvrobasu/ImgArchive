#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ImgArchive Studio v1.0
Neural Keyframe Image Archive System (.iarc)
Single-file application: Archiver + Viewer + AI Reconstruction Engine

Usage:
    python imgarchive_studio.py

Requirements:
    torch, torchvision, opencv-python, numpy, pillow
    Optional: insightface, mediapipe, onnxruntime-gpu
"""

import os
import sys
import io
import json
import struct
import hashlib
import shutil
import tempfile
import threading
import traceback
import time
import math
import glob
import re
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import IntEnum, auto
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinter import Canvas, Scrollbar, Menu
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2

warnings.filterwarnings("ignore")

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================

APP_NAME = "ImgArchive Studio"
APP_VERSION = "1.5.0"
ARCHIVE_EXTENSION = ".iarc"
ARCHIVE_MAGIC = b"IARC"
ARCHIVE_FORMAT_VERSION = 1

# Default paths
DEFAULT_RIFE_DIR = r"E:\ai\und\assets\pretrained_models\RIFE"
DEFAULT_RAFT_WEIGHTS = r"E:\ai\und\assets\pretrained_models\raft-things.pth"
DEFAULT_TRANSNET_WEIGHTS = r"E:\ai\und\assets\pretrained_models\models\transnetv2-pytorch-weights.pth"
DEFAULT_FACE_PARSE_WEIGHTS = r"E:\ai\und\assets\pretrained_models\79999_iter.pth"
DEFAULT_YOLO_SEG_WEIGHTS = r"E:\ai\und\assets\pretrained_models\yolo11x-seg.onnx"
DEFAULT_YOLO_POSE_WEIGHTS = r"E:\ai\und\assets\pretrained_models\yolo11x-pose.onnx"
DEFAULT_DEEPLAB_WEIGHTS = r"E:\ai\und\assets\pretrained_models\deeplabv3p-resnet50-human.onnx"
DEFAULT_ARCFACE_WEIGHTS = r"E:\ai\und\assets\pretrained_models\w600k_r50.onnx"
DEFAULT_DEPTH_WEIGHTS_B = r"E:\ai\und\assets\pretrained_models\depth_anything_v2_vitb.pth"
DEFAULT_DEPTH_WEIGHTS_L = r"E:\ai\und\assets\pretrained_models\depth_anything_v2_vitl.pth"
DEFAULT_DEPTH_WEIGHTS_S = r"E:\ai\und\assets\pretrained_models\depth_anything_v2_vits.pth"
DEFAULT_REALESRGAN_X2 = r"E:\ai\und\assets\pretrained_models\RealESRGAN_x2.pth"
DEFAULT_REALESRGAN_X4 = r"E:\ai\und\assets\pretrained_models\RealESRGAN_x4.pth"
DEFAULT_REALESRGAN_X4PLUS = r"E:\ai\und\assets\pretrained_models\RealESRGAN_x4plus.pth"
DEFAULT_REALESRGAN_X8 = r"E:\ai\und\assets\pretrained_models\RealESRGAN_x8.pth"
DEFAULT_SWINIR_X4 = r"E:\ai\und\assets\pretrained_models\SwinIR_4x.pth"
DEFAULT_GFPGAN_WEIGHTS = r"E:\ai\assets\pretrained_models\models\GFPGANv1.4.pth"
DEFAULT_FACE_DETECT_CAFFE = r"E:\ai\und\assets\pretrained_models\face_mask_detection.caffemodel"
DEFAULT_FACE_DETECT_PROTO = r"E:\ai\und\assets\pretrained_models\face_mask_detection.prototxt"

RAFT_WEIGHT_OPTIONS = {
    "raft-things": r"E:\ai\und\assets\pretrained_models\raft-things.pth",
    "raft-chairs": r"E:\ai\und\assets\pretrained_models\raft-chairs.pth",
    "raft-kitti": r"E:\ai\und\assets\pretrained_models\raft-kitti.pth",
    "raft-sintel": r"E:\ai\und\assets\pretrained_models\raft-sintel.pth",
    "raft-small": r"E:\ai\und\assets\pretrained_models\raft-small.pth",
}

DEPTH_WEIGHT_OPTIONS = {
    "vitb": DEFAULT_DEPTH_WEIGHTS_B,
    "vitl": DEFAULT_DEPTH_WEIGHTS_L,
    "vits": DEFAULT_DEPTH_WEIGHTS_S,
}

UPSCALER_OPTIONS = {
    "Off": None,
    "RealESRGAN x2": DEFAULT_REALESRGAN_X2,
    "RealESRGAN x4": DEFAULT_REALESRGAN_X4,
    "RealESRGAN x4+": DEFAULT_REALESRGAN_X4PLUS,
    "RealESRGAN x8": DEFAULT_REALESRGAN_X8,
    "SwinIR x4": DEFAULT_SWINIR_X4,
}

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp",".jpe",".gif"}

# UI Color constants (Light theme)
COLOR_BG = "#F5F5F5"
COLOR_BG_WHITE = "#FFFFFF"
COLOR_BG_PANEL = "#E8EAF6"
COLOR_BG_HEADER = "#C5CAE9"
COLOR_TEXT = "#212121"
COLOR_TEXT_SECONDARY = "#616161"
COLOR_ACCENT = "#3F51B5"
COLOR_ACCENT_LIGHT = "#7986CB"
COLOR_SUCCESS = "#4CAF50"
COLOR_WARNING = "#FF9800"
COLOR_DANGER = "#F44336"
COLOR_INFO = "#2196F3"
COLOR_BTN_PRIMARY = "#3F51B5"
COLOR_BTN_SUCCESS = "#4CAF50"
COLOR_BTN_WARNING = "#FF9800"
COLOR_BTN_DANGER = "#F44336"
COLOR_BTN_INFO = "#2196F3"
COLOR_BTN_TEXT = "#FFFFFF"
COLOR_KEYFRAME = "#4CAF50"
COLOR_INTERPOLATED = "#2196F3"
COLOR_RESIDUAL = "#FF9800"
COLOR_FORCED = "#9C27B0"
COLOR_DELETED = "#F44336"
COLOR_RISK = "#E91E63"

# Frame type constants
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

FRAME_TYPE_COLORS = {
    FrameType.KEYFRAME: COLOR_KEYFRAME,
    FrameType.INTERPOLATED: COLOR_INTERPOLATED,
    FrameType.RESIDUAL: COLOR_RESIDUAL,
    FrameType.FORCED_KEYFRAME: COLOR_FORCED,
    FrameType.DELETED: COLOR_DELETED,
}

FRAME_TYPE_NAMES = {
    FrameType.KEYFRAME: "Keyframe",
    FrameType.INTERPOLATED: "Interpolated",
    FrameType.RESIDUAL: "Interpolated + Residual",
    FrameType.FORCED_KEYFRAME: "Forced Keyframe (Face/Body/Cut)",
    FrameType.DELETED: "Deleted",
}


# ============================================================================
# DATA CLASSES — Archive Format Structures
# ============================================================================

@dataclass
class FrameEntry:
    """Represents a single frame entry in the archive index."""
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
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GOPEntry:
    """Group of Pictures metadata."""
    gop_id: int = 0
    start_frame: int = 0
    end_frame: int = 0
    keyframe_indices: List[int] = field(default_factory=list)
    frame_count: int = 0
    has_scene_cut: bool = False


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

    def to_dict(self) -> dict:
        d = asdict(self)
        d["magic"] = self.magic.decode("ascii")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ArchiveHeader":
        d["magic"] = d.get("magic", "IARC").encode("ascii")
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class ArchiveManifest:
    """Complete archive manifest."""
    header: ArchiveHeader = field(default_factory=ArchiveHeader)
    frames: List[FrameEntry] = field(default_factory=list)
    gops: List[GOPEntry] = field(default_factory=list)
    build_settings: Dict[str, Any] = field(default_factory=dict)
    model_versions: Dict[str, str] = field(default_factory=dict)


@dataclass
class AppSettings:
    """Application settings / preferences."""
    # Paths
    rife_model_dir: str = DEFAULT_RIFE_DIR
    raft_weights: str = DEFAULT_RAFT_WEIGHTS
    transnet_weights: str = DEFAULT_TRANSNET_WEIGHTS
    face_parse_weights: str = DEFAULT_FACE_PARSE_WEIGHTS
    yolo_seg_weights: str = DEFAULT_YOLO_SEG_WEIGHTS
    yolo_pose_weights: str = DEFAULT_YOLO_POSE_WEIGHTS
    deeplab_weights: str = DEFAULT_DEEPLAB_WEIGHTS
    arcface_weights: str = DEFAULT_ARCFACE_WEIGHTS
    depth_weights: str = DEFAULT_DEPTH_WEIGHTS_B
    realesrgan_weights: str = DEFAULT_REALESRGAN_X4PLUS
    swinir_weights: str = DEFAULT_SWINIR_X4
    gfpgan_weights: str = DEFAULT_GFPGAN_WEIGHTS
    face_detect_caffe: str = DEFAULT_FACE_DETECT_CAFFE
    face_detect_proto: str = DEFAULT_FACE_DETECT_PROTO
    temp_cache_dir: str = ""

    # Device
    device: str = "cuda"
    precision: str = "fp16"
    decode_batch_size: int = 1

    # Archive defaults
    default_extension: str = ARCHIVE_EXTENSION
    default_gop_size: int = 12
    default_quality: int = 92
    default_chroma: str = "4:2:0"

    # Feature toggles
    face_safe_mode: bool = True
    body_safe_mode: bool = True
    keep_residuals: bool = True
    residual_strength: str = "medium"
    use_scene_cut_detection: bool = True
    use_optical_flow: bool = True
    use_identity_check: bool = True
    use_archive_downscale: bool = False
    downscale_factor: float = 0.5
    use_depth_aware: bool = False

    # Decode toggles
    use_rife: bool = True
    upscaler: str = "Off"
    use_gfpgan: str = "Off"
    gfpgan_mode: str = "Off"

    # Thresholds
    similarity_threshold: float = 0.92
    face_similarity_threshold: float = 0.95
    body_similarity_threshold: float = 0.90
    scene_cut_sensitivity: float = 0.5
    identity_mismatch_threshold: float = 0.6

    # Validation
    verify_dimensions: bool = True
    detect_scene_cuts: bool = True
    reject_corrupted: bool = True
    save_build_manifest: bool = True

    # RAFT selector
    raft_model_name: str = "raft-things"
    depth_model_name: str = "vitb"

    # Compression mode
    compression_mode: str = "balanced"
    # Archive mode
    archive_mode: str = "sequence"  # "sequence" or "gallery"

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AppSettings":
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**valid)
        return cls()


SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgarchive_settings.json")

# ============================================================================
# PROJECT FILE (.ias) — Session state
# ============================================================================

@dataclass
class ProjectFile:
    """
    Project file storing session state.
    One .ias per archive, saved alongside .iarc.
    """
    # Paths
    source_folder: str = ""
    archive_path: str = ""
    audio_file: str = ""

    # Build settings snapshot
    codec: str = "webp"
    quality: int = 92
    gop_size: int = 12
    similarity_threshold: float = 0.92
    face_safe: bool = True
    body_safe: bool = True
    keep_residuals: bool = True
    scene_cut_detection: bool = True
    optical_flow: bool = True
    identity_check: bool = True
    archive_downscale: bool = False
    depth_aware: bool = False
    compression_mode: str = "balanced"

    # Export settings
    export_fps: int = 30
    export_format: str = "mp4"

    # Contact sheet settings
    contact_sheet_nth: int = 100
    contact_sheet_thumb_size: int = 128

    # Split settings
    split_mode: str = "range"
    split_size: int = 5000

    # Metadata
    created: str = ""
    last_modified: str = ""
    notes: str = ""
    bookmarks: List[int] = field(default_factory=list)
    tags: Dict[int, str] = field(default_factory=dict)

    def save(self, path: str):
        """Save project to .ias file."""
        self.last_modified = time.strftime("%Y-%m-%d %H:%M:%S")
        if not self.created:
            self.created = self.last_modified
        data = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ProjectFile":
        """Load project from .ias file."""
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid = {k: v for k, v in data.items()
                 if k in cls.__dataclass_fields__}
        return cls(**valid)

    @classmethod
    def from_settings(cls, settings: AppSettings,
                      source_folder: str = "",
                      archive_path: str = "",
                      audio_file: str = "") -> "ProjectFile":
        """Create project from current app settings."""
        return cls(
            source_folder=source_folder,
            archive_path=archive_path,
            audio_file=audio_file,
            codec="webp",
            quality=settings.default_quality,
            gop_size=settings.default_gop_size,
            similarity_threshold=settings.similarity_threshold,
            face_safe=settings.face_safe_mode,
            body_safe=settings.body_safe_mode,
            keep_residuals=settings.keep_residuals,
            scene_cut_detection=settings.use_scene_cut_detection,
            optical_flow=settings.use_optical_flow,
            identity_check=settings.use_identity_check,
            archive_downscale=settings.use_archive_downscale,
            depth_aware=settings.use_depth_aware,
            compression_mode=settings.compression_mode,
        )

    def ias_path_for_archive(self, archive_path: str) -> str:
        """Get .ias path for a given .iarc archive."""
        base = os.path.splitext(archive_path)[0]
        return base + ".ias"


def find_ias_for_iarc(iarc_path: str) -> Optional[str]:
    """Find matching .ias project file for an .iarc archive."""
    ias_path = os.path.splitext(iarc_path)[0] + ".ias"
    if os.path.exists(ias_path):
        return ias_path
    return None


def auto_save_project(archive_path: str, project: ProjectFile):
    """Auto-save .ias alongside .iarc."""
    ias_path = os.path.splitext(archive_path)[0] + ".ias"
    project.archive_path = archive_path
    project.save(ias_path)


# ============================================================================
# RECENT FILES MANAGER
# ============================================================================

class RecentFilesManager:
    """
    Manages a list of recently opened archives.
    Stored in global settings JSON.
    Max 12 entries.
    """

    MAX_RECENT = 12

    def __init__(self, settings_path: str):
        self.settings_path = settings_path
        self._recent: List[str] = []
        self._load()

    def _load(self):
        """Load recent files from settings JSON."""
        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, "r") as f:
                    data = json.load(f)
                self._recent = data.get("recent_files", [])
        except Exception:
            self._recent = []

    def _save(self):
        """Save recent files to settings JSON."""
        try:
            data = {}
            if os.path.exists(self.settings_path):
                with open(self.settings_path, "r") as f:
                    data = json.load(f)
            data["recent_files"] = self._recent[:self.MAX_RECENT]
            with open(self.settings_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def add(self, path: str):
        """Add a file to recent list."""
        path = os.path.abspath(path)
        if path in self._recent:
            self._recent.remove(path)
        self._recent.insert(0, path)
        self._recent = self._recent[:self.MAX_RECENT]
        self._save()

    def get_list(self) -> List[str]:
        """Get list of recent files (most recent first)."""
        return self._recent.copy()

    def clear(self):
        """Clear recent files."""
        self._recent = []
        self._save()

    def remove_missing(self):
        """Remove entries that no longer exist on disk."""
        self._recent = [p for p in self._recent if os.path.exists(p)]
        self._save()


# ============================================================================
# KEYBOARD SHORTCUT MANAGER
# ============================================================================

class ShortcutManager:
    """
    Global keyboard shortcut bindings.
    All shortcuts work regardless of which tab is active.
    """

    def __init__(self, app: tk.Tk):
        self.app = app
        self._bindings: Dict[str, Any] = {}

    def bind(self, key: str, callback, description: str = ""):
        """Bind a keyboard shortcut globally."""
        self._bindings[key] = {
            "callback": callback,
            "description": description
        }
        self.app.bind_all(key, lambda e: callback())

    def unbind(self, key: str):
        """Remove a keyboard shortcut."""
        if key in self._bindings:
            self.app.unbind_all(key)
            del self._bindings[key]

    def get_all(self) -> Dict[str, str]:
        """Get all shortcuts as {key: description}."""
        return {k: v["description"] for k, v in self._bindings.items()}


# ============================================================================
# VIDEO EXPORTER — ffmpeg based
# ============================================================================

class VideoExporter:
    """
    Export archive frames to video using ffmpeg subprocess.
    Supports optional audio muxing.
    """

    SUPPORTED_FPS = [24, 29, 30, 60]
    SUPPORTED_FORMATS = ["mp4", "webm", "avi", "mkv"]

    @staticmethod
    def check_ffmpeg() -> bool:
        """Check if ffmpeg is available in PATH."""
        try:
            import subprocess
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW
                if sys.platform == "win32" else 0
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def export(archive_reader,
               output_path: str,
               fps: int = 30,
               audio_path: str = None,
               quality: int = 23,
               callback=None) -> str:
        """
        Export archive to video file.

        Args:
            archive_reader: IARCReader or ArchiveEngine with open archive
            output_path: Output video file path
            fps: Frames per second
            audio_path: Optional audio file to mux
            quality: CRF quality (lower = better, 18-28 typical)
            callback: Optional fn(current, total, message)

        Returns:
            Output video path
        """
        import subprocess
        import tempfile

        frames = archive_reader.list_frames()
        live_frames = [f for f in frames if not f.is_deleted]
        total = len(live_frames)

        if total == 0:
            raise ValueError("No frames to export")

        # Get dimensions from header
        if hasattr(archive_reader, 'get_header'):
            header = archive_reader.get_header()
        elif hasattr(archive_reader, '_header'):
            header = archive_reader._header
        else:
            raise RuntimeError("Cannot determine frame dimensions")

        width = header.original_width
        height = header.original_height

        # Ensure even dimensions for video encoding
        width = width if width % 2 == 0 else width + 1
        height = height if height % 2 == 0 else height + 1

        # Build ffmpeg command — pipe raw frames via stdin
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
        ]

        if audio_path and os.path.exists(audio_path):
            cmd.extend(["-i", audio_path])
            cmd.extend(["-shortest"])

        ext = os.path.splitext(output_path)[1].lower()
        if ext in (".mp4", ".mkv"):
            cmd.extend([
                "-c:v", "libx264",
                "-crf", str(quality),
                "-preset", "medium",
                "-pix_fmt", "yuv420p",
            ])
        elif ext == ".webm":
            cmd.extend([
                "-c:v", "libvpx-vp9",
                "-crf", str(quality),
                "-b:v", "0",
                "-pix_fmt", "yuv420p",
            ])
        elif ext == ".avi":
            cmd.extend([
                "-c:v", "mjpeg",
                "-q:v", "3",
            ])
        else:
            cmd.extend([
                "-c:v", "libx264",
                "-crf", str(quality),
                "-pix_fmt", "yuv420p",
            ])

        if audio_path and os.path.exists(audio_path):
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])

        cmd.append(output_path)

        # Start ffmpeg process
        creationflags = subprocess.CREATE_NO_WINDOW \
            if sys.platform == "win32" else 0

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags
        )

        try:
            for pos, f in enumerate(live_frames):
                if hasattr(archive_reader, 'get_frame'):
                    img = archive_reader.get_frame(f.index)
                elif hasattr(archive_reader, 'extract_frame'):
                    img = archive_reader.extract_frame(f.index)
                else:
                    raise RuntimeError("No frame extraction method")

                # Resize to exact video dimensions if needed
                h, w = img.shape[:2]
                if w != width or h != height:
                    img = cv2.resize(img, (width, height),
                                     interpolation=cv2.INTER_LANCZOS4)

                process.stdin.write(img.tobytes())

                if callback:
                    callback(pos + 1, total, f"Frame {f.index}: {f.name}")

            process.stdin.close()
            process.wait(timeout=60)

        except Exception as e:
            process.kill()
            raise e

        if process.returncode != 0:
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg error: {stderr[:500]}")

        return output_path


# ============================================================================
# CONTACT SHEET GENERATOR
# ============================================================================

class ContactSheetGenerator:
    """
    Generate a thumbnail contact sheet image from an archive.
    Shows every Nth frame in a grid layout.
    """

    @staticmethod
    def generate(archive_reader,
                 output_path: str,
                 nth: int = 100,
                 thumb_size: int = 128,
                 max_cols: int = 0,
                 padding: int = 4,
                 bg_color: Tuple[int, int, int] = (240, 240, 240),
                 add_labels: bool = True,
                 callback=None) -> str:
        """
        Generate contact sheet.

        Args:
            archive_reader: Reader with open archive
            output_path: Output image path (.png/.jpg)
            nth: Show every Nth frame
            thumb_size: Thumbnail width/height in pixels
            max_cols: Max columns (0 = auto-calculate)
            padding: Pixels between thumbnails
            bg_color: Background color (BGR)
            add_labels: Draw frame index below each thumbnail
            callback: Optional fn(current, total, message)

        Returns:
            Output image path
        """
        frames = archive_reader.list_frames()
        live_frames = [f for f in frames if not f.is_deleted]

        # Select every Nth frame
        selected = live_frames[::nth]
        total = len(selected)

        if total == 0:
            raise ValueError("No frames to include in contact sheet")

        # Calculate grid
        if max_cols <= 0:
            max_cols = max(1, int(math.sqrt(total) * 1.5))

        rows = math.ceil(total / max_cols)
        cols = min(total, max_cols)

        label_height = 16 if add_labels else 0
        cell_w = thumb_size + padding
        cell_h = thumb_size + padding + label_height

        sheet_w = cols * cell_w + padding
        sheet_h = rows * cell_h + padding

        # Create canvas
        sheet = np.full(
            (sheet_h, sheet_w, 3), bg_color, dtype=np.uint8
        )

        for pos, f in enumerate(selected):
            row = pos // max_cols
            col = pos % max_cols

            x = padding + col * cell_w
            y = padding + row * cell_h

            try:
                if hasattr(archive_reader, 'get_frame'):
                    img = archive_reader.get_frame(f.index)
                elif hasattr(archive_reader, 'extract_frame'):
                    img = archive_reader.extract_frame(f.index)
                else:
                    continue

                # Resize to thumbnail
                h, w = img.shape[:2]
                scale = thumb_size / max(h, w)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                thumb = cv2.resize(
                    img, (new_w, new_h),
                    interpolation=cv2.INTER_AREA
                )

                # Center in cell
                offset_x = (thumb_size - new_w) // 2
                offset_y = (thumb_size - new_h) // 2

                sheet[
                    y + offset_y:y + offset_y + new_h,
                    x + offset_x:x + offset_x + new_w
                ] = thumb

                # Draw label
                if add_labels:
                    label = f"#{f.index}"
                    label_y = y + thumb_size + label_height - 2
                    cv2.putText(
                        sheet, label,
                        (x + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (80, 80, 80), 1,
                        cv2.LINE_AA
                    )

            except Exception:
                # Draw red X for failed frames
                cv2.rectangle(
                    sheet,
                    (x, y), (x + thumb_size, y + thumb_size),
                    (0, 0, 200), 2
                )
                cv2.line(
                    sheet,
                    (x, y), (x + thumb_size, y + thumb_size),
                    (0, 0, 200), 1
                )

            if callback:
                callback(pos + 1, total, f"Frame {f.index}")

        cv2.imwrite(output_path, sheet)
        return output_path


# ============================================================================
# ARCHIVE SPLITTER
# ============================================================================

class ArchiveSplitter:
    """
    Split an archive into multiple smaller archives.
    Supports splitting by frame range or GOP boundaries.
    Output extensions: .iar001, .iar002, ... .iar999
    """

    MAX_PARTS = 999

    @staticmethod
    def split_by_range(archive_engine,
                       image_paths: List[str],
                       frames_per_part: int,
                       output_base: str,
                       callback=None) -> List[str]:
        """
        Split by fixed frame count.

        Args:
            archive_engine: ArchiveEngine with open archive
            image_paths: Not used for reading, kept for API compat
            frames_per_part: Frames per split file
            output_base: Base path without extension
            callback: Optional fn(current, total, message)

        Returns:
            List of output file paths
        """
        frames = archive_engine.list_frames()
        live_frames = [f for f in frames if not f.is_deleted]
        total = len(live_frames)
        header = archive_engine.get_header()

        if total == 0:
            raise ValueError("No frames to split")

        num_parts = math.ceil(total / frames_per_part)
        if num_parts > ArchiveSplitter.MAX_PARTS:
            raise ValueError(
                f"Would create {num_parts} parts, max is "
                f"{ArchiveSplitter.MAX_PARTS}"
            )

        output_paths = []

        for part in range(num_parts):
            start = part * frames_per_part
            end = min(start + frames_per_part, total)

            part_path = f"{output_base}.iar{part + 1:03d}"
            output_paths.append(part_path)

            part_frames = live_frames[start:end]

            ArchiveSplitter._write_part(
                archive_engine, header, part_frames,
                part_path, part + 1, num_parts, callback
            )

        return output_paths

    @staticmethod
    def split_by_gop(archive_engine,
                     gops_per_part: int,
                     output_base: str,
                     callback=None) -> List[str]:
        """
        Split by GOP boundaries.

        Args:
            archive_engine: ArchiveEngine with open archive
            gops_per_part: Number of GOPs per split file
            output_base: Base path without extension
            callback: Optional fn(current, total, message)

        Returns:
            List of output file paths
        """
        frames = archive_engine.list_frames()
        live_frames = [f for f in frames if not f.is_deleted]
        header = archive_engine.get_header()
        gops = archive_engine.get_gops()

        if not gops:
            raise ValueError("No GOP data available")

        num_parts = math.ceil(len(gops) / gops_per_part)
        if num_parts > ArchiveSplitter.MAX_PARTS:
            raise ValueError(
                f"Would create {num_parts} parts, max is "
                f"{ArchiveSplitter.MAX_PARTS}"
            )

        output_paths = []

        for part in range(num_parts):
            gop_start = part * gops_per_part
            gop_end = min(gop_start + gops_per_part, len(gops))

            # Get frame range for these GOPs
            first_gop = gops[gop_start]
            last_gop = gops[gop_end - 1]

            part_frames = [
                f for f in live_frames
                if first_gop.start_frame <= f.index <= last_gop.end_frame
            ]

            if not part_frames:
                continue

            part_path = f"{output_base}.iar{part + 1:03d}"
            output_paths.append(part_path)

            ArchiveSplitter._write_part(
                archive_engine, header, part_frames,
                part_path, part + 1, num_parts, callback
            )

        return output_paths

    @staticmethod
    def _write_part(archive_engine, header, part_frames,
                    output_path, part_num, total_parts,
                    callback=None):
        """Write a single split part file."""
        total = len(part_frames)

        # Read data blobs from source archive
        data_blobs = {}
        residual_blobs = {}

        src_handle = archive_engine._file_handle
        src_data_start = archive_engine._data_start

        # Remap indices to 0-based for this part
        new_frames = []
        old_to_new = {}

        for new_idx, f in enumerate(part_frames):
            old_to_new[f.index] = new_idx

        for new_idx, f in enumerate(part_frames):
            # Read keyframe data
            if f.frame_type in (
                FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME
            ) and f.data_size > 0:
                src_handle.seek(src_data_start + f.data_offset)
                data_blobs[new_idx] = src_handle.read(f.data_size)

            # Read residual data
            if f.frame_type == FrameType.RESIDUAL \
               and f.residual_size > 0:
                src_handle.seek(src_data_start + f.residual_offset)
                residual_blobs[new_idx] = src_handle.read(
                    f.residual_size
                )

            # Remap parent references
            new_parent_a = old_to_new.get(f.parent_keyframe_a, -1)
            new_parent_b = old_to_new.get(f.parent_keyframe_b, -1)

            nf = FrameEntry(
                index=new_idx,
                name=f.name,
                frame_type=f.frame_type,
                width=f.width,
                height=f.height,
                data_size=f.data_size,
                residual_size=f.residual_size,
                parent_keyframe_a=new_parent_a,
                parent_keyframe_b=new_parent_b,
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

            # Fix orphaned frames — promote to keyframe
            if nf.frame_type in (
                FrameType.INTERPOLATED, FrameType.RESIDUAL
            ):
                if nf.parent_keyframe_a == -1 or \
                   nf.parent_keyframe_b == -1:
                    try:
                        img = archive_engine.extract_frame(f.index)
                        blob = image_to_webp_bytes(
                            img, header.compression_quality
                        )
                        data_blobs[new_idx] = blob
                        nf.frame_type = FrameType.KEYFRAME
                        nf.data_size = len(blob)
                        nf.residual_size = 0
                        nf.parent_keyframe_a = -1
                        nf.parent_keyframe_b = -1
                        if new_idx in residual_blobs:
                            del residual_blobs[new_idx]
                    except Exception:
                        nf.frame_type = FrameType.KEYFRAME
                        nf.data_size = 0

            new_frames.append(nf)

            if callback and new_idx % 100 == 0:
                callback(
                    new_idx + 1, total,
                    f"Part {part_num}/{total_parts}: {f.name}"
                )

        # Compute offsets
        offset = 0
        for i, f in enumerate(new_frames):
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

        # Build part header
        part_header = ArchiveHeader(
            total_frames=len(new_frames),
            keyframe_count=sum(
                1 for f in new_frames
                if f.frame_type == FrameType.KEYFRAME
            ),
            interpolated_count=sum(
                1 for f in new_frames
                if f.frame_type == FrameType.INTERPOLATED
            ),
            residual_count=sum(
                1 for f in new_frames
                if f.frame_type == FrameType.RESIDUAL
            ),
            forced_keyframe_count=sum(
                1 for f in new_frames
                if f.frame_type == FrameType.FORCED_KEYFRAME
            ),
            gop_size=header.gop_size,
            original_width=header.original_width,
            original_height=header.original_height,
            compression_codec=header.compression_codec,
            compression_quality=header.compression_quality,
            archive_downscale=header.archive_downscale,
            downscale_factor=header.downscale_factor,
            use_residuals=header.use_residuals,
            created_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            original_total_bytes=header.original_total_bytes,
        )

        h_json = json.dumps(
            part_header.to_dict(), separators=(",", ":")
        ).encode("utf-8")
        i_json = json.dumps(
            [f.to_dict() for f in new_frames],
            separators=(",", ":")
        ).encode("utf-8")

        # Write part file
        with open(output_path, "wb") as fp:
            fp.write(ARCHIVE_MAGIC)
            fp.write(struct.pack("<I", ARCHIVE_FORMAT_VERSION))
            fp.write(struct.pack("<Q", len(h_json)))
            fp.write(struct.pack("<Q", len(i_json)))
            fp.write(h_json)
            fp.write(i_json)

            for i in range(len(new_frames)):
                if i in data_blobs:
                    fp.write(data_blobs[i])
                if i in residual_blobs:
                    fp.write(residual_blobs[i])

            fp.write(b"\x00" * 64)
            arc_size = fp.tell()

        # Patch checksum
        with open(output_path, "r+b") as fp:
            fp.seek(0)
            all_data = fp.read(arc_size - 64)
            cs = compute_checksum(all_data)
            fp.seek(arc_size - 64)
            fp.write(cs.encode("ascii")[:64].ljust(64, b"\x00"))

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_device(settings: AppSettings) -> str:
    """Get torch device string."""
    import torch
    if settings.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_checksum(data: bytes) -> str:
    """Compute SHA256 checksum of data."""
    return hashlib.sha256(data).hexdigest()


def image_to_webp_bytes(img: np.ndarray, quality: int = 92) -> bytes:
    """Encode numpy image (BGR) to WebP bytes."""
    success, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not success:
        raise RuntimeError("WebP encoding failed")
    return buf.tobytes()


def webp_bytes_to_image(data: bytes) -> np.ndarray:
    """Decode WebP bytes to numpy image (BGR)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("WebP decoding failed")
    return img


def jpeg_bytes_to_image(data: bytes) -> np.ndarray:
    """Decode JPEG bytes to numpy image (BGR)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("JPEG decoding failed")
    return img


def image_to_jpeg_bytes(img: np.ndarray, quality: int = 92) -> bytes:
    """Encode numpy image (BGR) to JPEG bytes."""
    success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def load_image_file(path: str) -> np.ndarray:
    """Load an image file as BGR numpy array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return img


def scan_image_folder(folder: str) -> List[str]:
    """Scan a folder for supported image files, sorted by name."""
    files = []
    for f in sorted(os.listdir(folder)):
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            files.append(os.path.join(folder, f))
    return files


def compute_ssim_gpu(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images. GPU accelerated via torch."""
    import torch
    import torch.nn.functional as F

    def _to_tensor(img):
        t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return t.cuda() if torch.cuda.is_available() else t

    t1 = _to_tensor(img1)
    t2 = _to_tensor(img2)

    if t1.shape != t2.shape:
        t2 = F.interpolate(t2, size=t1.shape[2:], mode="bilinear", align_corners=False)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(t1, 11, 1, 5)
    mu2 = F.avg_pool2d(t2, 11, 1, 5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(t1 ** 2, 11, 1, 5) - mu1_sq
    sigma2_sq = F.avg_pool2d(t2 ** 2, 11, 1, 5) - mu2_sq
    sigma12 = F.avg_pool2d(t1 * t2, 11, 1, 5) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean().cpu())


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Resize image by scale factor."""
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to PIL Image (RGB)."""
    return Image.fromarray(bgr_to_rgb(img))


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image (RGB) to BGR numpy array."""
    return rgb_to_bgr(np.array(img))

# ============================================================================
# AI MODEL MANAGER — Lazy loading, GPU default
# ============================================================================

class ModelManager:
    """
    Centralized model loader and cache.
    All models are loaded lazily on first use and cached.
    GPU is the default device for all operations.
    """

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.device = get_device(settings)
        self.use_fp16 = settings.precision == "fp16"
        self._models = {}
        self._lock = threading.Lock()

    def _get_or_load(self, key: str, loader_fn):
        with self._lock:
            if key not in self._models:
                self._models[key] = loader_fn()
            return self._models[key]

    def unload_all(self):
        with self._lock:
            self._models.clear()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self, key: str):
        with self._lock:
            if key in self._models:
                del self._models[key]
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # RIFE — Frame Interpolation
    # ------------------------------------------------------------------

    def _load_rife(self):
        import torch
        rife_dir = self.settings.rife_model_dir
        if not os.path.isdir(rife_dir):
            raise FileNotFoundError(
                f"RIFE model directory not found: {rife_dir}"
            )

        if rife_dir not in sys.path:
            sys.path.insert(0, rife_dir)

        try:
            from model.RIFE_HDv3 import Model as RIFEModel
        except ImportError:
            try:
                from RIFE_HDv3 import Model as RIFEModel
            except ImportError:
                try:
                    from model.RIFE import Model as RIFEModel
                except ImportError:
                    raise ImportError(
                        f"Cannot import RIFE model from {rife_dir}. "
                        "Expected model/RIFE_HDv3.py or similar."
                    )

        model = RIFEModel()
        model.load_model(rife_dir, -1)
        model.eval()

        if self.device == "cuda":
            model.device()  # Move model to CUDA

            # Convert entire model to FP16 if requested
            # This must be done AFTER device() call
            # and must include ALL submodules + buffers
            if self.use_fp16:
                model.flownet = model.flownet.half()

        return model

    def get_rife(self):
        return self._get_or_load("rife", self._load_rife)

    def rife_interpolate(
            self,
            img1: np.ndarray,
            img2: np.ndarray,
            timestep: float = 0.5
    ) -> np.ndarray:
        """
        Interpolate between two BGR images using RIFE.
        FP16/FP32 consistent — model and inputs always same dtype.
        Returns BGR numpy array.
        """
        import torch

        model = self.get_rife()

        use_fp16 = self.use_fp16 and self.device == "cuda"

        def _np_to_tensor(img: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(
                img.copy()
            ).permute(2, 0, 1).float() / 255.0
            t = t.unsqueeze(0)
            if self.device == "cuda":
                t = t.cuda()
            if use_fp16:
                t = t.half()
            return t

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Both images must be same size for RIFE
        # If different, use the larger canvas and letterbox both
        mismatched = (h1 != h2 or w1 != w2)

        if mismatched:
            canvas_h = max(h1, h2)
            canvas_w = max(w1, w2)

            def _letterbox(img, target_h, target_w):
                """Resize preserving aspect ratio, pad to target."""
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

        # RIFE requires dimensions divisible by 64
        pad_h = ((h - 1) // 64 + 1) * 64
        pad_w = ((w - 1) // 64 + 1) * 64

        need_pad = (h != pad_h or w != pad_w)

        if need_pad:
            img1_in = cv2.resize(img1, (pad_w, pad_h),
                                 interpolation=cv2.INTER_LINEAR)
            img2_in = cv2.resize(img2, (pad_w, pad_h),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            img1_in = img1
            img2_in = img2

        t1 = _np_to_tensor(img1_in)
        t2 = _np_to_tensor(img2_in)

        with torch.no_grad():
            try:
                mid = model.inference(t1, t2, timestep=timestep)
            except TypeError:
                # Older RIFE versions without timestep arg
                mid = model.inference(t1, t2)

        # Always convert output back to FP32 before numpy
        result = (
                mid[0]
                .float()
                .clamp(0, 1)
                .cpu()
                .permute(1, 2, 0)
                .numpy() * 255
        ).astype(np.uint8)

        # if need_pad:
        #     result = cv2.resize(result, (w1, h1),
        #                         interpolation=cv2.INTER_LINEAR)
        if need_pad:
            result = cv2.resize(result, (w, h),
                                interpolation=cv2.INTER_LINEAR)

        # If images were mismatched, crop back to img1's original size
        if mismatched:
            x_off, y_off, new_w, new_h = box1
            result = result[y_off:y_off + new_h,
                            x_off:x_off + new_w]
            result = cv2.resize(result, (w1, h1),
                                interpolation=cv2.INTER_LANCZOS4)

        return result

    # ------------------------------------------------------------------
    # RAFT — Optical Flow
    # ------------------------------------------------------------------

    def _load_raft(self):
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

        weights_path = self.settings.raft_weights
        model_name = self.settings.raft_model_name

        if "small" in model_name:
            model = raft_small(weights=None)
        else:
            model = raft_large(weights=None)

        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            cleaned = {}
            for k, v in state.items():
                new_key = k.replace("module.", "")
                cleaned[new_key] = v
            try:
                model.load_state_dict(cleaned, strict=False)
            except Exception:
                pass

        model.eval()
        if self.device == "cuda":
            model = model.cuda()
        if self.use_fp16:
            model = model.half()

        return model

    def get_raft(self):
        return self._get_or_load("raft", self._load_raft)

    def compute_optical_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Compute optical flow between two BGR images. Returns flow array."""
        import torch

        model = self.get_raft()

        def _prep(img):
            t = torch.from_numpy(bgr_to_rgb(img)).permute(2, 0, 1).float().unsqueeze(0)
            if self.device == "cuda":
                t = t.cuda()
            if self.use_fp16:
                t = t.half()
            return t

        h, w = img1.shape[:2]
        pad_h = ((h - 1) // 8 + 1) * 8
        pad_w = ((w - 1) // 8 + 1) * 8
        i1 = cv2.resize(img1, (pad_w, pad_h))
        i2 = cv2.resize(img2, (pad_w, pad_h))

        t1 = _prep(i1)
        t2 = _prep(i2)

        with torch.no_grad():
            flow_list = model(t1, t2)
            flow = flow_list[-1]

        return flow[0].float().cpu().permute(1, 2, 0).numpy()

    def compute_motion_score(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute a scalar motion score from optical flow magnitude."""
        try:
            flow = self.compute_optical_flow(img1, img2)
            magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
            return float(np.mean(magnitude))
        except Exception:
            diff = cv2.absdiff(img1, img2)
            return float(np.mean(diff))

    # ------------------------------------------------------------------
    # TransNetV2 — Scene Cut Detection
    # ------------------------------------------------------------------

    def _load_transnet(self):
        import torch
        import torch.nn as nn

        class TransNetV2Simple(nn.Module):
            """Simplified TransNetV2 for scene boundary detection."""
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), padding=(0, 3, 3))
                self.bn1 = nn.BatchNorm3d(64)
                self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.bn2 = nn.BatchNorm3d(128)
                self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
                self.fc = nn.Linear(128, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.pool(x)
                x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)
                x = self.sigmoid(self.fc(x))
                return x.squeeze(-1)

        model = TransNetV2Simple()
        weights_path = self.settings.transnet_weights
        if os.path.exists(weights_path):
            try:
                state = torch.load(weights_path, map_location="cpu", weights_only=False)
                if isinstance(state, dict):
                    try:
                        model.load_state_dict(state, strict=False)
                    except Exception:
                        pass
            except Exception:
                pass

        model.eval()
        if self.device == "cuda":
            model = model.cuda()
        return model

    def get_transnet(self):
        return self._get_or_load("transnet", self._load_transnet)

    def detect_scene_cuts_batch(self, images: List[np.ndarray],
                                threshold: float = 0.5) -> List[bool]:
        """
        Detect scene cuts in a list of sequential images.
        Returns list of booleans (True = scene cut before this frame).
        Fallback: uses histogram comparison if TransNet fails.
        """
        results = [False] * len(images)
        if len(images) < 2:
            return results

        try:
            for i in range(1, len(images)):
                hist1 = cv2.calcHist([images[i - 1]], [0, 1, 2], None,
                                     [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist2 = cv2.calcHist([images[i]], [0, 1, 2], None,
                                     [8, 8, 8], [0, 256, 0, 256, 0, 256])
                cv2.normalize(hist1, hist1)
                cv2.normalize(hist2, hist2)
                corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                if corr < (1.0 - threshold):
                    results[i] = True
        except Exception:
            pass

        return results

    # ------------------------------------------------------------------
    # InsightFace — Face Detection & Landmarks
    # ------------------------------------------------------------------

    def _load_insightface(self):
        try:
            import logging
            # Suppress InsightFace verbose logging
            logging.getLogger("insightface").setLevel(logging.ERROR)

            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name="buffalo_l",
                root=os.path.dirname(self.settings.arcface_weights),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"]
            )
            app.prepare(
                ctx_id=0 if self.device == "cuda" else -1,
                det_size=(320, 320)  # 320 is enough for detection, faster than 640
            )
            return app
        except Exception:
            return None

    def get_insightface(self):
        return self._get_or_load("insightface", self._load_insightface)

    def detect_faces(self, img: np.ndarray) -> list:
        """Detect faces in a BGR image. Returns list of face objects."""
        app = self.get_insightface()
        if app is None:
            return self._detect_faces_cv(img)
        try:
            faces = app.get(img)
            return faces
        except Exception:
            return self._detect_faces_cv(img)

    def _detect_faces_cv(self, img: np.ndarray) -> list:
        """Fallback face detection using OpenCV DNN."""
        proto = self.settings.face_detect_proto
        caffe = self.settings.face_detect_caffe
        if not (os.path.exists(proto) and os.path.exists(caffe)):
            return []
        try:
            net = cv2.dnn.readNetFromCaffe(proto, caffe)
            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 123))
            net.setInput(blob)
            dets = net.forward()
            faces = []
            for i in range(dets.shape[2]):
                conf = dets[0, 0, i, 2]
                if conf > 0.5:
                    box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                    faces.append({"bbox": box.astype(int), "confidence": float(conf)})
            return faces
        except Exception:
            return []

    def compute_face_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute face-region similarity between two frames."""
        faces1 = self.detect_faces(img1)
        faces2 = self.detect_faces(img2)

        if not faces1 or not faces2:
            return 1.0

        try:
            f1 = faces1[0]
            f2 = faces2[0]

            if hasattr(f1, "bbox"):
                b1 = f1.bbox.astype(int)
                b2 = f2.bbox.astype(int)
            elif isinstance(f1, dict):
                b1 = f1["bbox"]
                b2 = f2["bbox"]
            else:
                return 1.0

            crop1 = img1[max(0, b1[1]):b1[3], max(0, b1[0]):b1[2]]
            crop2 = img2[max(0, b2[1]):b2[3], max(0, b2[0]):b2[2]]

            if crop1.size == 0 or crop2.size == 0:
                return 1.0

            crop2 = cv2.resize(crop2, (crop1.shape[1], crop1.shape[0]))
            return compute_ssim_gpu(crop1, crop2)
        except Exception:
            return 1.0

    # ------------------------------------------------------------------
    # YOLO — Body Segmentation & Pose
    # ------------------------------------------------------------------

    def _load_yolo_seg(self):
        try:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = \
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2  # Prevent CPU thrashing
            sess_options.inter_op_num_threads = 2
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(
                self.settings.yolo_seg_weights,
                sess_options=sess_options,
                providers=providers
            )
            return sess
        except Exception:
            return None

    def _load_yolo_pose(self):
        try:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = \
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 2
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(
                self.settings.yolo_pose_weights,
                sess_options=sess_options,
                providers=providers
            )
            return sess
        except Exception:
            return None

    def get_yolo_seg(self):
        return self._get_or_load("yolo_seg", self._load_yolo_seg)

    def get_yolo_pose(self):
        return self._get_or_load("yolo_pose", self._load_yolo_pose)

    def compute_body_score(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute body region similarity between two frames.
        Uses body segmentation or falls back to full-image SSIM.
        """
        try:
            mask1 = self._get_body_mask(img1)
            mask2 = self._get_body_mask(img2)

            if mask1 is None or mask2 is None:
                return compute_ssim_gpu(img1, img2)

            body1 = cv2.bitwise_and(img1, img1, mask=mask1)
            body2 = cv2.bitwise_and(img2, img2, mask=mask2)

            if np.sum(mask1) < 100 or np.sum(mask2) < 100:
                return 1.0

            return compute_ssim_gpu(body1, body2)
        except Exception:
            return 1.0

    def _get_body_mask(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Get binary body mask using DeepLab. GPU enforced, thread-safe."""
        try:
            import onnxruntime as ort
            if not os.path.exists(self.settings.deeplab_weights):
                return None

            if "deeplab_sess" not in self._models:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 2
                sess_options.inter_op_num_threads = 2
                self._models["deeplab_sess"] = ort.InferenceSession(
                    self.settings.deeplab_weights,
                    sess_options=sess_options,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )

            sess = self._models["deeplab_sess"]
            input_name = sess.get_inputs()[0].name
            input_shape = sess.get_inputs()[0].shape
            h, w = input_shape[2], input_shape[3]
            resized = cv2.resize(img, (w, h))
            blob = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
            output = sess.run(None, {input_name: blob})[0]
            mask = (output[0].argmax(axis=0) > 0).astype(np.uint8) * 255
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            return mask
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Identity Consistency — ArcFace embedding comparison
    # ------------------------------------------------------------------

    def compute_identity_embedding(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Extract face identity embedding from image."""
        app = self.get_insightface()
        if app is None:
            return None
        try:
            faces = app.get(img)
            if faces:
                return faces[0].embedding
        except Exception:
            pass
        return None

    def check_identity_consistency(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Check if faces in two images are the same person.
        Returns cosine similarity (1.0 = same, 0.0 = different).
        """
        emb1 = self.compute_identity_embedding(img1)
        emb2 = self.compute_identity_embedding(img2)

        if emb1 is None or emb2 is None:
            return 1.0

        cos_sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
        return max(0.0, cos_sim)

    # ------------------------------------------------------------------
    # Upscalers — RealESRGAN / SwinIR
    # ------------------------------------------------------------------

    def _load_realesrgan(self):
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            weights = self.settings.realesrgan_weights
            if not os.path.exists(weights):
                return None

            if "x2" in weights.lower():
                scale = 2
            elif "x8" in weights.lower():
                scale = 8
            else:
                scale = 4

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=scale)

            upsampler = RealESRGANer(
                scale=scale,
                model_path=weights,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=self.use_fp16 and self.device == "cuda",
                gpu_id=0 if self.device == "cuda" else None
            )
            return upsampler
        except Exception:
            return None

    def get_realesrgan(self):
        return self._get_or_load("realesrgan", self._load_realesrgan)

    def upscale_image(self, img: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Upscale image using selected upscaler. Returns BGR."""
        upscaler_name = self.settings.upscaler
        if upscaler_name == "Off":
            return img

        upsampler = self.get_realesrgan()
        if upsampler is None:
            return img

        try:
            output, _ = upsampler.enhance(img, outscale=None)
            if target_size:
                output = cv2.resize(output, target_size, interpolation=cv2.INTER_LANCZOS4)
            return output
        except Exception:
            return img

    # ------------------------------------------------------------------
    # GFPGAN — Face Enhancement
    # ------------------------------------------------------------------

    def _load_gfpgan(self):
        try:
            from gfpgan import GFPGANer
            weights = self.settings.gfpgan_weights
            if not os.path.exists(weights):
                return None

            restorer = GFPGANer(
                model_path=weights,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None
            )
            return restorer
        except Exception:
            return None

    def get_gfpgan(self):
        return self._get_or_load("gfpgan", self._load_gfpgan)

    def enhance_face(self, img: np.ndarray) -> np.ndarray:
        """Enhance faces in image using GFPGAN. Returns BGR."""
        mode = self.settings.gfpgan_mode
        if mode == "Off":
            return img

        restorer = self.get_gfpgan()
        if restorer is None:
            return img

        try:
            _, _, output = restorer.enhance(img, has_aligned=False, only_center_face=False)
            return output
        except Exception:
            return img

# ============================================================================
# ARCHIVE ENGINE — .iarc File I/O
# ============================================================================

class ArchiveEngine:
    """
    Handles all .iarc file operations:
    - Create archive from analyzed frame data
    - Read archive header/index
    - Random access frame extraction
    - Delete/restore/compact operations
    - Integrity verification

    File Layout:
    ┌─────────────────────────────┐
    │  MAGIC (4 bytes) "IARC"     │
    │  VERSION (4 bytes)          │
    │  HEADER_JSON_SIZE (8 bytes) │
    │  HEADER_JSON (variable)     │
    │  INDEX_JSON_SIZE (8 bytes)  │
    │  INDEX_JSON (variable)      │
    │  DATA_BLOCKS (variable)     │
    │    - keyframe blobs         │
    │    - residual blobs         │
    │  FOOTER_CHECKSUM (64 bytes) │
    └─────────────────────────────┘
    """

    def __init__(self, model_manager: ModelManager, settings: AppSettings):
        self.models = model_manager
        self.settings = settings
        self._current_archive_path: Optional[str] = None
        self._header: Optional[ArchiveHeader] = None
        self._frames: List[FrameEntry] = []
        self._gops: List[GOPEntry] = []
        self._file_handle: Optional[io.BufferedReader] = None
        self._data_start: int = 0

    def close(self):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    # ------------------------------------------------------------------
    # ANALYSIS — Scan source frames, compute compression plan
    # ------------------------------------------------------------------

    def analyze_sequence(self, image_paths: List[str],
                         progress_callback=None) -> Tuple[ArchiveHeader, List[FrameEntry], List[GOPEntry]]:
        """
        Analyze a sequence of images to produce a compression plan.
        Returns header, frame entries, and GOP entries.
        """
        total = len(image_paths)
        if total == 0:
            raise ValueError("No images to analyze")

        first_img = load_image_file(image_paths[0])
        orig_h, orig_w = first_img.shape[:2]

        frames: List[FrameEntry] = []
        gops: List[GOPEntry] = []
        gop_size = self.settings.default_gop_size

        # Load all images for analysis (in batches to save memory)
        images_cache = {}
        scene_cuts = [False] * total
        similarity_scores = [1.0] * total
        motion_scores = [0.0] * total
        face_scores = [1.0] * total
        body_scores = [1.0] * total
        identity_scores = [1.0] * total

        def _report(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)

        _report("Loading and analyzing frames...", 0)

        # Phase 1: Load images and compute pairwise metrics
        prev_img = None
        original_total_bytes = 0

        for i, path in enumerate(image_paths):
            img = load_image_file(path)
            images_cache[i] = img
            original_total_bytes += os.path.getsize(path)

            if prev_img is not None:
                # Similarity
                try:
                    similarity_scores[i] = compute_ssim_gpu(prev_img, img)
                except Exception:
                    similarity_scores[i] = 0.95

                # Motion
                if self.settings.use_optical_flow:
                    try:
                        motion_scores[i] = self.models.compute_motion_score(prev_img, img)
                    except Exception:
                        motion_scores[i] = 0.0

                # Face
                if self.settings.face_safe_mode:
                    try:
                        face_scores[i] = self.models.compute_face_similarity(prev_img, img)
                    except Exception:
                        face_scores[i] = 1.0

                # Body
                if self.settings.body_safe_mode:
                    try:
                        body_scores[i] = self.models.compute_body_score(prev_img, img)
                    except Exception:
                        body_scores[i] = 1.0

                # Identity
                if self.settings.use_identity_check:
                    try:
                        identity_scores[i] = self.models.check_identity_consistency(prev_img, img)
                    except Exception:
                        identity_scores[i] = 1.0

            prev_img = img
            pct = int((i + 1) / total * 40)
            _report(f"Analyzing frame {i + 1}/{total}", pct)

        # Phase 2: Scene cut detection
        _report("Detecting scene cuts...", 42)
        if self.settings.use_scene_cut_detection:
            batch_imgs = [images_cache[i] for i in range(total)]
            scene_cuts = self.models.detect_scene_cuts_batch(
                batch_imgs, threshold=self.settings.scene_cut_sensitivity
            )

        # Phase 3: Decide frame types (Keyframe / Interpolated / Residual / Forced)
        _report("Building compression plan...", 50)

        sim_thresh = self.settings.similarity_threshold
        face_thresh = self.settings.face_similarity_threshold
        body_thresh = self.settings.body_similarity_threshold
        id_thresh = self.settings.identity_mismatch_threshold

        gop_id = 0
        gop_start = 0
        gop_keyframes = []
        frames_since_keyframe = 0

        for i in range(total):
            is_first = (i == 0)
            is_scene_cut = scene_cuts[i]
            is_gop_boundary = (frames_since_keyframe >= gop_size)

            force_keyframe = False

            # Force keyframe conditions
            if is_first or is_scene_cut or is_gop_boundary:
                force_keyframe = True

            # Face safety
            if self.settings.face_safe_mode and face_scores[i] < face_thresh:
                force_keyframe = True

            # Body safety
            if self.settings.body_safe_mode and body_scores[i] < body_thresh:
                force_keyframe = True

            # Identity mismatch
            if self.settings.use_identity_check and identity_scores[i] < id_thresh:
                force_keyframe = True

            # Low similarity
            if similarity_scores[i] < sim_thresh:
                force_keyframe = True

            # Decide frame type
            if force_keyframe:
                if is_scene_cut or (face_scores[i] < face_thresh) or (body_scores[i] < body_thresh):
                    ftype = FrameType.FORCED_KEYFRAME
                else:
                    ftype = FrameType.KEYFRAME

                # New GOP if at boundary
                if is_gop_boundary or is_scene_cut or is_first:
                    if i > 0:
                        gops.append(GOPEntry(
                            gop_id=gop_id,
                            start_frame=gop_start,
                            end_frame=i - 1,
                            keyframe_indices=gop_keyframes[:],
                            frame_count=i - gop_start,
                            has_scene_cut=any(scene_cuts[gop_start:i])
                        ))
                    gop_id += 1
                    gop_start = i
                    gop_keyframes = []

                gop_keyframes.append(i)
                frames_since_keyframe = 0
            else:
                # Interpolated or Residual
                need_residual = False
                if self.settings.keep_residuals:
                    if similarity_scores[i] < (sim_thresh + 0.03):
                        need_residual = True
                    if face_scores[i] < (face_thresh + 0.02):
                        need_residual = True
                    if body_scores[i] < (body_thresh + 0.03):
                        need_residual = True

                ftype = FrameType.RESIDUAL if need_residual else FrameType.INTERPOLATED
                frames_since_keyframe += 1

            frame = FrameEntry(
                index=i,
                name=os.path.basename(image_paths[i]),
                frame_type=ftype,
                width=orig_w,
                height=orig_h,
                gop_id=gop_id,
                face_score=face_scores[i],
                body_score=body_scores[i],
                similarity_score=similarity_scores[i],
                motion_score=motion_scores[i],
                scene_cut=scene_cuts[i],
            )
            frames.append(frame)

            pct = 50 + int((i + 1) / total * 30)
            _report(f"Planning frame {i + 1}/{total}", pct)

        # Close last GOP
        if gop_start < total:
            gops.append(GOPEntry(
                gop_id=gop_id,
                start_frame=gop_start,
                end_frame=total - 1,
                keyframe_indices=gop_keyframes[:],
                frame_count=total - gop_start,
                has_scene_cut=any(scene_cuts[gop_start:total])
            ))

        # Phase 4: Assign parent keyframes for interpolated/residual frames
        _report("Assigning interpolation parents...", 82)

        keyframe_indices = [f.index for f in frames
                           if f.frame_type in (FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME)]

        for f in frames:
            if f.frame_type in (FrameType.INTERPOLATED, FrameType.RESIDUAL):
                # Find nearest keyframes before and after
                prev_kf = -1
                next_kf = -1
                for ki in keyframe_indices:
                    if ki < f.index:
                        prev_kf = ki
                    elif ki > f.index and next_kf == -1:
                        next_kf = ki
                        break

                if prev_kf == -1:
                    prev_kf = f.index
                if next_kf == -1:
                    next_kf = f.index

                f.parent_keyframe_a = prev_kf
                f.parent_keyframe_b = next_kf

                if next_kf != prev_kf:
                    f.interpolation_timestep = (f.index - prev_kf) / (next_kf - prev_kf)
                else:
                    f.interpolation_timestep = 0.0
                    f.frame_type = FrameType.KEYFRAME

        # Build header
        kf_count = sum(1 for f in frames if f.frame_type == FrameType.KEYFRAME)
        fkf_count = sum(1 for f in frames if f.frame_type == FrameType.FORCED_KEYFRAME)
        interp_count = sum(1 for f in frames if f.frame_type == FrameType.INTERPOLATED)
        res_count = sum(1 for f in frames if f.frame_type == FrameType.RESIDUAL)

        header = ArchiveHeader(
            total_frames=total,
            keyframe_count=kf_count,
            interpolated_count=interp_count,
            residual_count=res_count,
            forced_keyframe_count=fkf_count,
            gop_count=len(gops),
            gop_size=gop_size,
            original_width=orig_w,
            original_height=orig_h,
            compression_codec="webp",
            compression_quality=self.settings.default_quality,
            archive_downscale=self.settings.use_archive_downscale,
            downscale_factor=self.settings.downscale_factor if self.settings.use_archive_downscale else 1.0,
            use_residuals=self.settings.keep_residuals,
            residual_strength=self.settings.residual_strength,
            face_safe=self.settings.face_safe_mode,
            body_safe=self.settings.body_safe_mode,
            identity_check=self.settings.use_identity_check,
            depth_aware=self.settings.use_depth_aware,
            similarity_threshold=sim_thresh,
            face_threshold=face_thresh,
            body_threshold=body_thresh,
            rife_model=self.settings.rife_model_dir,
            raft_model=self.settings.raft_weights,
            original_total_bytes=original_total_bytes,
        )

        _report("Analysis complete", 100)
        self._images_cache = images_cache
        return header, frames, gops

    # ------------------------------------------------------------------
    # BUILD — Write .iarc archive to disk
    # ------------------------------------------------------------------

    def build_archive(self, output_path: str,
                      image_paths: List[str],
                      header: ArchiveHeader,
                      frames: List[FrameEntry],
                      gops: List[GOPEntry],
                      progress_callback=None) -> str:
        """
        Build the .iarc archive file.
        Returns the output path.
        """
        total = len(frames)
        quality = header.compression_quality
        use_downscale = header.archive_downscale
        downscale = header.downscale_factor

        def _report(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)

        _report("Building archive...", 0)

        images_cache = getattr(self, "_images_cache", {})

        # Encode keyframe/forced-keyframe data
        data_blobs = {}
        residual_blobs = {}

        for i, f in enumerate(frames):
            if f.frame_type in (FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME):
                if i in images_cache:
                    img = images_cache[i]
                else:
                    img = load_image_file(image_paths[i])

                if use_downscale and downscale < 1.0:
                    img = resize_image(img, downscale)

                blob = image_to_webp_bytes(img, quality)
                data_blobs[i] = blob
                f.checksum = compute_checksum(blob)

            elif f.frame_type == FrameType.RESIDUAL:
                if i in images_cache:
                    original = images_cache[i]
                else:
                    original = load_image_file(image_paths[i])

                try:
                    ka = f.parent_keyframe_a
                    kb = f.parent_keyframe_b

                    if ka in images_cache:
                        img_a = images_cache[ka]
                    else:
                        img_a = load_image_file(image_paths[ka])

                    if kb in images_cache:
                        img_b = images_cache[kb]
                    else:
                        img_b = load_image_file(image_paths[kb])

                    if use_downscale and downscale < 1.0:
                        img_a = resize_image(img_a, downscale)
                        img_b = resize_image(img_b, downscale)
                        original = resize_image(original, downscale)

                    interpolated = self.models.rife_interpolate(
                        img_a, img_b, f.interpolation_timestep
                    )

                    residual = cv2.subtract(original, interpolated)
                    res_blob = image_to_webp_bytes(
                        residual, max(50, quality - 20)
                    )
                    residual_blobs[i] = res_blob

                except Exception:
                    if use_downscale and downscale < 1.0:
                        original = resize_image(original, downscale)
                    blob = image_to_webp_bytes(original, quality)
                    data_blobs[i] = blob
                    f.frame_type = FrameType.KEYFRAME
                    f.checksum = compute_checksum(blob)

            pct = int((i + 1) / total * 70)
            _report(f"Encoding frame {i + 1}/{total}", pct)

        # ----------------------------------------------------------------
        # Compute data offsets
        # ----------------------------------------------------------------
        _report("Writing archive...", 72)

        current_offset = 0
        for i, f in enumerate(frames):
            if i in data_blobs:
                f.data_offset = current_offset
                f.data_size = len(data_blobs[i])
                current_offset += f.data_size
            else:
                f.data_offset = 0
                f.data_size = 0

            if i in residual_blobs:
                f.residual_offset = current_offset
                f.residual_size = len(residual_blobs[i])
                current_offset += f.residual_size
            else:
                f.residual_offset = 0
                f.residual_size = 0

        # ----------------------------------------------------------------
        # Serialize header and index JSON
        # ----------------------------------------------------------------
        header.created_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        header_json = json.dumps(
            header.to_dict(), separators=(",", ":")
        ).encode("utf-8")
        index_data = [f.to_dict() for f in frames]
        index_json = json.dumps(
            index_data, separators=(",", ":")
        ).encode("utf-8")
        header.index_size = len(index_json)

        # ----------------------------------------------------------------
        # Write archive — wb only writes, no reading
        # ----------------------------------------------------------------
        with open(output_path, "wb") as fp:
            fp.write(ARCHIVE_MAGIC)
            fp.write(struct.pack("<I", ARCHIVE_FORMAT_VERSION))
            fp.write(struct.pack("<Q", len(header_json)))
            fp.write(struct.pack("<Q", len(index_json)))
            fp.write(header_json)
            fp.write(index_json)

            header.data_start_offset = fp.tell()

            for i in range(total):
                if i in data_blobs:
                    fp.write(data_blobs[i])
                if i in residual_blobs:
                    fp.write(residual_blobs[i])

            # Footer placeholder — 64 zero bytes
            fp.write(b"\x00" * 64)
            archive_size = fp.tell()
        # ----------------------------------------------------------------
        # wb block closed. File is fully written and closed here.
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        # Update header stats now that we know archive_size
        # ----------------------------------------------------------------
        header.archive_total_bytes = archive_size
        if header.original_total_bytes > 0:
            header.reduction_percent = round(
                (1.0 - archive_size / header.original_total_bytes) * 100, 1
            )

        # ----------------------------------------------------------------
        # Reopen as r+b — patch updated header JSON then compute checksum
        # Do both in one open to keep checksum valid
        # ----------------------------------------------------------------
        header_json_updated = json.dumps(
            header.to_dict(), separators=(",", ":")
        ).encode("utf-8")

        with open(output_path, "r+b") as fp:
            # Read old header size
            fp.seek(4 + 4)
            old_header_size = struct.unpack("<Q", fp.read(8))[0]

            # Patch header JSON if it fits in reserved space
            if len(header_json_updated) <= old_header_size:
                fp.seek(4 + 4 + 8 + 8)
                padded = header_json_updated + b" " * (
                        old_header_size - len(header_json_updated)
                )
                fp.write(padded)

            # Compute checksum over everything except footer
            fp.seek(0)
            all_data = fp.read(archive_size - 64)
            archive_checksum = compute_checksum(all_data)

            # Patch footer with real checksum
            fp.seek(archive_size - 64)
            fp.write(
                archive_checksum.encode("ascii")[:64].ljust(64, b"\x00")
            )
        # ----------------------------------------------------------------
        # r+b block closed.
        # ----------------------------------------------------------------

        _report(f"Archive built: {human_readable_size(archive_size)}", 100)

        # Cleanup image cache
        if hasattr(self, "_images_cache"):
            del self._images_cache

        return output_path

    # ------------------------------------------------------------------
    # READ — Open and parse .iarc archive
    # ------------------------------------------------------------------

    def open_archive(self, path: str) -> Tuple[ArchiveHeader, List[FrameEntry], List[GOPEntry]]:
        """Open an .iarc archive and parse its header and index."""
        self.close()

        with open(path, "rb") as fp:
            magic = fp.read(4)
            if magic != ARCHIVE_MAGIC:
                raise ValueError(f"Not a valid .iarc archive (magic: {magic})")

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

        # Reconstruct GOPs from frame data
        gop_map = {}
        for f in self._frames:
            gid = f.gop_id
            if gid not in gop_map:
                gop_map[gid] = GOPEntry(gop_id=gid, start_frame=f.index, end_frame=f.index)
            gop_map[gid].end_frame = max(gop_map[gid].end_frame, f.index)
            gop_map[gid].frame_count += 1
            if f.frame_type in (FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME):
                gop_map[gid].keyframe_indices.append(f.index)
            if f.scene_cut:
                gop_map[gid].has_scene_cut = True

        self._gops = [gop_map[k] for k in sorted(gop_map.keys())]
        self._current_archive_path = path
        self._file_handle = open(path, "rb")

        return self._header, self._frames, self._gops

    # ------------------------------------------------------------------
    # EXTRACT — Decode a single frame on-the-fly
    # ------------------------------------------------------------------

    def extract_frame(self, frame_index: int,
                      _depth: int = 0) -> np.ndarray:
        """
        Extract and reconstruct a single frame from the archive.
        For keyframes: direct decode.
        For interpolated: RIFE reconstruction from parent keyframes.
        For residual: RIFE + residual patch.
        Returns BGR numpy array at original resolution.

        _depth is internal recursion tracker — do not pass manually.
        """
        if not self._file_handle:
            raise RuntimeError("No archive is open")

        if _depth > 10:
            raise RuntimeError(
                f"Recursion depth exceeded decoding frame {frame_index}. "
                f"Archive may have circular parent references."
            )

        if frame_index < 0 or frame_index >= len(self._frames):
            raise IndexError(
                f"Frame index {frame_index} out of range "
                f"(0-{len(self._frames) - 1})"
            )

        f = self._frames[frame_index]

        if f.is_deleted:
            raise RuntimeError(f"Frame {frame_index} is deleted")

        header = self._header

        # -----------------------------------------------------------
        # KEYFRAME or FORCED KEYFRAME — direct decode
        # -----------------------------------------------------------
        if f.frame_type in (FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME):
            if f.data_size <= 0:
                raise RuntimeError(
                    f"Keyframe {frame_index} has no stored data"
                )

            self._file_handle.seek(self._data_start + f.data_offset)
            data = self._file_handle.read(f.data_size)
            img = webp_bytes_to_image(data)

            if header.archive_downscale and header.downscale_factor < 1.0:
                img = cv2.resize(
                    img,
                    (header.original_width, header.original_height),
                    interpolation=cv2.INTER_LANCZOS4
                )
                if self.settings.upscaler != "Off":
                    img = self.models.upscale_image(
                        img,
                        (header.original_width, header.original_height)
                    )

            return img

        # -----------------------------------------------------------
        # INTERPOLATED or RESIDUAL — reconstruct via RIFE
        # -----------------------------------------------------------
        if f.frame_type in (FrameType.INTERPOLATED, FrameType.RESIDUAL):
            pa = f.parent_keyframe_a
            pb = f.parent_keyframe_b

            # Safety: parents must not be self
            # Safety: parents must not be self — resolve instead of crash
            if pa == frame_index:
                pa = self._resolve_to_keyframe(
                    frame_index - 1 if frame_index > 0 else 0,
                    _depth + 1
                )
            if pb == frame_index:
                pb = self._resolve_to_keyframe(
                    frame_index + 1
                    if frame_index < len(self._frames) - 1
                    else frame_index - 1,
                    _depth + 1
                )
            # If still self after resolve, treat as duplicate keyframe
            if pa == frame_index and pb == frame_index:
                # Last resort: find ANY keyframe in the archive
                for kf in self._frames:
                    if kf.index != frame_index and \
                            kf.frame_type in (FrameType.KEYFRAME,
                                              FrameType.FORCED_KEYFRAME) \
                            and not kf.is_deleted and kf.data_size > 0:
                        pa = kf.index
                        pb = kf.index
                        break

            # Safety: parents must not be the same interpolated type
            # Walk up to find actual keyframes if parents are also
            # interpolated (should not happen but defensive)
            pa = self._resolve_to_keyframe(pa, _depth + 1)
            pb = self._resolve_to_keyframe(pb, _depth + 1)

            # Decode parent keyframes
            img_a = self.extract_frame(pa, _depth=_depth + 1)
            img_b = self.extract_frame(pb, _depth=_depth + 1)

            # RIFE interpolation
            interpolated = self.models.rife_interpolate(
                img_a, img_b, f.interpolation_timestep
            )

            # Apply residual patch if present
            if f.frame_type == FrameType.RESIDUAL \
                    and f.residual_size > 0:
                self._file_handle.seek(
                    self._data_start + f.residual_offset
                )
                res_data = self._file_handle.read(f.residual_size)
                residual = webp_bytes_to_image(res_data)

                if residual.shape != interpolated.shape:
                    residual = cv2.resize(
                        residual,
                        (interpolated.shape[1], interpolated.shape[0])
                    )

                interpolated = cv2.add(interpolated, residual)

            return interpolated

        raise RuntimeError(f"Unknown frame type: {f.frame_type}")

    def _resolve_to_keyframe(self, index: int,
                             _depth: int = 0) -> int:
        """
        Given a frame index, walk parent references until we find
        an actual keyframe. Prevents infinite recursion when parents
        are incorrectly set to interpolated frames.
        """
        if _depth > 10:
            return index

        if index < 0 or index >= len(self._frames):
            return index

        f = self._frames[index]

        if f.frame_type in (FrameType.KEYFRAME,
                            FrameType.FORCED_KEYFRAME):
            return index

        # This frame is interpolated but was referenced as a parent.
        # Try to find the nearest actual keyframe instead.
        # Search backward first, then forward.
        for offset in range(1, len(self._frames)):
            # Search backward
            back = index - offset
            if back >= 0:
                bf = self._frames[back]
                if bf.frame_type in (FrameType.KEYFRAME,
                                     FrameType.FORCED_KEYFRAME) \
                        and not bf.is_deleted and bf.data_size > 0:
                    return back

            # Search forward
            fwd = index + offset
            if fwd < len(self._frames):
                ff = self._frames[fwd]
                if ff.frame_type in (FrameType.KEYFRAME,
                                     FrameType.FORCED_KEYFRAME) \
                        and not ff.is_deleted and ff.data_size > 0:
                    return fwd

            # Stop if we've searched far enough
            if back < 0 and fwd >= len(self._frames):
                break

        return index

    def extract_frame_to_file(self, frame_index: int, output_path: str):
        """Extract a frame and save to disk."""
        img = self.extract_frame(frame_index)
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".webp":
            cv2.imwrite(output_path, img, [cv2.IMWRITE_WEBP_QUALITY, 95])
        elif ext in (".jpg", ".jpeg"):
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(output_path, img)

    # ------------------------------------------------------------------
    # LIST / QUERY
    # ------------------------------------------------------------------

    def list_frames(self) -> List[FrameEntry]:
        return self._frames

    def get_frame_entry(self, index: int) -> FrameEntry:
        return self._frames[index]

    def get_header(self) -> ArchiveHeader:
        return self._header

    def get_gops(self) -> List[GOPEntry]:
        return self._gops

    # ------------------------------------------------------------------
    # DELETE / RESTORE / COMPACT
    # ------------------------------------------------------------------

    def mark_deleted(self, frame_index: int):
        self._frames[frame_index].is_deleted = True
        self._frames[frame_index].frame_type = FrameType.DELETED

    def restore_frame(self, frame_index: int, original_type: int = FrameType.KEYFRAME):
        self._frames[frame_index].is_deleted = False
        self._frames[frame_index].frame_type = original_type

    def save_index_changes(self):
        """Rewrite the index in the archive to reflect deletions/restorations."""
        if not self._current_archive_path:
            return

        index_data = [f.to_dict() for f in self._frames]
        new_index_json = json.dumps(index_data, separators=(",", ":")).encode("utf-8")

        self._file_handle.close()

        with open(self._current_archive_path, "r+b") as fp:
            fp.seek(4 + 4)
            header_size = struct.unpack("<Q", fp.read(8))[0]
            old_index_size = struct.unpack("<Q", fp.read(8))[0]

            if len(new_index_json) <= old_index_size:
                fp.seek(4 + 4 + 8 + 8 + header_size)
                padded = new_index_json.ljust(old_index_size, b" ")
                fp.write(padded)

        self._file_handle = open(self._current_archive_path, "rb")

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify archive integrity."""
        results = {
            "valid": True,
            "total_frames": len(self._frames),
            "corrupted_frames": [],
            "missing_data": [],
        }

        for f in self._frames:
            if f.is_deleted:
                continue
            if f.frame_type in (FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME):
                if f.data_size <= 0:
                    results["missing_data"].append(f.index)
                    results["valid"] = False

        return results

    # ------------------------------------------------------------------
    # COMPACT — Remove deleted frames, rebuild archive
    # ------------------------------------------------------------------

    def compact_archive(self, output_path: str = None,
                        progress_callback=None) -> str:
        """
        Compact the archive by removing deleted frames and
        rebuilding a clean contiguous file.
        If output_path is None, compacts in-place (writes to temp,
        then replaces original).

        Returns path to compacted archive.
        """
        if not self._file_handle or not self._frames:
            raise RuntimeError("No archive is open")

        source_path = self._current_archive_path
        if output_path is None:
            output_path = source_path + ".compact.tmp"
            in_place = True
        else:
            in_place = False

        header = self._header
        old_frames = self._frames
        total = len(old_frames)

        def _report(msg, current, total_count):
            if progress_callback:
                progress_callback(msg, current, total_count)

        _report("Compacting: reading live frames...", 0, total)

        # Phase 1: Collect live frames and their data
        live_frames: List[FrameEntry] = []
        data_blobs = {}
        residual_blobs = {}

        # Build old-index → new-index mapping
        old_to_new = {}
        new_index = 0

        for i, f in enumerate(old_frames):
            if f.is_deleted:
                continue
            old_to_new[f.index] = new_index
            new_index += 1

        new_index = 0
        for i, f in enumerate(old_frames):
            if f.is_deleted:
                continue

            # Read stored data for keyframes
            if f.frame_type in (FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME):
                if f.data_size > 0:
                    self._file_handle.seek(self._data_start + f.data_offset)
                    data_blobs[new_index] = self._file_handle.read(f.data_size)

            # Read stored residual data
            if f.frame_type == FrameType.RESIDUAL:
                if f.residual_size > 0:
                    self._file_handle.seek(
                        self._data_start + f.residual_offset
                    )
                    residual_blobs[new_index] = self._file_handle.read(
                        f.residual_size
                    )

            # Create new frame entry with remapped indices
            new_frame = FrameEntry(
                index=new_index,
                name=f.name,
                frame_type=f.frame_type,
                width=f.width,
                height=f.height,
                data_offset=0,
                data_size=f.data_size,
                residual_offset=0,
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
                identity_hash=f.identity_hash,
                is_deleted=False,
                checksum=f.checksum,
            )

            # Fix orphaned interpolated frames whose parents were deleted
            if new_frame.frame_type in (
                FrameType.INTERPOLATED, FrameType.RESIDUAL
            ):
                if new_frame.parent_keyframe_a == -1 or \
                   new_frame.parent_keyframe_b == -1:
                    # Promote to keyframe — need to decode and store
                    try:
                        img = self.extract_frame(f.index)
                        blob = image_to_webp_bytes(
                            img, header.compression_quality
                        )
                        data_blobs[new_index] = blob
                        new_frame.frame_type = FrameType.KEYFRAME
                        new_frame.data_size = len(blob)
                        new_frame.residual_size = 0
                        new_frame.parent_keyframe_a = -1
                        new_frame.parent_keyframe_b = -1
                        new_frame.interpolation_timestep = 0.0
                        if new_index in residual_blobs:
                            del residual_blobs[new_index]
                    except Exception:
                        pass

            live_frames.append(new_frame)
            new_index += 1

            if i % 500 == 0 or i == total - 1:
                _report("Compacting: reading frames...", i + 1, total)

        # Phase 2: Compute new offsets
        _report("Compacting: computing offsets...", 0, 1)

        current_offset = 0
        for i, f in enumerate(live_frames):
            if i in data_blobs:
                f.data_offset = current_offset
                f.data_size = len(data_blobs[i])
                current_offset += f.data_size
            else:
                f.data_offset = 0
                if f.frame_type not in (
                    FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME
                ):
                    f.data_size = 0

            if i in residual_blobs:
                f.residual_offset = current_offset
                f.residual_size = len(residual_blobs[i])
                current_offset += f.residual_size
            else:
                f.residual_offset = 0
                if f.frame_type != FrameType.RESIDUAL:
                    f.residual_size = 0

        # Phase 3: Update header counts
        kf_count = sum(
            1 for f in live_frames
            if f.frame_type == FrameType.KEYFRAME
        )
        fkf_count = sum(
            1 for f in live_frames
            if f.frame_type == FrameType.FORCED_KEYFRAME
        )
        interp_count = sum(
            1 for f in live_frames
            if f.frame_type == FrameType.INTERPOLATED
        )
        res_count = sum(
            1 for f in live_frames
            if f.frame_type == FrameType.RESIDUAL
        )

        new_header = ArchiveHeader(
            total_frames=len(live_frames),
            keyframe_count=kf_count,
            interpolated_count=interp_count,
            residual_count=res_count,
            forced_keyframe_count=fkf_count,
            deleted_count=0,
            gop_count=header.gop_count,
            gop_size=header.gop_size,
            original_width=header.original_width,
            original_height=header.original_height,
            compression_codec=header.compression_codec,
            compression_quality=header.compression_quality,
            archive_downscale=header.archive_downscale,
            downscale_factor=header.downscale_factor,
            use_residuals=header.use_residuals,
            residual_strength=header.residual_strength,
            face_safe=header.face_safe,
            body_safe=header.body_safe,
            identity_check=header.identity_check,
            depth_aware=header.depth_aware,
            similarity_threshold=header.similarity_threshold,
            face_threshold=header.face_threshold,
            body_threshold=header.body_threshold,
            rife_model=header.rife_model,
            raft_model=header.raft_model,
            upscaler_model=header.upscaler_model,
            created_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            original_total_bytes=header.original_total_bytes,
        )

        # Phase 4: Write new archive
        _report("Compacting: writing new archive...", 0, 1)

        header_json = json.dumps(
            new_header.to_dict(), separators=(",", ":")
        ).encode("utf-8")
        index_data = [f.to_dict() for f in live_frames]
        index_json = json.dumps(
            index_data, separators=(",", ":")
        ).encode("utf-8")

        with open(output_path, "wb") as fp:
            fp.write(ARCHIVE_MAGIC)
            fp.write(struct.pack("<I", ARCHIVE_FORMAT_VERSION))
            fp.write(struct.pack("<Q", len(header_json)))
            fp.write(struct.pack("<Q", len(index_json)))
            fp.write(header_json)
            fp.write(index_json)

            for i in range(len(live_frames)):
                if i in data_blobs:
                    fp.write(data_blobs[i])
                if i in residual_blobs:
                    fp.write(residual_blobs[i])

            fp.write(b"\x00" * 64)
            archive_size = fp.tell()

        # Patch header and checksum
        new_header.archive_total_bytes = archive_size
        if new_header.original_total_bytes > 0:
            new_header.reduction_percent = round(
                (1.0 - archive_size / new_header.original_total_bytes)
                * 100, 1
            )

        header_json_updated = json.dumps(
            new_header.to_dict(), separators=(",", ":")
        ).encode("utf-8")

        with open(output_path, "r+b") as fp:
            fp.seek(4 + 4)
            old_header_size = struct.unpack("<Q", fp.read(8))[0]

            if len(header_json_updated) <= old_header_size:
                fp.seek(4 + 4 + 8 + 8)
                padded = header_json_updated + b" " * (
                    old_header_size - len(header_json_updated)
                )
                fp.write(padded)

            fp.seek(0)
            all_data = fp.read(archive_size - 64)
            checksum = compute_checksum(all_data)
            fp.seek(archive_size - 64)
            fp.write(
                checksum.encode("ascii")[:64].ljust(64, b"\x00")
            )

        # Phase 5: If in-place, replace original
        if in_place:
            self.close()
            backup_path = source_path + ".backup"
            try:
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(source_path, backup_path)
                os.rename(output_path, source_path)
                os.remove(backup_path)
            except Exception:
                if os.path.exists(backup_path) and \
                   not os.path.exists(source_path):
                    os.rename(backup_path, source_path)
                raise

            self.open_archive(source_path)
            output_path = source_path

        _report("Compact complete", 1, 1)
        return output_path

    # ------------------------------------------------------------------
    # REPAIR — Fix corrupted archives
    # ------------------------------------------------------------------

    def repair_archive(self, output_path: str = None,
                       progress_callback=None) -> Dict[str, Any]:
        """
        Repair a damaged archive by:
        1. Validating every stored keyframe blob (decode test)
        2. Validating every residual blob (decode test)
        3. Fixing orphaned interpolated frames (missing parents)
        4. Fixing broken GOP chains
        5. Rebuilding index with corrections
        6. Optionally writing repaired archive to output_path

        Returns a report dict with repair details.
        """
        if not self._file_handle or not self._frames:
            raise RuntimeError("No archive is open")

        source_path = self._current_archive_path
        if output_path is None:
            output_path = source_path
            in_place = True
        else:
            in_place = False

        header = self._header
        frames = self._frames
        total = len(frames)

        def _report(msg, current, total_count):
            if progress_callback:
                progress_callback(msg, current, total_count)

        report = {
            "total_frames": total,
            "corrupted_keyframes": [],
            "corrupted_residuals": [],
            "orphaned_frames": [],
            "fixed_frames": [],
            "promoted_to_keyframe": [],
            "demoted_to_interpolated": [],
            "unrecoverable": [],
            "repaired": False,
        }

        # ----------------------------------------------------------------
        # Phase 1: Validate all stored blobs
        # ----------------------------------------------------------------
        _report("Repair Phase 1/4: Validating stored data...", 0, total)

        valid_keyframes = set()

        for i, f in enumerate(frames):
            if f.is_deleted:
                continue

            if f.frame_type in (FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME):
                if f.data_size > 0:
                    try:
                        self._file_handle.seek(
                            self._data_start + f.data_offset
                        )
                        data = self._file_handle.read(f.data_size)
                        img = webp_bytes_to_image(data)
                        if img is not None and img.size > 0:
                            valid_keyframes.add(i)
                        else:
                            report["corrupted_keyframes"].append(i)
                    except Exception:
                        report["corrupted_keyframes"].append(i)
                else:
                    report["corrupted_keyframes"].append(i)

            if f.frame_type == FrameType.RESIDUAL:
                if f.residual_size > 0:
                    try:
                        self._file_handle.seek(
                            self._data_start + f.residual_offset
                        )
                        data = self._file_handle.read(f.residual_size)
                        img = webp_bytes_to_image(data)
                        if img is None or img.size == 0:
                            report["corrupted_residuals"].append(i)
                    except Exception:
                        report["corrupted_residuals"].append(i)

            if i % 200 == 0 or i == total - 1:
                _report(
                    "Repair Phase 1/4: Validating data...",
                    i + 1, total
                )

        # ----------------------------------------------------------------
        # Phase 2: Check parent references
        # ----------------------------------------------------------------
        _report("Repair Phase 2/4: Checking parent refs...", 0, total)

        keyframe_set = set()
        for f in frames:
            if not f.is_deleted and f.frame_type in (
                FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME
            ) and f.index in valid_keyframes:
                keyframe_set.add(f.index)

        for i, f in enumerate(frames):
            if f.is_deleted:
                continue

            if f.frame_type in (FrameType.INTERPOLATED, FrameType.RESIDUAL):
                parent_a_ok = f.parent_keyframe_a in keyframe_set
                parent_b_ok = f.parent_keyframe_b in keyframe_set

                if not parent_a_ok or not parent_b_ok:
                    report["orphaned_frames"].append(i)

            if i % 500 == 0:
                _report(
                    "Repair Phase 2/4: Checking refs...",
                    i + 1, total
                )

        # ----------------------------------------------------------------
        # Phase 3: Apply fixes
        # ----------------------------------------------------------------
        _report("Repair Phase 3/4: Applying fixes...", 0, total)

        needs_rebuild = False

        # Fix corrupted keyframes — demote to interpolated if possible
        for i in report["corrupted_keyframes"]:
            f = frames[i]
            # Find nearest valid keyframes
            prev_kf = -1
            next_kf = -1
            for ki in sorted(keyframe_set):
                if ki < i:
                    prev_kf = ki
                elif ki > i and next_kf == -1:
                    next_kf = ki
                    break

            if prev_kf >= 0 and next_kf >= 0:
                f.frame_type = FrameType.INTERPOLATED
                f.parent_keyframe_a = prev_kf
                f.parent_keyframe_b = next_kf
                f.interpolation_timestep = \
                    (i - prev_kf) / (next_kf - prev_kf)
                f.data_offset = 0
                f.data_size = 0
                report["demoted_to_interpolated"].append(i)
                report["fixed_frames"].append(i)
                needs_rebuild = True
            else:
                report["unrecoverable"].append(i)

        # Fix corrupted residuals — demote to pure interpolated
        for i in report["corrupted_residuals"]:
            f = frames[i]
            f.frame_type = FrameType.INTERPOLATED
            f.residual_offset = 0
            f.residual_size = 0
            report["fixed_frames"].append(i)
            needs_rebuild = True

        # Fix orphaned frames — try to find new parents or promote
        for i in report["orphaned_frames"]:
            f = frames[i]
            prev_kf = -1
            next_kf = -1
            for ki in sorted(keyframe_set):
                if ki < i:
                    prev_kf = ki
                elif ki > i and next_kf == -1:
                    next_kf = ki
                    break

            if prev_kf >= 0 and next_kf >= 0:
                f.parent_keyframe_a = prev_kf
                f.parent_keyframe_b = next_kf
                f.interpolation_timestep = \
                    (i - prev_kf) / (next_kf - prev_kf)
                f.frame_type = FrameType.INTERPOLATED
                f.residual_offset = 0
                f.residual_size = 0
                report["fixed_frames"].append(i)
                needs_rebuild = True
            elif prev_kf >= 0:
                f.parent_keyframe_a = prev_kf
                f.parent_keyframe_b = prev_kf
                f.interpolation_timestep = 0.0
                f.frame_type = FrameType.INTERPOLATED
                report["fixed_frames"].append(i)
                needs_rebuild = True
            else:
                report["unrecoverable"].append(i)

        # ----------------------------------------------------------------
        # Phase 4: Rebuild if fixes were applied
        # ----------------------------------------------------------------
        if needs_rebuild:
            _report("Repair Phase 4/4: Rebuilding archive...", 0, 1)

            # Save updated index
            if in_place:
                self.save_index_changes()
                # Then compact to clean up dead data
                self.compact_archive(
                    output_path=None,
                    progress_callback=lambda m, c, t: _report(
                        f"Repair rebuild: {m}", c, t
                    )
                )
            else:
                # Write repaired copy
                self._frames = frames
                self.save_index_changes()
                self.compact_archive(
                    output_path=output_path,
                    progress_callback=lambda m, c, t: _report(
                        f"Repair rebuild: {m}", c, t
                    )
                )

            report["repaired"] = True
        else:
            report["repaired"] = False

        _report("Repair complete", 1, 1)

        # Deduplicate unrecoverable list
        report["unrecoverable"] = list(set(report["unrecoverable"]))
        report["fixed_frames"] = list(set(report["fixed_frames"]))

        return report

    # ------------------------------------------------------------------
    # COMPACT — Remove deleted frames, rebuild archive
    # ------------------------------------------------------------------

    def compact_archive(self, output_path: str = None,
                        progress_callback=None) -> str:
        """
        Compact the archive by removing deleted frames and
        rebuilding a clean contiguous file.
        If output_path is None, compacts in-place (writes to temp,
        then replaces original).

        Returns path to compacted archive.
        """
        if not self._file_handle or not self._frames:
            raise RuntimeError("No archive is open")

        source_path = self._current_archive_path
        if output_path is None:
            output_path = source_path + ".compact.tmp"
            in_place = True
        else:
            in_place = False

        header = self._header
        old_frames = self._frames
        total = len(old_frames)

        def _report(msg, current, total_count):
            if progress_callback:
                progress_callback(msg, current, total_count)

        _report("Compacting: reading live frames...", 0, total)

        # Phase 1: Collect live frames and their data
        live_frames: List[FrameEntry] = []
        data_blobs = {}
        residual_blobs = {}

        # Build old-index → new-index mapping
        old_to_new = {}
        new_index = 0

        for i, f in enumerate(old_frames):
            if f.is_deleted:
                continue
            old_to_new[f.index] = new_index
            new_index += 1

        new_index = 0
        for i, f in enumerate(old_frames):
            if f.is_deleted:
                continue

            # Read stored data for keyframes
            if f.frame_type in (FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME):
                if f.data_size > 0:
                    self._file_handle.seek(self._data_start + f.data_offset)
                    data_blobs[new_index] = self._file_handle.read(f.data_size)

            # Read stored residual data
            if f.frame_type == FrameType.RESIDUAL:
                if f.residual_size > 0:
                    self._file_handle.seek(
                        self._data_start + f.residual_offset
                    )
                    residual_blobs[new_index] = self._file_handle.read(
                        f.residual_size
                    )

            # Create new frame entry with remapped indices
            new_frame = FrameEntry(
                index=new_index,
                name=f.name,
                frame_type=f.frame_type,
                width=f.width,
                height=f.height,
                data_offset=0,
                data_size=f.data_size,
                residual_offset=0,
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
                identity_hash=f.identity_hash,
                is_deleted=False,
                checksum=f.checksum,
            )

            # Fix orphaned interpolated frames whose parents were deleted
            if new_frame.frame_type in (
                FrameType.INTERPOLATED, FrameType.RESIDUAL
            ):
                if new_frame.parent_keyframe_a == -1 or \
                   new_frame.parent_keyframe_b == -1:
                    # Promote to keyframe — need to decode and store
                    try:
                        img = self.extract_frame(f.index)
                        blob = image_to_webp_bytes(
                            img, header.compression_quality
                        )
                        data_blobs[new_index] = blob
                        new_frame.frame_type = FrameType.KEYFRAME
                        new_frame.data_size = len(blob)
                        new_frame.residual_size = 0
                        new_frame.parent_keyframe_a = -1
                        new_frame.parent_keyframe_b = -1
                        new_frame.interpolation_timestep = 0.0
                        if new_index in residual_blobs:
                            del residual_blobs[new_index]
                    except Exception:
                        pass

            live_frames.append(new_frame)
            new_index += 1

            if i % 500 == 0 or i == total - 1:
                _report("Compacting: reading frames...", i + 1, total)

        # Phase 2: Compute new offsets
        _report("Compacting: computing offsets...", 0, 1)

        current_offset = 0
        for i, f in enumerate(live_frames):
            if i in data_blobs:
                f.data_offset = current_offset
                f.data_size = len(data_blobs[i])
                current_offset += f.data_size
            else:
                f.data_offset = 0
                if f.frame_type not in (
                    FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME
                ):
                    f.data_size = 0

            if i in residual_blobs:
                f.residual_offset = current_offset
                f.residual_size = len(residual_blobs[i])
                current_offset += f.residual_size
            else:
                f.residual_offset = 0
                if f.frame_type != FrameType.RESIDUAL:
                    f.residual_size = 0

        # Phase 3: Update header counts
        kf_count = sum(
            1 for f in live_frames
            if f.frame_type == FrameType.KEYFRAME
        )
        fkf_count = sum(
            1 for f in live_frames
            if f.frame_type == FrameType.FORCED_KEYFRAME
        )
        interp_count = sum(
            1 for f in live_frames
            if f.frame_type == FrameType.INTERPOLATED
        )
        res_count = sum(
            1 for f in live_frames
            if f.frame_type == FrameType.RESIDUAL
        )

        new_header = ArchiveHeader(
            total_frames=len(live_frames),
            keyframe_count=kf_count,
            interpolated_count=interp_count,
            residual_count=res_count,
            forced_keyframe_count=fkf_count,
            deleted_count=0,
            gop_count=header.gop_count,
            gop_size=header.gop_size,
            original_width=header.original_width,
            original_height=header.original_height,
            compression_codec=header.compression_codec,
            compression_quality=header.compression_quality,
            archive_downscale=header.archive_downscale,
            downscale_factor=header.downscale_factor,
            use_residuals=header.use_residuals,
            residual_strength=header.residual_strength,
            face_safe=header.face_safe,
            body_safe=header.body_safe,
            identity_check=header.identity_check,
            depth_aware=header.depth_aware,
            similarity_threshold=header.similarity_threshold,
            face_threshold=header.face_threshold,
            body_threshold=header.body_threshold,
            rife_model=header.rife_model,
            raft_model=header.raft_model,
            upscaler_model=header.upscaler_model,
            created_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            original_total_bytes=header.original_total_bytes,
        )

        # Phase 4: Write new archive
        _report("Compacting: writing new archive...", 0, 1)

        header_json = json.dumps(
            new_header.to_dict(), separators=(",", ":")
        ).encode("utf-8")
        index_data = [f.to_dict() for f in live_frames]
        index_json = json.dumps(
            index_data, separators=(",", ":")
        ).encode("utf-8")

        with open(output_path, "wb") as fp:
            fp.write(ARCHIVE_MAGIC)
            fp.write(struct.pack("<I", ARCHIVE_FORMAT_VERSION))
            fp.write(struct.pack("<Q", len(header_json)))
            fp.write(struct.pack("<Q", len(index_json)))
            fp.write(header_json)
            fp.write(index_json)

            for i in range(len(live_frames)):
                if i in data_blobs:
                    fp.write(data_blobs[i])
                if i in residual_blobs:
                    fp.write(residual_blobs[i])

            fp.write(b"\x00" * 64)
            archive_size = fp.tell()

        # Patch header and checksum
        new_header.archive_total_bytes = archive_size
        if new_header.original_total_bytes > 0:
            new_header.reduction_percent = round(
                (1.0 - archive_size / new_header.original_total_bytes)
                * 100, 1
            )

        header_json_updated = json.dumps(
            new_header.to_dict(), separators=(",", ":")
        ).encode("utf-8")

        with open(output_path, "r+b") as fp:
            fp.seek(4 + 4)
            old_header_size = struct.unpack("<Q", fp.read(8))[0]

            if len(header_json_updated) <= old_header_size:
                fp.seek(4 + 4 + 8 + 8)
                padded = header_json_updated + b" " * (
                    old_header_size - len(header_json_updated)
                )
                fp.write(padded)

            fp.seek(0)
            all_data = fp.read(archive_size - 64)
            checksum = compute_checksum(all_data)
            fp.seek(archive_size - 64)
            fp.write(
                checksum.encode("ascii")[:64].ljust(64, b"\x00")
            )

        # Phase 5: If in-place, replace original
        if in_place:
            self.close()
            backup_path = source_path + ".backup"
            try:
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(source_path, backup_path)
                os.rename(output_path, source_path)
                os.remove(backup_path)
            except Exception:
                if os.path.exists(backup_path) and \
                   not os.path.exists(source_path):
                    os.rename(backup_path, source_path)
                raise

            self.open_archive(source_path)
            output_path = source_path

        _report("Compact complete", 1, 1)
        return output_path

    # ------------------------------------------------------------------
    # REPAIR — Fix corrupted archives
    # ------------------------------------------------------------------

    def repair_archive(self, output_path: str = None,
                       progress_callback=None) -> Dict[str, Any]:
        """
        Repair a damaged archive by:
        1. Validating every stored keyframe blob (decode test)
        2. Validating every residual blob (decode test)
        3. Fixing orphaned interpolated frames (missing parents)
        4. Fixing broken GOP chains
        5. Rebuilding index with corrections
        6. Optionally writing repaired archive to output_path

        Returns a report dict with repair details.
        """
        if not self._file_handle or not self._frames:
            raise RuntimeError("No archive is open")

        source_path = self._current_archive_path
        if output_path is None:
            output_path = source_path
            in_place = True
        else:
            in_place = False

        header = self._header
        frames = self._frames
        total = len(frames)

        def _report(msg, current, total_count):
            if progress_callback:
                progress_callback(msg, current, total_count)

        report = {
            "total_frames": total,
            "corrupted_keyframes": [],
            "corrupted_residuals": [],
            "orphaned_frames": [],
            "fixed_frames": [],
            "promoted_to_keyframe": [],
            "demoted_to_interpolated": [],
            "unrecoverable": [],
            "repaired": False,
        }

        # ----------------------------------------------------------------
        # Phase 1: Validate all stored blobs
        # ----------------------------------------------------------------
        _report("Repair Phase 1/4: Validating stored data...", 0, total)

        valid_keyframes = set()

        for i, f in enumerate(frames):
            if f.is_deleted:
                continue

            if f.frame_type in (FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME):
                if f.data_size > 0:
                    try:
                        self._file_handle.seek(
                            self._data_start + f.data_offset
                        )
                        data = self._file_handle.read(f.data_size)
                        img = webp_bytes_to_image(data)
                        if img is not None and img.size > 0:
                            valid_keyframes.add(i)
                        else:
                            report["corrupted_keyframes"].append(i)
                    except Exception:
                        report["corrupted_keyframes"].append(i)
                else:
                    report["corrupted_keyframes"].append(i)

            if f.frame_type == FrameType.RESIDUAL:
                if f.residual_size > 0:
                    try:
                        self._file_handle.seek(
                            self._data_start + f.residual_offset
                        )
                        data = self._file_handle.read(f.residual_size)
                        img = webp_bytes_to_image(data)
                        if img is None or img.size == 0:
                            report["corrupted_residuals"].append(i)
                    except Exception:
                        report["corrupted_residuals"].append(i)

            if i % 200 == 0 or i == total - 1:
                _report(
                    "Repair Phase 1/4: Validating data...",
                    i + 1, total
                )

        # ----------------------------------------------------------------
        # Phase 2: Check parent references
        # ----------------------------------------------------------------
        _report("Repair Phase 2/4: Checking parent refs...", 0, total)

        keyframe_set = set()
        for f in frames:
            if not f.is_deleted and f.frame_type in (
                FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME
            ) and f.index in valid_keyframes:
                keyframe_set.add(f.index)

        for i, f in enumerate(frames):
            if f.is_deleted:
                continue

            if f.frame_type in (FrameType.INTERPOLATED, FrameType.RESIDUAL):
                parent_a_ok = f.parent_keyframe_a in keyframe_set
                parent_b_ok = f.parent_keyframe_b in keyframe_set

                if not parent_a_ok or not parent_b_ok:
                    report["orphaned_frames"].append(i)

            if i % 500 == 0:
                _report(
                    "Repair Phase 2/4: Checking refs...",
                    i + 1, total
                )

        # ----------------------------------------------------------------
        # Phase 3: Apply fixes
        # ----------------------------------------------------------------
        _report("Repair Phase 3/4: Applying fixes...", 0, total)

        needs_rebuild = False

        # Fix corrupted keyframes — demote to interpolated if possible
        for i in report["corrupted_keyframes"]:
            f = frames[i]
            # Find nearest valid keyframes
            prev_kf = -1
            next_kf = -1
            for ki in sorted(keyframe_set):
                if ki < i:
                    prev_kf = ki
                elif ki > i and next_kf == -1:
                    next_kf = ki
                    break

            if prev_kf >= 0 and next_kf >= 0:
                f.frame_type = FrameType.INTERPOLATED
                f.parent_keyframe_a = prev_kf
                f.parent_keyframe_b = next_kf
                f.interpolation_timestep = \
                    (i - prev_kf) / (next_kf - prev_kf)
                f.data_offset = 0
                f.data_size = 0
                report["demoted_to_interpolated"].append(i)
                report["fixed_frames"].append(i)
                needs_rebuild = True
            else:
                report["unrecoverable"].append(i)

        # Fix corrupted residuals — demote to pure interpolated
        for i in report["corrupted_residuals"]:
            f = frames[i]
            f.frame_type = FrameType.INTERPOLATED
            f.residual_offset = 0
            f.residual_size = 0
            report["fixed_frames"].append(i)
            needs_rebuild = True

        # Fix orphaned frames — try to find new parents or promote
        for i in report["orphaned_frames"]:
            f = frames[i]
            prev_kf = -1
            next_kf = -1
            for ki in sorted(keyframe_set):
                if ki < i:
                    prev_kf = ki
                elif ki > i and next_kf == -1:
                    next_kf = ki
                    break

            if prev_kf >= 0 and next_kf >= 0:
                f.parent_keyframe_a = prev_kf
                f.parent_keyframe_b = next_kf
                f.interpolation_timestep = \
                    (i - prev_kf) / (next_kf - prev_kf)
                f.frame_type = FrameType.INTERPOLATED
                f.residual_offset = 0
                f.residual_size = 0
                report["fixed_frames"].append(i)
                needs_rebuild = True
            elif prev_kf >= 0:
                f.parent_keyframe_a = prev_kf
                f.parent_keyframe_b = prev_kf
                f.interpolation_timestep = 0.0
                f.frame_type = FrameType.INTERPOLATED
                report["fixed_frames"].append(i)
                needs_rebuild = True
            else:
                report["unrecoverable"].append(i)

        # ----------------------------------------------------------------
        # Phase 4: Rebuild if fixes were applied
        # ----------------------------------------------------------------
        if needs_rebuild:
            _report("Repair Phase 4/4: Rebuilding archive...", 0, 1)

            # Save updated index
            if in_place:
                self.save_index_changes()
                # Then compact to clean up dead data
                self.compact_archive(
                    output_path=None,
                    progress_callback=lambda m, c, t: _report(
                        f"Repair rebuild: {m}", c, t
                    )
                )
            else:
                # Write repaired copy
                self._frames = frames
                self.save_index_changes()
                self.compact_archive(
                    output_path=output_path,
                    progress_callback=lambda m, c, t: _report(
                        f"Repair rebuild: {m}", c, t
                    )
                )

            report["repaired"] = True
        else:
            report["repaired"] = False

        _report("Repair complete", 1, 1)

        # Deduplicate unrecoverable list
        report["unrecoverable"] = list(set(report["unrecoverable"]))
        report["fixed_frames"] = list(set(report["fixed_frames"]))

        return report

# ============================================================================
# FAST ANALYSIS ENGINE — High-speed analyzer for 45k-100k+ sequences
# ============================================================================

class FastAnalyzer:
    """
    High-speed sequence analyzer for 45k-100k+ image sequences.

    Strategy (4 phases):
      Phase 1: Ultra-fast histogram scan on ALL frames (~1ms/frame)
      Phase 2: Thumbnail SSIM only on uncertain frames (~5ms/frame)
      Phase 3: AI analysis ONLY on boundary/risk frames (~100-200ms/frame)
      Phase 4: Build compression plan from collected scores

    For a typical 45k similar sequence:
      Phase 1: ~45 seconds (all frames)
      Phase 2: ~2 minutes (5-10% of frames)
      Phase 3: ~1-2 minutes (1-5% of frames)
      Phase 4: ~10 seconds
      Total: ~4-6 minutes instead of 3+ hours
    """

    THUMB_SIZE = 256
    HIST_BINS = 16

    # Histogram correlation thresholds
    HIST_CERTAIN_SAME = 0.985
    HIST_PROBABLY_SAME = 0.95
    HIST_MAYBE_DIFFERENT = 0.85
    HIST_SCENE_CUT = 0.60

    def __init__(self, model_manager: ModelManager, settings: AppSettings):
        self.models = model_manager
        self.settings = settings
        self._thumb_cache = {}

    def clear_cache(self):
        self._thumb_cache.clear()

    # ------------------------------------------------------------------
    # Fast primitives
    # ------------------------------------------------------------------

    @staticmethod
    def _fast_histogram(img: np.ndarray) -> np.ndarray:
        """Compact color histogram. ~0.2ms."""
        hist = cv2.calcHist(
            [img], [0, 1, 2], None,
            [FastAnalyzer.HIST_BINS, FastAnalyzer.HIST_BINS, FastAnalyzer.HIST_BINS],
            [0, 256, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist

    @staticmethod
    def _hist_correlation(h1: np.ndarray, h2: np.ndarray) -> float:
        """Compare two histograms. 0.0=different, 1.0=identical. ~0.01ms."""
        return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))

    @staticmethod
    def _load_thumbnail(path: str, size: int = 256) -> np.ndarray:
        """Load image as small thumbnail. ~2ms (disk IO bound)."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read: {path}")
        h, w = img.shape[:2]
        scale = size / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def _thumbnail_ssim(thumb1: np.ndarray, thumb2: np.ndarray) -> float:
        """Fast SSIM on small thumbnails using OpenCV CPU. ~2ms."""
        if thumb1.shape != thumb2.shape:
            thumb2 = cv2.resize(thumb2, (thumb1.shape[1], thumb1.shape[0]))

        gray1 = cv2.cvtColor(thumb1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray2 = cv2.cvtColor(thumb2, cv2.COLOR_BGR2GRAY).astype(np.float64)

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu12 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(gray1 * gray1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2 * gray2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(np.mean(ssim_map))

    @staticmethod
    def _pixel_diff_score(thumb1: np.ndarray, thumb2: np.ndarray) -> float:
        """Ultra-fast pixel difference. ~0.1ms on thumbnails."""
        if thumb1.shape != thumb2.shape:
            thumb2 = cv2.resize(thumb2, (thumb1.shape[1], thumb1.shape[0]))
        diff = cv2.absdiff(thumb1, thumb2)
        return float(np.mean(diff)) / 255.0

    # ------------------------------------------------------------------
    # Main analysis pipeline
    # ------------------------------------------------------------------

    def analyze_sequence(
        self,
        image_paths: List[str],
        progress_callback=None
    ) -> Tuple[ArchiveHeader, List[FrameEntry], List[GOPEntry]]:
        """
        Run full 4-phase analysis on an image sequence.
        Returns (header, frames, gops) or None if cancelled.
        """
        total = len(image_paths)
        if total == 0:
            raise ValueError("No images to analyze")

        settings = self.settings
        gop_size = settings.default_gop_size
        sim_thresh = settings.similarity_threshold
        face_thresh = settings.face_similarity_threshold
        body_thresh = settings.body_similarity_threshold
        id_thresh = settings.identity_mismatch_threshold

        # Result arrays (numpy for speed)
        similarity_scores = np.ones(total, dtype=np.float32)
        scene_cuts = np.zeros(total, dtype=bool)
        face_scores = np.ones(total, dtype=np.float32)
        body_scores = np.ones(total, dtype=np.float32)
        identity_scores = np.ones(total, dtype=np.float32)
        motion_scores = np.zeros(total, dtype=np.float32)
        frame_types = np.full(total, FrameType.INTERPOLATED, dtype=np.int32)

        # Get dimensions from first image
        first_img = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
        if first_img is None:
            raise RuntimeError(f"Cannot read first image: {image_paths[0]}")
        orig_h, orig_w = first_img.shape[:2]
        del first_img

        original_total_bytes = 0
        needs_ssim = set()
        needs_ai = set()

        def _report(msg, current, total_count, detail=""):
            if progress_callback:
                progress_callback.update_progress(msg, current, total_count, detail)

        def _cancelled():
            return progress_callback and progress_callback.is_cancelled()

        # ==============================================================
        # PHASE 1: Ultra-fast histogram scan — ALL frames (~1ms/frame)
        # ==============================================================
        _report("Phase 1/4: Fast histogram scan...", 0, total)

        prev_hist = None
        prev_thumb = None

        for i in range(total):
            if _cancelled():
                return None

            try:
                thumb = self._load_thumbnail(image_paths[i], self.THUMB_SIZE)
                original_total_bytes += os.path.getsize(image_paths[i])
            except Exception:
                frame_types[i] = FrameType.FORCED_KEYFRAME
                similarity_scores[i] = 0.0
                if i % 500 == 0:
                    _report("Phase 1/4: Fast histogram scan...", i + 1, total,
                            f"Frame {i}: CORRUPT — skipped")
                continue

            hist = self._fast_histogram(thumb)
            self._thumb_cache[i] = thumb

            if i == 0:
                frame_types[i] = FrameType.KEYFRAME
                similarity_scores[i] = 1.0
            elif prev_hist is not None:
                corr = self._hist_correlation(prev_hist, hist)

                if corr >= self.HIST_CERTAIN_SAME:
                    pdiff = self._pixel_diff_score(prev_thumb, thumb)
                    similarity_scores[i] = 1.0 - pdiff

                elif corr >= self.HIST_PROBABLY_SAME:
                    similarity_scores[i] = corr
                    needs_ssim.add(i)

                elif corr >= self.HIST_MAYBE_DIFFERENT:
                    similarity_scores[i] = corr
                    needs_ssim.add(i)
                    needs_ai.add(i)

                elif corr >= self.HIST_SCENE_CUT:
                    similarity_scores[i] = corr
                    scene_cuts[i] = True
                    needs_ai.add(i)

                else:
                    similarity_scores[i] = corr
                    scene_cuts[i] = True
                    frame_types[i] = FrameType.FORCED_KEYFRAME

            prev_hist = hist
            prev_thumb = thumb

            if i % 500 == 0 or i == total - 1:
                _report("Phase 1/4: Fast histogram scan...", i + 1, total,
                        f"Frame {i}: corr={similarity_scores[i]:.4f} | "
                        f"Uncertain: {len(needs_ssim)} | AI queue: {len(needs_ai)}")

        _report(f"Phase 1 done: {len(needs_ssim):,} need SSIM, "
                f"{len(needs_ai):,} need AI", total, total)

        # ==============================================================
        # PHASE 2: Thumbnail SSIM — only uncertain frames (~5ms/frame)
        # ==============================================================
        ssim_list = sorted(needs_ssim)
        ssim_total = len(ssim_list)

        if ssim_total > 0:
            _report("Phase 2/4: Thumbnail SSIM on uncertain frames...", 0, ssim_total)

            for pos, i in enumerate(ssim_list):
                if _cancelled():
                    return None

                prev_i = i - 1
                if prev_i < 0:
                    continue

                thumb_cur = self._thumb_cache.get(i)
                thumb_prev = self._thumb_cache.get(prev_i)

                if thumb_cur is not None and thumb_prev is not None:
                    ssim_val = self._thumbnail_ssim(thumb_prev, thumb_cur)
                    similarity_scores[i] = ssim_val

                    if ssim_val < sim_thresh + 0.05:
                        needs_ai.add(i)
                    if ssim_val < sim_thresh - 0.1:
                        scene_cuts[i] = True

                if pos % 200 == 0 or pos == ssim_total - 1:
                    _report("Phase 2/4: Thumbnail SSIM...", pos + 1, ssim_total,
                            f"Frame {i}: SSIM={similarity_scores[i]:.4f} | "
                            f"AI queue: {len(needs_ai)}")


        # ==============================================================
        # PHASE 3: AI analysis — ONLY risk frames
        # GPU-enforced, mid-resolution, with CPU throttle prevention
        # ==============================================================
        ai_list = sorted(needs_ai)
        ai_total = len(ai_list)

        # Mid-resolution for AI analysis — 512px is enough for face/body
        # detection while being 4x fewer pixels than 1024px originals
        AI_ANALYSIS_SIZE = 512

        if ai_total > 0:
            _report("Phase 3/4: AI analysis on risk frames...", 0, ai_total)

            # Pre-load mid-res images in small batches to control memory
            # and prevent CPU thrashing from sequential cv2.imread
            AI_BATCH_SIZE = 20
            ai_batches = [ai_list[b:b + AI_BATCH_SIZE]
                          for b in range(0, ai_total, AI_BATCH_SIZE)]

            processed = 0

            for batch in ai_batches:
                if _cancelled():
                    return None

                # Pre-load this batch of images at mid-resolution
                batch_images = {}
                for i in batch:
                    prev_i = i - 1
                    for idx in (prev_i, i):
                        if idx >= 0 and idx not in batch_images:
                            try:
                                img = cv2.imread(image_paths[idx], cv2.IMREAD_COLOR)
                                if img is not None:
                                    h, w = img.shape[:2]
                                    scale = AI_ANALYSIS_SIZE / max(h, w)
                                    if scale < 1.0:
                                        img = cv2.resize(
                                            img,
                                            (int(w * scale), int(h * scale)),
                                            interpolation=cv2.INTER_AREA
                                        )
                                    batch_images[idx] = img
                            except Exception:
                                pass

                # Process each frame in this batch
                for i in batch:
                    if _cancelled():
                        return None

                    prev_i = i - 1
                    if prev_i < 0:
                        processed += 1
                        continue

                    img_cur = batch_images.get(i)
                    img_prev = batch_images.get(prev_i)

                    if img_cur is None or img_prev is None:
                        face_scores[i] = 0.5
                        body_scores[i] = 0.5
                        processed += 1
                        continue

                    # Face analysis (GPU via InsightFace CUDA provider)
                    if settings.face_safe_mode:
                        try:
                            face_scores[i] = self.models.compute_face_similarity(
                                img_prev, img_cur
                            )
                        except Exception:
                            face_scores[i] = 1.0

                    # Body analysis (GPU via ONNX CUDA provider)
                    if settings.body_safe_mode:
                        try:
                            body_scores[i] = self.models.compute_body_score(
                                img_prev, img_cur
                            )
                        except Exception:
                            body_scores[i] = 1.0

                    # Identity consistency (GPU via InsightFace)
                    if settings.use_identity_check:
                        try:
                            identity_scores[i] = \
                                self.models.check_identity_consistency(
                                    img_prev, img_cur
                                )
                        except Exception:
                            identity_scores[i] = 1.0

                    # Motion scoring on thumbnails (already cached, fast)
                    if settings.use_optical_flow:
                        try:
                            t_cur = self._thumb_cache.get(i)
                            t_prev = self._thumb_cache.get(prev_i)
                            if t_cur is not None and t_prev is not None:
                                motion_scores[i] = \
                                    self.models.compute_motion_score(
                                        t_prev, t_cur
                                    )
                        except Exception:
                            motion_scores[i] = 0.0

                    processed += 1

                    if processed % 10 == 0 or processed == ai_total:
                        _report(
                            "Phase 3/4: AI analysis...",
                            processed, ai_total,
                            f"Frame {i}: face={face_scores[i]:.3f} "
                            f"body={body_scores[i]:.3f} "
                            f"id={identity_scores[i]:.3f}"
                        )

                # Free batch memory explicitly
                del batch_images

                # Yield to OS / prevent CPU starvation
                # This is the key throttle-prevention mechanism
                time.sleep(0.01)

        # ==============================================================
        # PHASE 4: Build compression plan
        # ==============================================================
        _report("Phase 4/4: Building compression plan...", 0, total)

        frames: List[FrameEntry] = []
        gops: List[GOPEntry] = []
        gop_id = 0
        gop_start = 0
        gop_keyframes = []
        frames_since_keyframe = 0

        for i in range(total):
            if _cancelled():
                return None

            is_first = (i == 0)
            is_scene_cut = bool(scene_cuts[i])
            is_gop_boundary = (frames_since_keyframe >= gop_size)

            # Already classified as forced keyframe in phase 1 (corrupt / drastic cut)
            if frame_types[i] == FrameType.FORCED_KEYFRAME:
                force_keyframe = True
                force_reason = "phase1_forced"
            elif is_first:
                force_keyframe = True
                force_reason = "first_frame"
            elif is_scene_cut:
                force_keyframe = True
                force_reason = "scene_cut"
            elif is_gop_boundary:
                force_keyframe = True
                force_reason = "gop_boundary"
            elif similarity_scores[i] < sim_thresh:
                force_keyframe = True
                force_reason = "low_similarity"
            elif settings.face_safe_mode and face_scores[i] < face_thresh:
                force_keyframe = True
                force_reason = "face_risk"
            elif settings.body_safe_mode and body_scores[i] < body_thresh:
                force_keyframe = True
                force_reason = "body_risk"
            elif settings.use_identity_check and identity_scores[i] < id_thresh:
                force_keyframe = True
                force_reason = "identity_mismatch"
            else:
                force_keyframe = False
                force_reason = ""

            if force_keyframe:
                if force_reason in ("scene_cut", "face_risk", "body_risk",
                                    "identity_mismatch", "phase1_forced"):
                    ftype = FrameType.FORCED_KEYFRAME
                else:
                    ftype = FrameType.KEYFRAME

                if is_gop_boundary or is_scene_cut or is_first or \
                   force_reason == "phase1_forced":
                    if i > 0 and gop_keyframes:
                        gops.append(GOPEntry(
                            gop_id=gop_id,
                            start_frame=gop_start,
                            end_frame=i - 1,
                            keyframe_indices=gop_keyframes[:],
                            frame_count=i - gop_start,
                            has_scene_cut=bool(np.any(scene_cuts[gop_start:i]))
                        ))
                    gop_id += 1
                    gop_start = i
                    gop_keyframes = []

                gop_keyframes.append(i)
                frames_since_keyframe = 0
            else:
                need_residual = False
                if settings.keep_residuals:
                    if similarity_scores[i] < (sim_thresh + 0.03):
                        need_residual = True
                    if i in needs_ai:
                        if face_scores[i] < (face_thresh + 0.02):
                            need_residual = True
                        if body_scores[i] < (body_thresh + 0.03):
                            need_residual = True

                ftype = FrameType.RESIDUAL if need_residual else FrameType.INTERPOLATED
                frames_since_keyframe += 1

            frame_types[i] = ftype

            frame = FrameEntry(
                index=i,
                name=os.path.basename(image_paths[i]),
                frame_type=int(ftype),
                width=orig_w,
                height=orig_h,
                gop_id=gop_id,
                face_score=float(face_scores[i]),
                body_score=float(body_scores[i]),
                similarity_score=float(similarity_scores[i]),
                motion_score=float(motion_scores[i]),
                scene_cut=is_scene_cut,
            )
            frames.append(frame)

            if i % 2000 == 0 or i == total - 1:
                _report("Phase 4/4: Building plan...", i + 1, total,
                        f"Frame {i}: {FRAME_TYPE_LABELS.get(ftype, '?')} | "
                        f"sim={similarity_scores[i]:.3f}")

        # Close last GOP
        if gop_start < total and gop_keyframes:
            gops.append(GOPEntry(
                gop_id=gop_id,
                start_frame=gop_start,
                end_frame=total - 1,
                keyframe_indices=gop_keyframes[:],
                frame_count=total - gop_start,
                has_scene_cut=bool(np.any(scene_cuts[gop_start:total]))
            ))

        # ==============================================================
        # Assign parent keyframes for interpolated/residual frames
        # Uses numpy for O(n) instead of O(n*k) nested loop
        # ==============================================================
        _report("Assigning interpolation parents...", 0, 1)

        kf_set = set()
        for f in frames:
            if f.frame_type in (FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME):
                kf_set.add(f.index)

        prev_kf_map = np.full(total, -1, dtype=np.int64)
        next_kf_map = np.full(total, -1, dtype=np.int64)

        last_kf = -1
        for i in range(total):
            if i in kf_set:
                last_kf = i
            prev_kf_map[i] = last_kf

        last_kf = -1
        for i in range(total - 1, -1, -1):
            if i in kf_set:
                last_kf = i
            next_kf_map[i] = last_kf

        for f in frames:
            if f.frame_type in (FrameType.INTERPOLATED, FrameType.RESIDUAL):
                prev_kf = int(prev_kf_map[f.index])
                next_kf = int(next_kf_map[f.index])

                if prev_kf == -1:
                    prev_kf = f.index
                if next_kf == -1:
                    next_kf = f.index

                f.parent_keyframe_a = prev_kf
                f.parent_keyframe_b = next_kf

                if next_kf != prev_kf:
                    f.interpolation_timestep = \
                        (f.index - prev_kf) / (next_kf - prev_kf)
                else:
                    f.interpolation_timestep = 0.0
                    f.frame_type = FrameType.KEYFRAME
                    frame_types[f.index] = FrameType.KEYFRAME

        # ==============================================================
        # Build header
        # ==============================================================
        kf_count = int(np.sum(frame_types == FrameType.KEYFRAME))
        fkf_count = int(np.sum(frame_types == FrameType.FORCED_KEYFRAME))
        interp_count = int(np.sum(frame_types == FrameType.INTERPOLATED))
        res_count = int(np.sum(frame_types == FrameType.RESIDUAL))

        header = ArchiveHeader(
            total_frames=total,
            keyframe_count=kf_count,
            interpolated_count=interp_count,
            residual_count=res_count,
            forced_keyframe_count=fkf_count,
            gop_count=len(gops),
            gop_size=gop_size,
            original_width=orig_w,
            original_height=orig_h,
            compression_codec="webp",
            compression_quality=settings.default_quality,
            archive_downscale=settings.use_archive_downscale,
            downscale_factor=settings.downscale_factor
            if settings.use_archive_downscale else 1.0,
            use_residuals=settings.keep_residuals,
            residual_strength=settings.residual_strength,
            face_safe=settings.face_safe_mode,
            body_safe=settings.body_safe_mode,
            identity_check=settings.use_identity_check,
            depth_aware=settings.use_depth_aware,
            similarity_threshold=sim_thresh,
            face_threshold=face_thresh,
            body_threshold=body_thresh,
            rife_model=settings.rife_model_dir,
            raft_model=settings.raft_weights,
            original_total_bytes=int(original_total_bytes),
        )

        self.clear_cache()

        _report(
            f"Done: {kf_count + fkf_count:,} keyframes, "
            f"{interp_count:,} interpolated, {res_count:,} residual",
            total, total
        )

        return header, frames, gops

# ============================================================================
# GALLERY ANALYZER — For non-sequential mixed-face image collections
# ============================================================================

class GalleryAnalyzer:
    """
    Analyzer for gallery-mode archives (mixed images, not video frames).
    Groups images by face identity, sorts within groups by similarity,
    then applies GOP compression within each group.
    """

    def __init__(self, model_manager: ModelManager, settings: AppSettings):
        self.models = model_manager
        self.settings = settings

    def analyze_sequence(
        self,
        image_paths: List[str],
        progress_callback=None
    ) -> Tuple[ArchiveHeader, List[FrameEntry], List[GOPEntry]]:

        total = len(image_paths)
        if total == 0:
            raise ValueError("No images to analyze")

        settings = self.settings
        gop_size = settings.default_gop_size
        sim_thresh = settings.similarity_threshold

        def _report(msg, current, total_count, detail=""):
            if progress_callback:
                progress_callback.update_progress(
                    msg, current, total_count, detail
                )

        def _cancelled():
            return progress_callback and progress_callback.is_cancelled()

        # Get dimensions
        first_img = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
        if first_img is None:
            raise RuntimeError(f"Cannot read: {image_paths[0]}")
        orig_h, orig_w = first_img.shape[:2]
        del first_img

        original_total_bytes = 0

        # ==============================================================
        # PHASE 1: Extract face embeddings for all images
        # ==============================================================
        _report("Gallery Phase 1/4: Extracting face embeddings...", 0, total)

        embeddings = {}
        no_face = []

        for i, path in enumerate(image_paths):
            if _cancelled():
                return None

            original_total_bytes += os.path.getsize(path)

            try:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    no_face.append(i)
                    continue

                # Resize for faster face detection
                h, w = img.shape[:2]
                scale = 512 / max(h, w)
                if scale < 1.0:
                    img = cv2.resize(
                        img, (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA
                    )

                emb = self.models.compute_identity_embedding(img)
                if emb is not None:
                    embeddings[i] = emb
                else:
                    no_face.append(i)

                del img

            except Exception:
                no_face.append(i)

            if i % 50 == 0 or i == total - 1:
                _report(
                    "Gallery Phase 1/4: Face embeddings...",
                    i + 1, total,
                    f"Faces found: {len(embeddings)}, "
                    f"No face: {len(no_face)}"
                )

        # ==============================================================
        # PHASE 2: Cluster by identity
        # ==============================================================
        _report("Gallery Phase 2/4: Clustering by identity...", 0, 1)

        clusters: Dict[int, List[int]] = {}
        assigned = set()
        cluster_id = 0
        emb_indices = list(embeddings.keys())

        id_thresh = 0.5  # Cosine similarity threshold for same person

        for i in emb_indices:
            if _cancelled():
                return None
            if i in assigned:
                continue

            cluster = [i]
            assigned.add(i)

            for j in emb_indices:
                if j in assigned:
                    continue

                cos_sim = float(np.dot(
                    embeddings[i], embeddings[j]
                ) / (
                    np.linalg.norm(embeddings[i]) *
                    np.linalg.norm(embeddings[j]) + 1e-8
                ))

                if cos_sim > id_thresh:
                    cluster.append(j)
                    assigned.add(j)

            clusters[cluster_id] = cluster
            cluster_id += 1

        _report(
            f"Gallery Phase 2/4: Found {len(clusters)} people, "
            f"{len(no_face)} images without faces",
            1, 1
        )

        # ==============================================================
        # PHASE 3: Sort within each cluster by visual similarity
        # ==============================================================
        _report("Gallery Phase 3/4: Sorting within clusters...", 0,
                len(clusters))

        THUMB = 192

        def _load_thumb(idx):
            img = cv2.imread(image_paths[idx], cv2.IMREAD_COLOR)
            if img is None:
                return np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
            h, w = img.shape[:2]
            s = THUMB / max(h, w)
            if s < 1.0:
                img = cv2.resize(
                    img, (int(w * s), int(h * s)),
                    interpolation=cv2.INTER_AREA
                )
            return img

        sorted_order = []

        for cid, indices in enumerate(clusters.items()):
            if _cancelled():
                return None

            cluster_key, cluster_indices = indices

            if len(cluster_indices) <= 2:
                sorted_order.extend(cluster_indices)
                continue

            # Greedy nearest-neighbor sort within cluster
            thumbs = {
                idx: _load_thumb(idx)
                for idx in cluster_indices
            }

            remaining = set(cluster_indices)
            current = cluster_indices[0]
            chain = [current]
            remaining.remove(current)

            while remaining:
                best_idx = None
                best_score = -1

                current_thumb = thumbs[current]

                for candidate in remaining:
                    cand_thumb = thumbs[candidate]
                    if current_thumb.shape != cand_thumb.shape:
                        cand_thumb = cv2.resize(
                            cand_thumb,
                            (current_thumb.shape[1],
                             current_thumb.shape[0])
                        )
                    diff = cv2.absdiff(current_thumb, cand_thumb)
                    score = 1.0 - (float(np.mean(diff)) / 255.0)

                    if score > best_score:
                        best_score = score
                        best_idx = candidate

                if best_idx is not None:
                    chain.append(best_idx)
                    remaining.remove(best_idx)
                    current = best_idx
                else:
                    break

            sorted_order.extend(chain)
            del thumbs

            if cid % 5 == 0:
                _report(
                    "Gallery Phase 3/4: Sorting clusters...",
                    cid + 1, len(clusters),
                    f"Cluster {cid}: {len(cluster_indices)} images"
                )

        # Append no-face images at the end
        sorted_order.extend(no_face)

        # ==============================================================
        # PHASE 4: Build compression plan on sorted order
        # ==============================================================
        _report("Gallery Phase 4/4: Building compression plan...", 0, total)

        frames: List[FrameEntry] = []
        gops: List[GOPEntry] = []
        gop_id = 0
        gop_start = 0
        gop_keyframes = []
        frames_since_keyframe = 0

        # Build cluster boundary set
        cluster_boundaries = set()
        pos = 0
        for _, cluster_indices in clusters.items():
            pos += len(cluster_indices)
            cluster_boundaries.add(pos)

        # Map sorted position back to original index
        sorted_paths = [image_paths[idx] for idx in sorted_order]

        prev_thumb = None

        for pos_i in range(len(sorted_order)):
            if _cancelled():
                return None

            orig_idx = sorted_order[pos_i]
            is_first = (pos_i == 0)
            is_cluster_boundary = (pos_i in cluster_boundaries)
            is_gop_boundary = (frames_since_keyframe >= gop_size)
            is_no_face = (orig_idx in no_face)

            # Load thumbnail for similarity check
            curr_thumb = _load_thumb(orig_idx)
            similarity = 1.0

            if prev_thumb is not None and not is_cluster_boundary:
                ct = curr_thumb
                pt = prev_thumb
                if ct.shape != pt.shape:
                    ct = cv2.resize(
                        ct, (pt.shape[1], pt.shape[0])
                    )
                diff = cv2.absdiff(pt, ct)
                similarity = 1.0 - (float(np.mean(diff)) / 255.0)

            # force_keyframe = False
            #
            # if is_first or is_cluster_boundary or is_no_face:
            #     force_keyframe = True
            # elif is_gop_boundary:
            #     force_keyframe = True
            # elif similarity < sim_thresh:
            #     force_keyframe = True
            force_keyframe = False

            if is_first or is_cluster_boundary or is_no_face:
                force_keyframe = True
            elif is_gop_boundary:
                force_keyframe = True
            elif similarity < 0.97:
                # Gallery mode needs MUCH higher threshold than sequence
                # Only interpolate between very similar images
                # Otherwise RIFE produces ghostly blends
                force_keyframe = True

            if force_keyframe:
                if is_cluster_boundary:
                    ftype = FrameType.FORCED_KEYFRAME
                else:
                    ftype = FrameType.KEYFRAME

                if is_gop_boundary or is_cluster_boundary or is_first:
                    if pos_i > 0 and gop_keyframes:
                        gops.append(GOPEntry(
                            gop_id=gop_id,
                            start_frame=gop_start,
                            end_frame=pos_i - 1,
                            keyframe_indices=gop_keyframes[:],
                            frame_count=pos_i - gop_start
                        ))
                    gop_id += 1
                    gop_start = pos_i
                    gop_keyframes = []

                gop_keyframes.append(pos_i)
                frames_since_keyframe = 0
            else:
                need_residual = settings.keep_residuals and \
                    similarity < (sim_thresh + 0.03)
                ftype = FrameType.RESIDUAL if need_residual \
                    else FrameType.INTERPOLATED
                frames_since_keyframe += 1

            frame = FrameEntry(
                index=pos_i,
                name=os.path.basename(image_paths[orig_idx]),
                frame_type=int(ftype),
                width=orig_w,
                height=orig_h,
                gop_id=gop_id,
                similarity_score=float(similarity),
                face_score=1.0 if orig_idx not in no_face else 0.0,
                body_score=1.0,
                motion_score=0.0,
                scene_cut=is_cluster_boundary,
            )
            frames.append(frame)

            prev_thumb = curr_thumb

            if pos_i % 100 == 0 or pos_i == total - 1:
                _report(
                    "Gallery Phase 4/4: Building plan...",
                    pos_i + 1, total,
                    f"Frame {pos_i}: "
                    f"{FRAME_TYPE_LABELS.get(ftype, '?')} "
                    f"sim={similarity:.3f}"
                )

        # Close last GOP
        if gop_start < len(sorted_order) and gop_keyframes:
            gops.append(GOPEntry(
                gop_id=gop_id,
                start_frame=gop_start,
                end_frame=len(sorted_order) - 1,
                keyframe_indices=gop_keyframes[:],
                frame_count=len(sorted_order) - gop_start
            ))

        # Assign parents (same logic as FastAnalyzer)
        kf_set = set()
        for f in frames:
            if f.frame_type in (FrameType.KEYFRAME,
                                FrameType.FORCED_KEYFRAME):
                kf_set.add(f.index)

        prev_kf_map = np.full(len(frames), -1, dtype=np.int64)
        next_kf_map = np.full(len(frames), -1, dtype=np.int64)

        last_kf = -1
        for i in range(len(frames)):
            if i in kf_set:
                last_kf = i
            prev_kf_map[i] = last_kf

        last_kf = -1
        for i in range(len(frames) - 1, -1, -1):
            if i in kf_set:
                last_kf = i
            next_kf_map[i] = last_kf

        for f in frames:
            if f.frame_type in (FrameType.INTERPOLATED,
                                FrameType.RESIDUAL):
                prev_kf = int(prev_kf_map[f.index])
                next_kf = int(next_kf_map[f.index])
                if prev_kf == -1:
                    prev_kf = f.index
                if next_kf == -1:
                    next_kf = f.index
                f.parent_keyframe_a = prev_kf
                f.parent_keyframe_b = next_kf
                if next_kf != prev_kf:
                    f.interpolation_timestep = \
                        (f.index - prev_kf) / (next_kf - prev_kf)
                else:
                    f.frame_type = FrameType.KEYFRAME

        # Build header
        kf_count = sum(1 for f in frames
                       if f.frame_type == FrameType.KEYFRAME)
        fkf_count = sum(1 for f in frames
                        if f.frame_type == FrameType.FORCED_KEYFRAME)
        interp_count = sum(1 for f in frames
                           if f.frame_type == FrameType.INTERPOLATED)
        res_count = sum(1 for f in frames
                        if f.frame_type == FrameType.RESIDUAL)

        header = ArchiveHeader(
            total_frames=len(frames),
            keyframe_count=kf_count,
            interpolated_count=interp_count,
            residual_count=res_count,
            forced_keyframe_count=fkf_count,
            gop_count=len(gops),
            gop_size=gop_size,
            original_width=orig_w,
            original_height=orig_h,
            compression_codec="webp",
            compression_quality=settings.default_quality,
            archive_downscale=settings.use_archive_downscale,
            downscale_factor=settings.downscale_factor
                if settings.use_archive_downscale else 1.0,
            use_residuals=settings.keep_residuals,
            residual_strength=settings.residual_strength,
            face_safe=settings.face_safe_mode,
            body_safe=settings.body_safe_mode,
            similarity_threshold=sim_thresh,
            original_total_bytes=int(original_total_bytes),
        )

        # Store sorted order mapping for build phase
        self._sorted_order = sorted_order
        self._sorted_paths = sorted_paths

        _report(
            f"Gallery done: {kf_count + fkf_count} KFs, "
            f"{interp_count} interp, {res_count} residual, "
            f"{len(clusters)} people",
            total, total
        )

        return header, frames, gops

# ============================================================================
# GUI FRAMEWORK — Tkinter Application Base
# ============================================================================

import tkinter.font as tkfont


# ============================================================================
# TOOLTIP SYSTEM — Single global manager, no duplicates possible
# ============================================================================

class _ToolTipManager:
    """
    Global singleton tooltip manager.
    Only ONE tooltip can ever exist at any time.
    All tooltip show/hide goes through this single manager.
    """

    def __init__(self):
        self._tip_window = None
        self._scheduled_id = None
        self._owner_widget = None
        self._delay = 500

    def schedule(self, widget, text: str):
        """Schedule a tooltip to appear for the given widget."""
        # If same widget already scheduled, skip
        if self._owner_widget is widget and self._tip_window:
            return

        # Kill any existing tooltip or pending schedule
        self.cancel()
        self.hide()

        self._owner_widget = widget

        try:
            if not widget.winfo_exists():
                return
            self._scheduled_id = widget.after(
                self._delay,
                lambda: self._show(widget, text)
            )
        except Exception:
            pass

    def cancel(self):
        """Cancel any pending tooltip."""
        if self._scheduled_id and self._owner_widget:
            try:
                self._owner_widget.after_cancel(self._scheduled_id)
            except Exception:
                pass
        self._scheduled_id = None

    def hide(self):
        """Immediately hide the current tooltip."""
        self.cancel()
        if self._tip_window:
            try:
                self._tip_window.destroy()
            except Exception:
                pass
            self._tip_window = None
        self._owner_widget = None

    def _show(self, widget, text: str):
        """Actually show the tooltip."""
        self._scheduled_id = None

        # Verify widget is still alive and visible
        try:
            if not widget.winfo_exists():
                return
            if not widget.winfo_viewable():
                return
            # Check mouse is still over this widget
            mx = widget.winfo_pointerx()
            my = widget.winfo_pointery()
            wx = widget.winfo_rootx()
            wy = widget.winfo_rooty()
            ww = widget.winfo_width()
            wh = widget.winfo_height()
            if not (wx <= mx <= wx + ww and wy <= my <= wy + wh):
                return
        except Exception:
            return

        # Kill any leftover
        if self._tip_window:
            try:
                self._tip_window.destroy()
            except Exception:
                pass
            self._tip_window = None

        try:
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + widget.winfo_height() + 5
        except Exception:
            return

        tw = tk.Toplevel(widget)
        tw.wm_overrideredirect(True)
        tw.wm_attributes("-topmost", True)

        frame = tk.Frame(tw, bg="#FFFDE7", relief="solid", bd=1)
        frame.pack()

        label = tk.Label(
            frame,
            text=text,
            justify="left",
            bg="#FFFDE7",
            fg="#333333",
            font=("Segoe UI", 9),
            padx=8,
            pady=4,
            wraplength=350
        )
        label.pack()

        # Position — keep on screen
        tw.update_idletasks()
        try:
            screen_w = widget.winfo_screenwidth()
            screen_h = widget.winfo_screenheight()
        except Exception:
            screen_w, screen_h = 1920, 1080

        tip_w = tw.winfo_width()
        tip_h = tw.winfo_height()

        if x + tip_w > screen_w:
            x = screen_w - tip_w - 10
        if y + tip_h > screen_h:
            y = widget.winfo_rooty() - tip_h - 5
        if x < 0:
            x = 5
        if y < 0:
            y = 5

        tw.wm_geometry(f"+{x}+{y}")

        self._tip_window = tw
        self._owner_widget = widget

        # Auto-hide after 3 seconds
        try:
            widget.after(8000, self.hide)
        except Exception:
            pass


# Single global instance — created once at module level
_tooltip_mgr = _ToolTipManager()


class ToolTip:
    """
    Tooltip binding for a widget.
    Delegates everything to the single global _ToolTipManager.
    No per-instance windows. No possibility of duplicates.
    """

    def __init__(self, widget, text: str, delay: int = 500):
        self.widget = widget
        self.text = text
        _tooltip_mgr._delay = delay

        # Use add="+" so we don't clobber other bindings
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")
        widget.bind("<MouseWheel>", self._on_leave, add="+")
        widget.bind("<Destroy>", self._on_destroy, add="+")

    def _on_enter(self, event=None):
        _tooltip_mgr.schedule(self.widget, self.text)

    def _on_leave(self, event=None):
        # Only hide if we are the current owner
        if _tooltip_mgr._owner_widget is self.widget:
            _tooltip_mgr.hide()

    def _on_destroy(self, event=None):
        if _tooltip_mgr._owner_widget is self.widget:
            _tooltip_mgr.hide()

    def update_text(self, new_text: str):
        self.text = new_text


def create_tooltip(widget, text: str) -> ToolTip:
    """Attach a tooltip to a widget."""
    return ToolTip(widget, text)


def center_window(win, parent=None, width=None, height=None):
    """Center a Toplevel window on its parent or screen."""
    win.update_idletasks()

    if width is None:
        width = win.winfo_width()
    if height is None:
        height = win.winfo_height()

    if parent and parent.winfo_exists():
        px = parent.winfo_x()
        py = parent.winfo_y()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        x = px + (pw - width) // 2
        y = py + (ph - height) // 2
    else:
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        x = (sw - width) // 2
        y = (sh - height) // 2

    x = max(0, x)
    y = max(0, y)

    win.geometry(f"{width}x{height}+{x}+{y}")

# ============================================================================
# STYLED WIDGETS — Light Theme, Colored Buttons, Icons
# ============================================================================

class StyledButton(tk.Button):
    """
    Custom styled button with color variants.
    Variants: primary, success, warning, danger, info, secondary
    """

    COLORS = {
        "primary": {"bg": COLOR_BTN_PRIMARY, "fg": COLOR_BTN_TEXT, "active": "#303F9F"},
        "success": {"bg": COLOR_BTN_SUCCESS, "fg": COLOR_BTN_TEXT, "active": "#388E3C"},
        "warning": {"bg": COLOR_BTN_WARNING, "fg": "#212121", "active": "#F57C00"},
        "danger": {"bg": COLOR_BTN_DANGER, "fg": COLOR_BTN_TEXT, "active": "#D32F2F"},
        "info": {"bg": COLOR_BTN_INFO, "fg": COLOR_BTN_TEXT, "active": "#1976D2"},
        "secondary": {"bg": "#9E9E9E", "fg": COLOR_BTN_TEXT, "active": "#757575"},
    }

    def __init__(self, parent, text: str = "", variant: str = "primary",
                 command=None, tooltip: str = "", width: int = None, **kwargs):
        colors = self.COLORS.get(variant, self.COLORS["primary"])

        super().__init__(
            parent,
            text=text,
            bg=colors["bg"],
            fg=colors["fg"],
            activebackground=colors["active"],
            activeforeground=colors["fg"],
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            cursor="hand2",
            padx=16,
            pady=6,
            command=command,
            **kwargs
        )

        if width:
            self.config(width=width)

        if tooltip:
            create_tooltip(self, tooltip)

        self.bind("<Enter>", lambda e: self.config(bg=colors["active"]))
        self.bind("<Leave>", lambda e: self.config(bg=colors["bg"]))


class StyledLabel(tk.Label):
    """Styled label with consistent typography."""

    def __init__(self, parent, text: str = "", style: str = "body",
                 tooltip: str = "", **kwargs):

        fonts = {
            "title": ("Segoe UI", 18, "bold"),
            "heading": ("Segoe UI", 14, "bold"),
            "subheading": ("Segoe UI", 12, "bold"),
            "body": ("Segoe UI", 10),
            "caption": ("Segoe UI", 9),
            "mono": ("Consolas", 10),
        }

        colors = {
            "title": COLOR_TEXT,
            "heading": COLOR_TEXT,
            "subheading": COLOR_TEXT,
            "body": COLOR_TEXT,
            "caption": COLOR_TEXT_SECONDARY,
            "mono": COLOR_TEXT,
        }

        super().__init__(
            parent,
            text=text,
            font=fonts.get(style, fonts["body"]),
            fg=colors.get(style, COLOR_TEXT),
            bg=kwargs.pop("bg", COLOR_BG),
            **kwargs
        )

        if tooltip:
            create_tooltip(self, tooltip)


class StyledEntry(tk.Entry):
    """Styled text entry field."""

    def __init__(self, parent, tooltip: str = "", width: int = 40, **kwargs):
        super().__init__(
            parent,
            font=("Segoe UI", 10),
            relief="solid",
            bd=1,
            width=width,
            bg=COLOR_BG_WHITE,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            **kwargs
        )

        if tooltip:
            create_tooltip(self, tooltip)


class StyledCombobox(ttk.Combobox):
    """Styled dropdown combobox."""

    def __init__(self, parent, values: list = None, tooltip: str = "",
                 width: int = 30, **kwargs):
        super().__init__(
            parent,
            values=values or [],
            font=("Segoe UI", 10),
            width=width,
            state="readonly",
            **kwargs
        )

        if tooltip:
            create_tooltip(self, tooltip)


class StyledCheckbox(tk.Checkbutton):
    """Styled checkbox with variable."""

    def __init__(self, parent, text: str = "", variable: tk.BooleanVar = None,
                 tooltip: str = "", command=None, **kwargs):

        if variable is None:
            variable = tk.BooleanVar(value=False)

        self.var = variable

        super().__init__(
            parent,
            text=text,
            variable=variable,
            font=("Segoe UI", 10),
            bg=kwargs.pop("bg", COLOR_BG),
            fg=COLOR_TEXT,
            activebackground=COLOR_BG,
            selectcolor=COLOR_BG_WHITE,
            command=command,
            **kwargs
        )

        if tooltip:
            create_tooltip(self, tooltip)


class StyledScale(tk.Scale):
    """Styled slider/scale widget."""

    def __init__(self, parent, from_: float = 0, to: float = 100,
                 orient: str = "horizontal", tooltip: str = "",
                 variable: tk.Variable = None, **kwargs):

        super().__init__(
            parent,
            from_=from_,
            to=to,
            orient=orient,
            font=("Segoe UI", 9),
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            troughcolor=COLOR_BG_PANEL,
            activebackground=COLOR_ACCENT,
            highlightthickness=0,
            variable=variable,
            **kwargs
        )

        if tooltip:
            create_tooltip(self, tooltip)


class StyledFrame(tk.Frame):
    """Styled frame container."""

    def __init__(self, parent, style: str = "default", **kwargs):
        colors = {
            "default": COLOR_BG,
            "panel": COLOR_BG_PANEL,
            "white": COLOR_BG_WHITE,
            "header": COLOR_BG_HEADER,
        }

        super().__init__(
            parent,
            bg=colors.get(style, COLOR_BG),
            **kwargs
        )


class StyledLabelFrame(tk.LabelFrame):
    """Styled labeled frame container."""

    def __init__(self, parent, text: str = "", tooltip: str = "", **kwargs):
        super().__init__(
            parent,
            text=text,
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            padx=10,
            pady=10,
            **kwargs
        )

        if tooltip:
            create_tooltip(self, tooltip)


class StyledListbox(tk.Listbox):
    """Styled listbox widget."""

    def __init__(self, parent, tooltip: str = "", **kwargs):
        super().__init__(
            parent,
            font=("Consolas", 10),
            bg=COLOR_BG_WHITE,
            fg=COLOR_TEXT,
            selectbackground=COLOR_ACCENT,
            selectforeground=COLOR_BTN_TEXT,
            relief="solid",
            bd=1,
            highlightthickness=0,
            **kwargs
        )

        if tooltip:
            create_tooltip(self, tooltip)


class StyledText(tk.Text):
    """Styled multi-line text widget (for logs)."""

    def __init__(self, parent, tooltip: str = "", **kwargs):
        super().__init__(
            parent,
            font=("Consolas", 9),
            bg=COLOR_BG_WHITE,
            fg=COLOR_TEXT,
            relief="solid",
            bd=1,
            wrap="word",
            **kwargs
        )

        if tooltip:
            create_tooltip(self, tooltip)


class ScrollableFrame(StyledFrame):
    """A frame with vertical scrollbar. Mousewheel only active when hovered."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.canvas = tk.Canvas(self, bg=COLOR_BG, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = StyledFrame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mousewheel only when mouse is inside this specific frame
        self._bound = False
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)
        self.scrollable_frame.bind("<Enter>", self._bind_mousewheel)
        self.scrollable_frame.bind("<Leave>", self._unbind_mousewheel)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _bind_mousewheel(self, event=None):
        if not self._bound:
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
            self._bound = True

    def _unbind_mousewheel(self, event=None):
        if self._bound:
            self.canvas.unbind_all("<MouseWheel>")
            self._bound = False

    def _on_mousewheel(self, event):
        # Only scroll if there is actually overflow
        try:
            bbox = self.canvas.bbox("all")
            if bbox:
                content_height = bbox[3] - bbox[1]
                canvas_height = self.canvas.winfo_height()
                if content_height > canvas_height:
                    self.canvas.yview_scroll(
                        int(-1 * (event.delta / 120)), "units"
                    )
        except Exception:
            pass


# ============================================================================
# PROGRESS DIALOG
# ============================================================================

class ProgressDialog(tk.Toplevel):
    """Modal progress dialog with cancel support."""

    def __init__(self, parent, title: str = "Processing...", cancelable: bool = True):
        super().__init__(parent)

        self.title(title)
        self.geometry("450x150")
        center_window(dlg, self)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.cancelled = False

        self.configure(bg=COLOR_BG)

        # Center on parent
        self.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_x()
        py = parent.winfo_y()
        w = 450
        h = 150
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        # Message label
        self.message_var = tk.StringVar(value="Initializing...")
        self.message_label = StyledLabel(self, textvariable=self.message_var, style="body")
        self.message_label.pack(pady=(20, 10))

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self, variable=self.progress_var, maximum=100, length=380
        )
        self.progress_bar.pack(pady=10)

        # Percentage
        self.percent_var = tk.StringVar(value="0%")
        self.percent_label = StyledLabel(self, textvariable=self.percent_var, style="caption")
        self.percent_label.pack()

        # Cancel button
        if cancelable:
            self.cancel_btn = StyledButton(
                self, text="Cancel", variant="danger",
                command=self._on_cancel, tooltip="Cancel the current operation"
            )
            self.cancel_btn.pack(pady=10)

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _on_cancel(self):
        self.cancelled = True

    def update_progress(self, message: str, percent: float):
        self.message_var.set(message)
        self.progress_var.set(percent)
        self.percent_var.set(f"{int(percent)}%")
        self.update()

    def is_cancelled(self) -> bool:
        return self.cancelled

    def close(self):
        self.grab_release()
        self.destroy()


# ============================================================================
# IMAGE PREVIEW CANVAS
# ============================================================================

class ImagePreviewCanvas(tk.Canvas):
    """Canvas for displaying image previews with zoom and pan."""

    def __init__(self, parent, tooltip: str = "", **kwargs):
        super().__init__(
            parent,
            bg=COLOR_BG_PANEL,
            highlightthickness=1,
            highlightbackground="#BDBDBD",
            **kwargs
        )

        self._image = None
        self._photo = None
        self._image_id = None
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._drag_start = None

        # Bindings
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<ButtonPress-1>", self._on_drag_start)
        self.bind("<B1-Motion>", self._on_drag_motion)
        self.bind("<Double-Button-1>", self._on_fit)
        self.bind("<Configure>", self._on_resize)

        if tooltip:
            create_tooltip(self, tooltip)

    def set_image(self, img: np.ndarray):
        """Set image from BGR numpy array."""
        if img is None:
            self.clear()
            return

        self._image = img
        self._render()

    def set_image_pil(self, pil_img: Image.Image):
        """Set image from PIL Image."""
        if pil_img is None:
            self.clear()
            return
        self._image = pil_to_numpy(pil_img)
        self._render()

    def clear(self):
        """Clear the canvas."""
        self._image = None
        self._photo = None
        self.delete("all")

    def _render(self):
        if self._image is None:
            return

        self.delete("all")

        h, w = self._image.shape[:2]
        new_w = int(w * self._scale)
        new_h = int(h * self._scale)

        if new_w < 1 or new_h < 1:
            return

        resized = cv2.resize(self._image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_img = numpy_to_pil(resized)
        self._photo = ImageTk.PhotoImage(pil_img)

        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        x = (canvas_w - new_w) // 2 + self._offset_x
        y = (canvas_h - new_h) // 2 + self._offset_y

        self._image_id = self.create_image(x, y, anchor="nw", image=self._photo)

    def _on_mousewheel(self, event):
        if self._image is None:
            return

        factor = 1.1 if event.delta > 0 else 0.9
        self._scale *= factor
        self._scale = max(0.1, min(10.0, self._scale))
        self._render()

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_drag_motion(self, event):
        if self._drag_start:
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            self._offset_x += dx
            self._offset_y += dy
            self._drag_start = (event.x, event.y)
            self._render()

    def _on_fit(self, event=None):
        """Fit image to canvas."""
        if self._image is None:
            return

        h, w = self._image.shape[:2]
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()

        scale_w = canvas_w / w
        scale_h = canvas_h / h
        self._scale = min(scale_w, scale_h) * 0.95
        self._offset_x = 0
        self._offset_y = 0
        self._render()

    def _on_resize(self, event):
        if self._image is not None:
            self._render()

    def zoom_in(self):
        self._scale *= 1.2
        self._scale = min(10.0, self._scale)
        self._render()

    def zoom_out(self):
        self._scale *= 0.8
        self._scale = max(0.1, self._scale)
        self._render()

    def fit(self):
        self._on_fit()

    def zoom_100(self):
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._render()


# ============================================================================
# FRAME LIST WIDGET — Table-like display for archive frames
# ============================================================================

class FrameListWidget(StyledFrame):
    """
    Custom widget for displaying list of frames with type indicators.
    Supports selection, scrolling, filtering.
    """

    def __init__(self, parent, on_select=None, **kwargs):
        super().__init__(parent, style="white", **kwargs)

        self.on_select = on_select
        self._frames: List[FrameEntry] = []
        self._filtered_indices: List[int] = []
        self._selected_index: int = -1

        # Header
        header_frame = StyledFrame(self, style="header")
        header_frame.pack(fill="x")

        headers = [("#", 50), ("Name", 180), ("Type", 50), ("Size", 80), ("Status", 70)]
        for text, width in headers:
            lbl = StyledLabel(header_frame, text=text, style="caption", bg=COLOR_BG_HEADER)
            lbl.pack(side="left", padx=5)
            lbl.config(width=width // 10)

        # List container
        list_container = StyledFrame(self, style="white")
        list_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(list_container, bg=COLOR_BG_WHITE, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.canvas.yview)

        self.list_frame = StyledFrame(self.canvas, style="white")

        self.canvas.create_window((0, 0), window=self.list_frame, anchor="nw", tags="frame")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.list_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        # self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig("frame", width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def set_frames(self, frames: List[FrameEntry]):
        """Set the frames list."""
        self._frames = frames
        self._filtered_indices = list(range(len(frames)))
        self._render_list()

    def filter_by_type(self, frame_type: Optional[int]):
        """Filter frames by type. None = show all."""
        if frame_type is None:
            self._filtered_indices = list(range(len(self._frames)))
        else:
            self._filtered_indices = [
                i for i, f in enumerate(self._frames) if f.frame_type == frame_type
            ]
        self._render_list()

    def search(self, query: str):
        """Search frames by name."""
        if not query:
            self._filtered_indices = list(range(len(self._frames)))
        else:
            query = query.lower()
            self._filtered_indices = [
                i for i, f in enumerate(self._frames) if query in f.name.lower()
            ]
        self._render_list()

    def _render_list(self):
        """Render the visible frame list."""
        for widget in self.list_frame.winfo_children():
            widget.destroy()

        for idx in self._filtered_indices:
            f = self._frames[idx]
            row = self._create_row(idx, f)
            row.pack(fill="x", pady=1)

    def _create_row(self, index: int, frame: FrameEntry) -> tk.Frame:
        """Create a single row widget."""
        bg = COLOR_BG_WHITE if index % 2 == 0 else "#F5F5F5"

        row = tk.Frame(self.list_frame, bg=bg, cursor="hand2")

        # Index
        idx_lbl = tk.Label(row, text=str(index), width=5, bg=bg, fg=COLOR_TEXT,
                           font=("Consolas", 9), anchor="w")
        idx_lbl.pack(side="left", padx=5)

        # Name
        name_lbl = tk.Label(row, text=frame.name[:25], width=20, bg=bg, fg=COLOR_TEXT,
                            font=("Consolas", 9), anchor="w")
        name_lbl.pack(side="left", padx=5)

        # Type indicator
        type_color = FRAME_TYPE_COLORS.get(frame.frame_type, COLOR_TEXT_SECONDARY)
        type_label = FRAME_TYPE_LABELS.get(frame.frame_type, "?")
        type_lbl = tk.Label(row, text=type_label, width=3, bg=type_color, fg="white",
                            font=("Consolas", 9, "bold"))
        type_lbl.pack(side="left", padx=5)

        # Size
        if frame.data_size > 0:
            size_str = human_readable_size(frame.data_size)
        elif frame.residual_size > 0:
            size_str = human_readable_size(frame.residual_size)
        else:
            size_str = "---"
        size_lbl = tk.Label(row, text=size_str, width=10, bg=bg, fg=COLOR_TEXT,
                            font=("Consolas", 9), anchor="e")
        size_lbl.pack(side="left", padx=5)

        # Status
        status = "Deleted" if frame.is_deleted else "OK"
        status_color = COLOR_DANGER if frame.is_deleted else COLOR_SUCCESS
        status_lbl = tk.Label(row, text=status, width=8, bg=bg, fg=status_color,
                              font=("Segoe UI", 9))
        status_lbl.pack(side="left", padx=5)

        # Bind click
        def _on_click(e, i=index):
            self._select(i)

        for widget in [row, idx_lbl, name_lbl, type_lbl, size_lbl, status_lbl]:
            widget.bind("<Button-1>", _on_click)

        # Tooltip
        tip_text = (
            f"Frame {index}: {frame.name}\n"
            f"Type: {FRAME_TYPE_NAMES.get(frame.frame_type, 'Unknown')}\n"
            f"Similarity: {frame.similarity_score:.3f}\n"
            f"Face Score: {frame.face_score:.3f}\n"
            f"Body Score: {frame.body_score:.3f}\n"
            f"Motion: {frame.motion_score:.2f}"
        )
        create_tooltip(row, tip_text)

        return row

    def _select(self, index: int):
        """Select a frame."""
        self._selected_index = index
        if self.on_select:
            self.on_select(index)

    def get_selected_index(self) -> int:
        return self._selected_index

    def select_index(self, index: int):
        self._selected_index = index


# ============================================================================
# HELP VIEWER — Renders HELP.LLM markup in light theme
# ============================================================================

class HelpViewer:
    """
    Renders HELP.LLM markup with light theme styling.
    Supports: # headings, [color:X], [b], [mono] blocks,
    and clickable [link:anchor] navigation.
    """

    COLORS = {
        "cyan": "#0277BD",
        "green": "#2E7D32",
        "red": "#C62828",
        "yellow": "#F57F17",
        "accent": "#3F51B5",
        "dim": "#757575",
    }

    @staticmethod
    def _slug(heading):
        return re.sub(
            r'[^a-z0-9]+', '_', heading.strip().lower()
        ).strip('_')

    @classmethod
    def open(cls, root, path=None):
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "image_archive.hld"
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            content = (
                "# Help File Not Found\n\n"
                "[color:red]HELP was not found in the "
                "application directory.[/color]\n\n"
                "Please ensure image_archive.hld is in the same folder "
                "as the application."
            )

        win = tk.Toplevel(root)
        win.title(f"{APP_NAME} — Help")
        center_window(win, root, 1050, 700)
        BG_color= "#E0B0FF"
        win.configure(bg=BG_color)
        win.minsize(750, 750)


        # Top bar with TOC button
        top_bar = tk.Frame(win, bg=COLOR_BG_HEADER)
        top_bar.pack(fill="x")

        tk.Label(
            top_bar, text=f"📖  {APP_NAME} Help",
            font=("Segoe UI", 12, "bold"),
            bg=COLOR_BG_HEADER, fg=COLOR_TEXT
        ).pack(side="left", padx=15, pady=8)

        search_var = tk.StringVar()
        search_entry = tk.Entry(
            top_bar, textvariable=search_var,
            font=("Segoe UI", 10), width=25,
            relief="solid", bd=1
        )
        search_entry.pack(side="right", padx=10, pady=8)

        tk.Label(
            top_bar, text="Search:",
            font=("Segoe UI", 10),
            bg=COLOR_BG_HEADER, fg=COLOR_TEXT
        ).pack(side="right")

        # Scrollbar + Text widget
        text_frame = tk.Frame(win, bg=COLOR_BG)
        text_frame.pack(fill="both", expand=True)

        scb = ttk.Scrollbar(text_frame)
        scb.pack(side="right", fill="y")

        txt = tk.Text(
            text_frame,
            wrap="word",
            bg=COLOR_BG_WHITE,
            fg=COLOR_TEXT,
            font=("Segoe UI", 10),
            padx=20,
            pady=15,
            state="disabled",
            relief="flat",
            spacing1=2,
            spacing2=2,
            spacing3=4,
            yscrollcommand=scb.set
        )
        scb.config(command=txt.yview)
        txt.pack(fill="both", expand=True)

        # Tag configurations — light theme
        for name, col in cls.COLORS.items():
            txt.tag_config(name, foreground=col)

        txt.tag_config(
            "h1",
            font=("Segoe UI", 16, "bold"),
            foreground=COLOR_ACCENT,
            spacing1=15,
            spacing3=8
        )
        txt.tag_config(
            "h2",
            font=("Segoe UI", 13, "bold"),
            foreground="#1A237E",
            spacing1=12,
            spacing3=6
        )
        txt.tag_config(
            "h3",
            font=("Segoe UI", 11, "bold"),
            foreground="#303F9F",
            spacing1=8,
            spacing3=4
        )
        txt.tag_config(
            "mono",
            font=("Consolas", 9),
            background="#ECEFF1",
            foreground="#37474F"
        )
        txt.tag_config(
            "b",
            font=("Segoe UI", 10, "bold")
        )
        txt.tag_config(
            "link",
            foreground=COLOR_ACCENT,
            underline=True
        )
        txt.tag_config(
            "link_hover",
            foreground="#1565C0",
            underline=True
        )
        txt.tag_config(
            "bullet",
            lmargin1=30,
            lmargin2=45
        )
        txt.tag_config(
            "separator",
            foreground="#BDBDBD"
        )
        txt.tag_config(
            "shortcut_key",
            font=("Consolas", 10, "bold"),
            foreground="#1B5E20",
            background="#E8F5E9"
        )
        txt.tag_config(
            "warning",
            foreground="#E65100",
            font=("Segoe UI", 10, "bold")
        )

        anchors = {}
        link_targets = {}
        link_counter = [0]

        # Parse and render content
        block = "text"
        for line in content.split("\n"):
            s = line.rstrip()

            if s == "[mono]":
                block = "mono"
                continue
            if s == "[/mono]":
                block = "text"
                continue
            if s.startswith("[img:") and s.endswith("]"):
                continue
            if s.startswith("[icon:") and s.endswith("]"):
                continue

            base = block

            if block == "text":
                if s.startswith("# "):
                    base = "h1"
                    s = s[2:]
                    m = re.search(r'\s*\{#([\w]+)\}\s*$', s)
                    if m:
                        anchors[m.group(1)] = txt.index("end")
                        s = s[:m.start()]
                    anchors[cls._slug(s)] = txt.index("end")
                elif s.startswith("## "):
                    base = "h2"
                    s = s[3:]
                    anchors[cls._slug(s)] = txt.index("end")
                elif s.startswith("### "):
                    base = "h3"
                    s = s[4:]
                    anchors[cls._slug(s)] = txt.index("end")
                elif s.startswith("• "):
                    base = "bullet"
                elif s.startswith("---"):
                    txt.config(state="normal")
                    txt.insert(
                        "end",
                        "─" * 70 + "\n",
                        ("separator",)
                    )
                    txt.config(state="disabled")
                    continue

            txt.config(state="normal")
            cls._inline(
                txt, s, base, anchors,
                link_targets, link_counter, win
            )
            txt.insert("end", "\n")
            txt.config(state="disabled")

        # Navigation
        def _goto(target):
            idx = anchors.get(target)
            if idx:
                txt.see(idx)
                txt.mark_set("insert", idx)

        # Bind link clicks
        for tag_name, target in link_targets.items():
            txt.tag_bind(
                tag_name, "<Button-1>",
                lambda e, t=target: _goto(t)
            )
            txt.tag_bind(
                tag_name, "<Enter>",
                lambda e: txt.config(cursor="hand2")
            )
            txt.tag_bind(
                tag_name, "<Leave>",
                lambda e: txt.config(cursor="")
            )

        # Search functionality
        def _search(*args):
            txt.tag_remove("search_hit", "1.0", "end")
            query = search_var.get().strip()
            if not query or len(query) < 2:
                return
            txt.tag_config(
                "search_hit",
                background="#FFF176",
                foreground="#212121"
            )
            start = "1.0"
            while True:
                pos = txt.search(
                    query, start, stopindex="end",
                    nocase=True
                )
                if not pos:
                    break
                end = f"{pos}+{len(query)}c"
                txt.tag_add("search_hit", pos, end)
                start = end
            # Jump to first hit
            first = txt.tag_ranges("search_hit")
            if first:
                txt.see(first[0])

        search_var.trace_add("write", _search)

        # Keyboard nav
        win.bind("<Escape>", lambda e: win.destroy())
        win.bind("<Home>", lambda e: txt.see("1.0"))
        win.bind(
            "<End>",
            lambda e: txt.see("end")
        )

        txt._goto = _goto
        win._help_text = txt

    @classmethod
    def _inline(cls, txt, line, base, anchors,
                link_targets, link_counter, win):
        """Parse inline markup and insert into text widget."""
        parts = re.split(r'(\[/?[\w:#\./]+\])', line)
        tags = {base} if base != "text" else set()
        link_target = None

        for part in parts:
            if not part:
                continue

            if part.startswith("[") and part.endswith("]") \
               and " " not in part[1:-1]:
                c = part[1:-1]

                if c.startswith("link:"):
                    link_target = c.split(":", 1)[1]
                    link_counter[0] += 1
                    tag_name = f"link_{link_counter[0]}"
                    link_targets[tag_name] = link_target
                    tags.add("link")
                    tags.add(tag_name)
                    continue

                if c == "/link":
                    tags.discard("link")
                    # Remove link-specific tags
                    tags = {
                        t for t in tags
                        if not t.startswith("link_")
                    }
                    link_target = None
                    continue

                if c.startswith("/"):
                    t = c[1:]
                    if t == "color":
                        tags = {
                            x for x in tags
                            if x not in cls.COLORS
                        }
                    elif t == "b":
                        tags.discard("b")
                    continue

                if ":" in c:
                    typ, val = c.split(":", 1)
                    if typ == "color" and val in cls.COLORS:
                        tags = {
                            x for x in tags
                            if x not in cls.COLORS
                        } | {val}
                    elif typ == "icon":
                        continue
                    continue

                if c == "b":
                    tags.add("b")
                    continue

                continue

            txt.insert("end", part, tuple(tags))

# ============================================================================
# MAIN APPLICATION WINDOW
# ============================================================================

class ImgArchiveStudioApp(tk.Tk):
    """
    Main application window.
    Full screen, light theme, tabbed interface.
    """

    def __init__(self):
        super().__init__()

        # Window setup
        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.state("zoomed")  # Full screen on Windows
        self.configure(bg=COLOR_BG)
        self.minsize(1200, 800)

        # Load settings
        self.settings = AppSettings.load(SETTINGS_FILE)

        # Initialize model manager
        self.model_manager = ModelManager(self.settings)

        # Initialize archive engine
        self.archive_engine = ArchiveEngine(self.model_manager, self.settings)

        # Project file
        self.project = ProjectFile()
        self.project_path: Optional[str] = None

        # Recent files
        self.recent_manager = RecentFilesManager(SETTINGS_FILE)

        # Shortcuts
        self.shortcut_manager = ShortcutManager(self)

        # State variables
        self.current_archive_path: Optional[str] = None
        self.source_folder: Optional[str] = None
        self.source_images: List[str] = []
        self.analysis_header: Optional[ArchiveHeader] = None
        self.analysis_frames: List[FrameEntry] = []
        self.analysis_gops: List[GOPEntry] = []

        # Setup styles
        self._setup_styles()

        # Create UI
        self._create_menu()
        self._create_toolbar()
        self._create_main_content()
        self._create_status_bar()

        # Protocol
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # Setup keyboard shortcuts
        self._setup_shortcuts()

        # Clean up stale recent files
        self.recent_manager.remove_missing()

        # Update recent menu on startup
        self._update_recent_menu()

    def _setup_styles(self):
        """Configure ttk styles for light theme."""
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background=COLOR_BG, foreground=COLOR_TEXT,
                        font=("Segoe UI", 10))
        style.configure("TNotebook", background=COLOR_BG)
        style.configure("TNotebook.Tab", background=COLOR_BG_PANEL, padding=[12, 6],
                        font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", COLOR_ACCENT)],
                  foreground=[("selected", COLOR_BTN_TEXT)])

        style.configure("TProgressbar", troughcolor=COLOR_BG_PANEL,
                        background=COLOR_ACCENT, thickness=20)

        style.configure("Treeview", background=COLOR_BG_WHITE, foreground=COLOR_TEXT,
                        fieldbackground=COLOR_BG_WHITE, font=("Consolas", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"),
                        background=COLOR_BG_HEADER, foreground=COLOR_TEXT)

    def _create_menu(self):
        """Create application menu bar with recent files."""
        menubar = Menu(self, bg=COLOR_BG, fg=COLOR_TEXT)
        self.config(menu=menubar)

        # File menu
        file_menu = Menu(menubar, tearoff=0, bg=COLOR_BG_WHITE,
                         fg=COLOR_TEXT)
        menubar.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(
            label="Open Archive...", command=self._open_archive,
            accelerator="Ctrl+O"
        )
        file_menu.add_command(
            label="New Archive from Folder...",
            command=self._new_archive_from_folder,
            accelerator="Ctrl+N"
        )
        file_menu.add_command(
            label="Open Project (.ias)...",
            command=self._open_project,
            accelerator="Ctrl+Shift+O"
        )
        file_menu.add_command(
            label="Save Project",
            command=self._save_project,
            accelerator="Ctrl+S"
        )

        file_menu.add_separator()

        # Recent files submenu
        self.recent_menu = Menu(file_menu, tearoff=0,
                                bg=COLOR_BG_WHITE, fg=COLOR_TEXT)
        file_menu.add_cascade(label="Recent Archives",
                              menu=self.recent_menu)
        self._update_recent_menu()

        file_menu.add_separator()
        file_menu.add_command(label="Save Settings",
                              command=self._save_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close,
                              accelerator="Alt+F4")

        # Export menu
        export_menu = Menu(menubar, tearoff=0, bg=COLOR_BG_WHITE,
                           fg=COLOR_TEXT)
        menubar.add_cascade(label="Export", menu=export_menu)

        export_menu.add_command(
            label="Export to Video...",
            command=self._export_video,
            accelerator="Ctrl+Shift+V"
        )
        export_menu.add_command(
            label="Generate Contact Sheet...",
            command=self._generate_contact_sheet,
            accelerator="Ctrl+Shift+C"
        )
        export_menu.add_separator()
        export_menu.add_command(
            label="Split Archive (by range)...",
            command=self._split_by_range
        )
        export_menu.add_command(
            label="Split Archive (by GOP)...",
            command=self._split_by_gop
        )

        # Tools menu
        tools_menu = Menu(menubar, tearoff=0, bg=COLOR_BG_WHITE,
                          fg=COLOR_TEXT)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(
            label="Verify Archive Integrity",
            command=self._verify_archive
        )
        tools_menu.add_command(
            label="Compact Archive",
            command=self._compact_archive
        )
        tools_menu.add_command(
            label="Repair Archive",
            command=lambda: self.viewer_tab._repair_archive()
        )
        tools_menu.add_separator()
        tools_menu.add_command(
            label="Batch Archive Builder...",
            command=self._batch_builder
        )
        tools_menu.add_separator()
        tools_menu.add_command(label="Test RIFE Model",
                               command=self._test_rife)
        tools_menu.add_command(label="Test RAFT Model",
                               command=self._test_raft)
        tools_menu.add_separator()
        tools_menu.add_command(
            label="Keyboard Shortcuts...",
            command=self._show_shortcuts
        )

        # Help menu
        help_menu = Menu(menubar, tearoff=0, bg=COLOR_BG_WHITE,
                         fg=COLOR_TEXT)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(
            label="Help Contents", command=self._show_help,
            accelerator="F1"
        )
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)

        # Keyboard bindings
        self.bind("<Control-o>", lambda e: self._open_archive())
        self.bind("<Control-n>", lambda e: self._new_archive_from_folder())
        self.bind("<Control-s>", lambda e: self._save_project())
        self.bind("<Control-Shift-O>", lambda e: self._open_project())
        self.bind("<Control-Shift-V>", lambda e: self._export_video())
        self.bind("<Control-Shift-C>",
                  lambda e: self._generate_contact_sheet())

    def _create_toolbar(self):
        """Create main toolbar with colored icon buttons."""
        toolbar = StyledFrame(self, style="header")
        toolbar.pack(fill="x", padx=0, pady=0)

        # Left side buttons
        left_frame = StyledFrame(toolbar, style="header")
        left_frame.pack(side="left", padx=10, pady=5)

        StyledButton(
            left_frame, text="📁 Open Archive", variant="primary",
            command=self._open_archive,
            tooltip="Open an existing .iarc archive file for viewing and extraction"
        ).pack(side="left", padx=3)

        StyledButton(
            left_frame, text="📂 New from Folder", variant="success",
            command=self._new_archive_from_folder,
            tooltip="Create a new archive from a folder of sequential images"
        ).pack(side="left", padx=3)

        ttk.Separator(left_frame, orient="vertical").pack(side="left", fill="y", padx=10)

        StyledButton(
            left_frame, text="🔍 Analyze", variant="info",
            command=self._analyze_source,
            tooltip="Analyze the source folder to generate a compression plan"
        ).pack(side="left", padx=3)

        StyledButton(
            left_frame, text="🔨 Build Archive", variant="success",
            command=self._build_archive,
            tooltip="Build the .iarc archive from the analyzed source"
        ).pack(side="left", padx=3)

        ttk.Separator(left_frame, orient="vertical").pack(side="left", fill="y", padx=10)

        StyledButton(
            left_frame, text="📤 Extract Selected", variant="warning",
            command=self._extract_selected,
            tooltip="Extract the currently selected frame to disk"
        ).pack(side="left", padx=3)

        StyledButton(
            left_frame, text="📤 Extract All", variant="warning",
            command=self._extract_all,
            tooltip="Extract all frames from the archive to a folder"
        ).pack(side="left", padx=3)

        # Frame counter — always visible
        counter_frame = StyledFrame(toolbar, style="header")
        counter_frame.pack(side="right", padx=20, pady=5)

        self.frame_counter_var = tk.StringVar(
            value="Frame: -- / --"
        )
        counter_label = StyledLabel(
            counter_frame, textvariable=self.frame_counter_var,
            style="mono", bg=COLOR_BG_HEADER,
            tooltip="Current frame position in the archive"
        )
        counter_label.pack(side="right")

        # Right side — GPU status
        right_frame = StyledFrame(toolbar, style="header")
        right_frame.pack(side="right", padx=10, pady=5)

        self.gpu_status_label = StyledLabel(
            right_frame, text="GPU: Checking...", style="caption", bg=COLOR_BG_HEADER
        )
        self.gpu_status_label.pack(side="right")
        create_tooltip(self.gpu_status_label, "GPU/CUDA status for AI operations")

        self.after(500, self._update_gpu_status)

    def _create_main_content(self):
        """Create the main tabbed content area."""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)

        # Tab 1: Archive Builder
        self.builder_tab = BuilderTab(self.notebook, self)
        self.notebook.add(self.builder_tab, text="  Archive Builder  ")

        # Tab 2: Archive Viewer
        self.viewer_tab = ViewerTab(self.notebook, self)
        self.notebook.add(self.viewer_tab, text="  Archive Viewer  ")

        # Tab 3: Models / Settings
        self.settings_tab = SettingsTab(self.notebook, self)
        self.notebook.add(self.settings_tab, text="  Models / Settings  ")

    def _create_status_bar(self):
        """Create bottom status bar."""
        status_frame = StyledFrame(self, style="panel")
        status_frame.pack(fill="x", side="bottom")

        self.status_var = tk.StringVar(value="Ready")
        status_label = StyledLabel(
            status_frame, textvariable=self.status_var, style="caption",
            bg=COLOR_BG_PANEL, anchor="w"
        )
        status_label.pack(side="left", padx=10, pady=5)

        self.archive_info_var = tk.StringVar(value="No archive loaded")
        archive_label = StyledLabel(
            status_frame, textvariable=self.archive_info_var, style="caption",
            bg=COLOR_BG_PANEL, anchor="e"
        )
        archive_label.pack(side="right", padx=10, pady=5)

    def _update_gpu_status(self):
        """Update GPU status indicator."""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                self.gpu_status_label.config(
                    text=f"🟢 GPU: {name} ({mem:.1f}GB)",
                    fg=COLOR_SUCCESS
                )
            else:
                self.gpu_status_label.config(text="🔴 GPU: CUDA not available", fg=COLOR_DANGER)
        except Exception:
            self.gpu_status_label.config(text="🔴 GPU: Error", fg=COLOR_DANGER)

    def set_status(self, message: str):
        """Update status bar message."""
        self.status_var.set(message)
        self.update_idletasks()

    def _on_close(self):
        """Handle application close."""
        self.archive_engine.close()
        self.model_manager.unload_all()
        self.settings.save(SETTINGS_FILE)
        self.destroy()

    # ------------------------------------------------------------------
    # Menu/Toolbar Command Stubs (implemented in tabs or below)
    # ------------------------------------------------------------------

    def _open_archive(self):
        """Open archive dialog."""
        path = filedialog.askopenfilename(
            title="Open Archive",
            filetypes=[
                ("ImgArchive Files", "*.iarc"),
                ("All Files", "*.*")
            ]
        )
        if path:
            self.viewer_tab.open_archive(path)
            self.notebook.select(1)

            # Track in recent files
            self.recent_manager.add(path)
            self._update_recent_menu()

            # Auto-load .ias project if present
            ias = find_ias_for_iarc(path)
            if ias:
                self.project = ProjectFile.load(ias)
                self.project_path = ias

    def _new_archive_from_folder(self):
        """Select folder for new archive."""
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.builder_tab.set_source_folder(folder)
            self.notebook.select(0)  # Switch to builder tab

    def _analyze_source(self):
        """Trigger analysis in builder tab."""
        self.builder_tab.analyze_sequence()

    def _build_archive(self):
        """Trigger build in builder tab."""
        self.builder_tab.build_archive()

    def _extract_selected(self):
        """Extract selected frame in viewer tab."""
        self.viewer_tab.extract_selected()

    def _extract_all(self):
        """Extract all frames in viewer tab."""
        self.viewer_tab.extract_all()

    def _verify_archive(self):
        """Verify current archive integrity."""
        self.viewer_tab.verify_archive()

    def _compact_archive(self):
        """Compact current archive via viewer tab."""
        self.viewer_tab._compact_archive()

    def _test_rife(self):
        """Test RIFE model loading."""
        self.set_status("Testing RIFE model...")
        try:
            self.model_manager.get_rife()
            messagebox.showinfo("RIFE Test", "RIFE model loaded successfully!")
            self.set_status("RIFE model OK")
        except Exception as e:
            messagebox.showerror("RIFE Test Failed", str(e))
            self.set_status("RIFE test failed")

    def _test_raft(self):
        """Test RAFT model loading."""
        self.set_status("Testing RAFT model...")
        try:
            self.model_manager.get_raft()
            messagebox.showinfo("RAFT Test", "RAFT model loaded successfully!")
            self.set_status("RAFT model OK")
        except Exception as e:
            messagebox.showerror("RAFT Test Failed", str(e))
            self.set_status("RAFT test failed")

    def _save_settings(self):
        """Save current settings."""
        self.settings_tab.save_settings()
        self.settings.save(SETTINGS_FILE)
        self.set_status("Settings saved")

    def _show_help(self):
        """Open the help viewer."""
        HelpViewer.open(self)

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About",
            f"{APP_NAME} v{APP_VERSION}\n\n"
            "Neural Keyframe Image Archive System\n\n"
            "Custom archive format for image sequences\n"
            "with AI-powered compression using RIFE interpolation.\n\n"
            "GPU accelerated • Face-safe • Body-safe\n\n"
            "(c) © Shuvro Basu, 2026."
        )

    # ------------------------------------------------------------------
    # Project file operations
    # ------------------------------------------------------------------

    def _open_project(self):
        """Open a .ias project file."""
        path = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("IARC Project", "*.ias"), ("All", "*.*")]
        )
        if not path:
            return

        self.project = ProjectFile.load(path)
        self.project_path = path

        # Apply project settings
        if self.project.source_folder and \
           os.path.isdir(self.project.source_folder):
            self.builder_tab.set_source_folder(
                self.project.source_folder
            )

        if self.project.archive_path and \
           os.path.exists(self.project.archive_path):
            self.viewer_tab.open_archive(self.project.archive_path)
            self.recent_manager.add(self.project.archive_path)
            self._update_recent_menu()
            self.notebook.select(1)

        self.set_status(f"Project loaded: {os.path.basename(path)}")

    def _save_project(self):
        """Save current state to .ias project file."""
        if not self.project_path:
            # Derive from archive path
            archive_path = self.viewer_tab.archive_var.get().strip()
            if archive_path:
                self.project_path = \
                    os.path.splitext(archive_path)[0] + ".ias"
            else:
                self.project_path = filedialog.asksaveasfilename(
                    title="Save Project",
                    defaultextension=".ias",
                    filetypes=[("IARC Project", "*.ias")]
                )

        if not self.project_path:
            return

        self.project.source_folder = \
            self.builder_tab.source_var.get().strip()
        self.project.archive_path = \
            self.viewer_tab.archive_var.get().strip()

        self.project.save(self.project_path)
        self.set_status(
            f"Project saved: {os.path.basename(self.project_path)}"
        )

    # ------------------------------------------------------------------
    # Recent files
    # ------------------------------------------------------------------

    def _update_recent_menu(self):
        """Rebuild the recent files submenu."""
        self.recent_menu.delete(0, "end")

        recent = self.recent_manager.get_list()
        if not recent:
            self.recent_menu.add_command(
                label="(no recent files)", state="disabled"
            )
            return

        for path in recent:
            name = os.path.basename(path)
            self.recent_menu.add_command(
                label=f"{name}  —  {path}",
                command=lambda p=path: self._open_recent(p)
            )

        self.recent_menu.add_separator()
        self.recent_menu.add_command(
            label="Clear Recent Files",
            command=self._clear_recent
        )

    def _open_recent(self, path: str):
        """Open a recent archive."""
        if os.path.exists(path):
            self.viewer_tab.open_archive(path)
            self.recent_manager.add(path)
            self._update_recent_menu()
            self.notebook.select(1)

            # Auto-load .ias if present
            ias = find_ias_for_iarc(path)
            if ias:
                self.project = ProjectFile.load(ias)
                self.project_path = ias
        else:
            messagebox.showwarning(
                "File Not Found",
                f"Archive no longer exists:\n{path}"
            )
            self.recent_manager.remove_missing()
            self._update_recent_menu()

    def _clear_recent(self):
        """Clear recent files list."""
        self.recent_manager.clear()
        self._update_recent_menu()

    # ------------------------------------------------------------------
    # Keyboard shortcuts setup
    # ------------------------------------------------------------------

    def _setup_shortcuts(self):
        """Register all global keyboard shortcuts."""
        sm = self.shortcut_manager

        sm.bind("<Left>", self._shortcut_prev,
                "Previous frame")
        sm.bind("<Right>", self._shortcut_next,
                "Next frame")
        sm.bind("<space>", self._shortcut_decode,
                "Decode current frame")
        sm.bind("<Control-e>", self._shortcut_extract,
                "Extract selected frame")
        sm.bind("<Home>", self._shortcut_first,
                "Go to first frame")
        sm.bind("<End>", self._shortcut_last,
                "Go to last frame")
        sm.bind("<F5>", self._shortcut_reload,
                "Reload archive")
        sm.bind("<Delete>", self._shortcut_delete,
                "Delete selected frame")
        sm.bind("<F11>",
                lambda: self.state(
                    "normal" if self.state() == "zoomed"
                    else "zoomed"
                ),
                "Toggle fullscreen")

    def _shortcut_prev(self):
        if self.notebook.index(self.notebook.select()) == 1:
            self.viewer_tab._prev_frame()

    def _shortcut_next(self):
        if self.notebook.index(self.notebook.select()) == 1:
            self.viewer_tab._next_frame()

    def _shortcut_decode(self):
        if self.notebook.index(self.notebook.select()) == 1:
            self.viewer_tab._decode_current()

    def _shortcut_extract(self):
        if self.notebook.index(self.notebook.select()) == 1:
            self.viewer_tab.extract_selected()

    def _shortcut_first(self):
        if self.notebook.index(self.notebook.select()) == 1:
            self.viewer_tab.viewer_frame_table.select_index(0)
            self.viewer_tab._on_frame_select(0)
            self.viewer_tab._decode_frame(0)

    def _shortcut_last(self):
        if self.notebook.index(self.notebook.select()) == 1:
            last = len(self.viewer_tab._frames) - 1
            if last >= 0:
                self.viewer_tab.viewer_frame_table.select_index(last)
                self.viewer_tab._on_frame_select(last)
                self.viewer_tab._decode_frame(last)

    def _shortcut_reload(self):
        if self.notebook.index(self.notebook.select()) == 1:
            path = self.viewer_tab.archive_var.get().strip()
            if path:
                self.viewer_tab.open_archive(path)

    def _shortcut_delete(self):
        if self.notebook.index(self.notebook.select()) == 1:
            self.viewer_tab._delete_selected()

    def _show_shortcuts(self):
        """Show keyboard shortcuts dialog."""
        shortcuts = self.shortcut_manager.get_all()

        dlg = tk.Toplevel(self)
        dlg.title("Keyboard Shortcuts")
        dlg.transient(self)
        dlg.configure(bg=COLOR_BG)
        center_window(dlg, self, 400, 350)

        StyledLabel(
            dlg, text="Keyboard Shortcuts", style="heading"
        ).pack(pady=(15, 10))

        text = tk.Text(
            dlg, font=("Consolas", 10), bg=COLOR_BG_WHITE,
            fg=COLOR_TEXT, relief="solid", bd=1, height=15
        )
        text.pack(fill="both", expand=True, padx=20, pady=10)

        for key, desc in shortcuts.items():
            display_key = key.replace("<", "").replace(">", "")
            text.insert("end", f"  {display_key:<20} {desc}\n")

        text.config(state="disabled")

        StyledButton(
            dlg, text="Close", variant="secondary",
            command=dlg.destroy
        ).pack(pady=10)

    # ------------------------------------------------------------------
    # Export to Video
    # ------------------------------------------------------------------

    def _export_video(self):
        """Export archive to video with optional audio."""
        if not self.viewer_tab._frames:
            messagebox.showinfo("Export", "No archive loaded.")
            return

        if not VideoExporter.check_ffmpeg():
            messagebox.showerror(
                "ffmpeg Not Found",
                "ffmpeg is required for video export.\n"
                "Please ensure ffmpeg is in your system PATH."
            )
            return

        # Export dialog
        dlg = tk.Toplevel(self)
        dlg.title("Export to Video")
        dlg.transient(self)
        dlg.configure(bg=COLOR_BG)
        center_window(dlg, self, 500, 350)


        StyledLabel(dlg, text="Export to Video",
                    style="heading").pack(pady=(15, 10))

        # Output path
        out_frame = StyledFrame(dlg)
        out_frame.pack(fill="x", padx=20, pady=5)
        StyledLabel(out_frame, text="Output:").pack(side="left")
        out_var = tk.StringVar()
        StyledEntry(out_frame, textvariable=out_var,
                    width=35).pack(side="left", padx=5)
        StyledButton(
            out_frame, text="...", variant="secondary",
            command=lambda: out_var.set(
                filedialog.asksaveasfilename(
                    title="Save Video",
                    defaultextension=".mp4",
                    filetypes=[
                        ("MP4", "*.mp4"), ("WebM", "*.webm"),
                        ("AVI", "*.avi"), ("MKV", "*.mkv")
                    ]
                ) or out_var.get()
            )
        ).pack(side="left", padx=2)

        # FPS
        fps_frame = StyledFrame(dlg)
        fps_frame.pack(fill="x", padx=20, pady=5)
        StyledLabel(fps_frame, text="FPS:").pack(side="left")
        fps_var = tk.IntVar(value=self.project.export_fps)
        StyledCombobox(
            fps_frame, values=["24", "29", "30", "60"],
            width=6, tooltip="Frames per second"
        ).pack(side="left", padx=5)

        # Audio
        aud_frame = StyledFrame(dlg)
        aud_frame.pack(fill="x", padx=20, pady=5)
        StyledLabel(aud_frame, text="Audio:").pack(side="left")
        aud_var = tk.StringVar(value=self.project.audio_file)
        StyledEntry(aud_frame, textvariable=aud_var,
                    width=35).pack(side="left", padx=5)
        StyledButton(
            aud_frame, text="...", variant="secondary",
            command=lambda: aud_var.set(
                filedialog.askopenfilename(
                    title="Select Audio",
                    filetypes=[
                        ("Audio", "*.mp3 *.wav *.aac *.m4a *.ogg"),
                        ("All", "*.*")
                    ]
                ) or aud_var.get()
            )
        ).pack(side="left", padx=2)
        StyledButton(
            aud_frame, text="Clear", variant="secondary",
            command=lambda: aud_var.set("")
        ).pack(side="left", padx=2)

        # Quality
        q_frame = StyledFrame(dlg)
        q_frame.pack(fill="x", padx=20, pady=5)
        StyledLabel(q_frame, text="Quality (CRF):").pack(side="left")
        q_var = tk.IntVar(value=23)
        StyledScale(
            q_frame, from_=15, to=35, variable=q_var,
            orient="horizontal",
            tooltip="Lower = better quality, larger file"
        ).pack(side="left", fill="x", expand=True, padx=5)

        def do_export():
            output = out_var.get().strip()
            if not output:
                messagebox.showwarning("Export", "Select output path.")
                return
            audio = aud_var.get().strip() or None
            fps = fps_var.get()
            crf = q_var.get()

            self.project.audio_file = audio or ""
            self.project.export_fps = fps

            dlg.destroy()
            self.set_status("Exporting video...")

            def export_task(progress):
                def cb(cur, tot, msg):
                    progress.update_progress(
                        "Exporting video...", cur, tot, msg
                    )

                VideoExporter.export(
                    self.archive_engine, output,
                    fps=fps, audio_path=audio,
                    quality=crf, callback=cb
                )
                return output

            def on_complete(result):
                size = human_readable_size(os.path.getsize(result))
                messagebox.showinfo(
                    "Export Complete",
                    f"Video exported:\n{result}\nSize: {size}"
                )
                self.set_status("Video export complete")

            task = ThreadedTask(
                self, export_task,
                on_complete=on_complete,
                title="Exporting Video"
            )
            task.start()

        StyledButton(
            dlg, text="Export", variant="success",
            command=do_export
        ).pack(pady=15)

    # ------------------------------------------------------------------
    # Contact Sheet
    # ------------------------------------------------------------------

    def _generate_contact_sheet(self):
        """Generate thumbnail contact sheet."""
        if not self.viewer_tab._frames:
            messagebox.showinfo("Contact Sheet", "No archive loaded.")
            return

        dlg = tk.Toplevel(self)
        dlg.title("Contact Sheet Settings")
        dlg.transient(self)
        dlg.configure(bg=COLOR_BG)
        center_window(dlg, self, 420, 320)

        StyledLabel(
            dlg, text="📋 Contact Sheet Settings",
            style="heading"
        ).pack(pady=(15, 10))

        # Every Nth frame
        nth_frame = StyledFrame(dlg)
        nth_frame.pack(fill="x", padx=30, pady=5)
        StyledLabel(
            nth_frame, text="Every Nth frame:",
            tooltip="Show every Nth frame in the grid. "
                    "E.g. 100 = every 100th frame."
        ).pack(side="left")
        nth_var = tk.IntVar(value=self.project.contact_sheet_nth)
        StyledEntry(
            nth_frame, textvariable=nth_var, width=8,
            tooltip="Number of frames to skip between thumbnails."
        ).pack(side="right", padx=5)

        # Thumbnail size
        sz_frame = StyledFrame(dlg)
        sz_frame.pack(fill="x", padx=30, pady=5)
        StyledLabel(
            sz_frame, text="Thumbnail size (px):",
            tooltip="Width and height of each thumbnail in pixels."
        ).pack(side="left")
        sz_var = tk.IntVar(
            value=self.project.contact_sheet_thumb_size
        )
        StyledEntry(
            sz_frame, textvariable=sz_var, width=8,
            tooltip="Thumbnail dimension in pixels (e.g. 128, 192, 256)."
        ).pack(side="right", padx=5)

        # Labels toggle
        lbl_var = tk.BooleanVar(value=True)
        StyledCheckbox(
            dlg, text="Show frame index labels",
            variable=lbl_var,
            tooltip="Draw frame index number below each thumbnail."
        ).pack(anchor="w", padx=30, pady=5)

        # Info line
        total = len(self.viewer_tab._frames)
        preview_count = max(1, total // max(1, nth_var.get()))
        info_var = tk.StringVar(
            value=f"~{preview_count} thumbnails from {total:,} frames"
        )
        StyledLabel(
            dlg, textvariable=info_var, style="caption"
        ).pack(pady=5)

        def update_info(*args):
            try:
                n = max(1, nth_var.get())
                count = max(1, total // n)
                info_var.set(
                    f"~{count} thumbnails from {total:,} frames"
                )
            except Exception:
                pass

        nth_var.trace_add("write", update_info)

        # Buttons
        btn_frame = StyledFrame(dlg)
        btn_frame.pack(fill="x", padx=30, pady=(15, 10))

        def do_generate():
            nth = nth_var.get()
            size = sz_var.get()
            labels = lbl_var.get()

            self.project.contact_sheet_nth = nth
            self.project.contact_sheet_thumb_size = size

            dlg.destroy()
            self.set_status("Generating contact sheet...")

            def cs_task(progress):
                def cb(cur, tot, msg):
                    progress.update_progress(
                        "Generating contact sheet...",
                        cur, tot, msg
                    )

                import tempfile
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                )
                tmp_path = tmp.name
                tmp.close()

                ContactSheetGenerator.generate(
                    self.archive_engine, tmp_path,
                    nth=nth, thumb_size=size,
                    add_labels=labels, callback=cb
                )
                return tmp_path

            def on_complete(tmp_path):
                self.set_status("Contact sheet generated")

                cs_win = tk.Toplevel(self)
                cs_win.title("Contact Sheet Preview")
                cs_win.configure(bg=COLOR_BG)
                center_window(cs_win, self, 1000, 750)

                # Top bar
                top = StyledFrame(cs_win, style="header")
                top.pack(fill="x")

                StyledLabel(
                    top, text="📋 Contact Sheet Preview",
                    style="subheading", bg=COLOR_BG_HEADER
                ).pack(side="left", padx=10, pady=5)

                def save_sheet():
                    save_path = filedialog.asksaveasfilename(
                        title="Save Contact Sheet",
                        defaultextension=".png",
                        filetypes=[
                            ("PNG", "*.png"),
                            ("JPEG", "*.jpg")
                        ]
                    )
                    if save_path:
                        shutil.copy2(tmp_path, save_path)
                        messagebox.showinfo(
                            "Saved",
                            f"Contact sheet saved to:\n{save_path}"
                        )

                StyledButton(
                    top, text="💾 Save As...",
                    variant="success",
                    command=save_sheet,
                    tooltip="Save the contact sheet to disk"
                ).pack(side="right", padx=10, pady=5)

                StyledButton(
                    top, text="🔍 100%",
                    variant="info",
                    command=lambda: canvas.zoom_100(),
                    tooltip="View at actual size"
                ).pack(side="right", padx=3, pady=5)

                StyledButton(
                    top, text="📐 Fit",
                    variant="info",
                    command=lambda: canvas.fit(),
                    tooltip="Fit to window"
                ).pack(side="right", padx=3, pady=5)

                # Preview canvas
                canvas = ImagePreviewCanvas(
                    cs_win,
                    tooltip="Scroll to zoom, drag to pan, "
                            "double-click to fit"
                )
                canvas.pack(
                    fill="both", expand=True,
                    padx=5, pady=5
                )

                sheet_img = cv2.imread(
                    tmp_path, cv2.IMREAD_COLOR
                )
                if sheet_img is not None:
                    canvas.set_image(sheet_img)
                    canvas.after(200, canvas.fit)

                def on_cs_close():
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                    cs_win.destroy()

                cs_win.protocol(
                    "WM_DELETE_WINDOW", on_cs_close
                )

            def on_error(err):
                messagebox.showerror(
                    "Contact Sheet Error", str(err)
                )

            task = ThreadedTask(
                self, cs_task,
                on_complete=on_complete,
                title="Generating Contact Sheet"
            )
            task.start()

        StyledButton(
            btn_frame, text="🖼 Generate",
            variant="success",
            command=do_generate,
            tooltip="Generate the contact sheet and show preview"
        ).pack(side="left", padx=5)

        StyledButton(
            btn_frame, text="Cancel",
            variant="secondary",
            command=dlg.destroy,
            tooltip="Close without generating"
        ).pack(side="right", padx=5)

    # ------------------------------------------------------------------
    # Split Archive
    # ------------------------------------------------------------------

    def _split_by_range(self):
        """Split archive by frame range."""
        if not self.viewer_tab._frames:
            messagebox.showinfo("Split", "No archive loaded.")
            return

        dlg = tk.Toplevel(self)
        dlg.title("Split by Range")
        dlg.transient(self)
        dlg.configure(bg=COLOR_BG)
        center_window(dlg, self, 400, 200)

        StyledLabel(dlg, text="Split by Frame Count",
                    style="heading").pack(pady=(15, 10))

        r_frame = StyledFrame(dlg)
        r_frame.pack(fill="x", padx=20, pady=5)
        StyledLabel(r_frame, text="Frames per part:").pack(side="left")
        count_var = tk.IntVar(value=5000)
        StyledEntry(
            r_frame, textvariable=count_var, width=10
        ).pack(side="left", padx=5)

        total = len(self.viewer_tab._frames)
        StyledLabel(
            dlg, text=f"Total frames: {total:,}",
            style="caption"
        ).pack(pady=5)

        def do_split():
            base = filedialog.asksaveasfilename(
                title="Output Base Name (without extension)",
                defaultextension="",
                filetypes=[("All", "*.*")]
            )
            if not base:
                return
            base = os.path.splitext(base)[0]
            dlg.destroy()

            self.set_status("Splitting archive...")

            def split_task(progress):
                def cb(cur, tot, msg):
                    progress.update_progress(
                        "Splitting...", cur, tot, msg
                    )

                return ArchiveSplitter.split_by_range(
                    self.archive_engine, [],
                    count_var.get(), base, cb
                )

            def on_complete(paths):
                messagebox.showinfo(
                    "Split Complete",
                    f"Created {len(paths)} parts:\n" +
                    "\n".join(os.path.basename(p) for p in paths[:10])
                )

            task = ThreadedTask(
                self, split_task,
                on_complete=on_complete,
                title="Splitting Archive"
            )
            task.start()

        StyledButton(
            dlg, text="Split", variant="success",
            command=do_split
        ).pack(pady=15)

    def _split_by_gop(self):
        """Split archive by GOP boundaries."""
        if not self.viewer_tab._frames:
            messagebox.showinfo("Split", "No archive loaded.")
            return

        dlg = tk.Toplevel(self)
        dlg.title("Split by GOP")
        dlg.transient(self)
        dlg.configure(bg=COLOR_BG)
        center_window(dlg, self, 400, 200)

        StyledLabel(dlg, text="Split by GOP Boundaries",
                    style="heading").pack(pady=(15, 10))

        g_frame = StyledFrame(dlg)
        g_frame.pack(fill="x", padx=20, pady=5)
        StyledLabel(g_frame, text="GOPs per part:").pack(side="left")
        gop_var = tk.IntVar(value=50)
        StyledEntry(
            g_frame, textvariable=gop_var, width=10
        ).pack(side="left", padx=5)

        gop_count = len(self.viewer_tab._gops) \
            if self.viewer_tab._gops else 0
        StyledLabel(
            dlg, text=f"Total GOPs: {gop_count}",
            style="caption"
        ).pack(pady=5)

        def do_split():
            base = filedialog.asksaveasfilename(
                title="Output Base Name",
                defaultextension="",
                filetypes=[("All", "*.*")]
            )
            if not base:
                return
            base = os.path.splitext(base)[0]
            dlg.destroy()

            self.set_status("Splitting by GOP...")

            def split_task(progress):
                def cb(cur, tot, msg):
                    progress.update_progress(
                        "Splitting...", cur, tot, msg
                    )

                return ArchiveSplitter.split_by_gop(
                    self.archive_engine,
                    gop_var.get(), base, cb
                )

            def on_complete(paths):
                messagebox.showinfo(
                    "Split Complete",
                    f"Created {len(paths)} parts:\n" +
                    "\n".join(os.path.basename(p) for p in paths[:10])
                )

            task = ThreadedTask(
                self, split_task,
                on_complete=on_complete,
                title="Splitting Archive (GOP)"
            )
            task.start()

        StyledButton(
            dlg, text="Split", variant="success",
            command=do_split
        ).pack(pady=15)

    # ------------------------------------------------------------------
    # Batch Builder
    # ------------------------------------------------------------------

    def _batch_builder(self):
        """Batch archive builder — queue multiple folders."""
        dlg = tk.Toplevel(self)
        dlg.title("Batch Archive Builder")
        dlg.transient(self)
        dlg.configure(bg=COLOR_BG)
        center_window(dlg, self, 700, 500)

        StyledLabel(dlg, text="Batch Archive Builder",
                    style="heading").pack(pady=(15, 5))
        StyledLabel(
            dlg,
            text="Add folders. Each will be built into a separate "
                 ".iarc archive using current settings.",
            style="caption"
        ).pack(pady=(0, 10))

        # Queue list
        list_frame = StyledFrame(dlg, style="white")
        list_frame.pack(fill="both", expand=True, padx=20, pady=5)

        cols = ("folder", "output", "status")
        tree = ttk.Treeview(
            list_frame, columns=cols, show="headings", height=12
        )
        tree.heading("folder", text="Source Folder")
        tree.heading("output", text="Output Archive")
        tree.heading("status", text="Status")
        tree.column("folder", width=280)
        tree.column("output", width=280)
        tree.column("status", width=80)
        tree.pack(fill="both", expand=True)

        queue_items = []

        def add_folder():
            folder = filedialog.askdirectory(
                title="Select Image Folder"
            )
            if not folder:
                return
            base = os.path.basename(folder.rstrip("/\\"))
            output = os.path.join(folder, f"{base}.iarc")

            iid = tree.insert(
                "", "end",
                values=(folder, output, "Pending")
            )
            queue_items.append({
                "iid": iid,
                "folder": folder,
                "output": output,
                "status": "Pending"
            })

        def remove_selected():
            for sel in tree.selection():
                tree.delete(sel)
                queue_items[:] = [
                    q for q in queue_items if q["iid"] != sel
                ]

        def start_batch():
            if not queue_items:
                messagebox.showinfo("Batch", "Add folders first.")
                return

            dlg.destroy()
            self.set_status("Batch build starting...")

            def batch_task(progress):
                total = len(queue_items)
                results = []

                for pos, item in enumerate(queue_items):
                    if progress.is_cancelled():
                        break

                    progress.update_progress(
                        f"Building {pos + 1}/{total}...",
                        pos, total,
                        f"Folder: {os.path.basename(item['folder'])}"
                    )

                    try:
                        images = scan_image_folder(item["folder"])
                        if not images:
                            results.append(
                                (item["folder"], "No images")
                            )
                            continue

                        analyzer = FastAnalyzer(
                            self.model_manager, self.settings
                        )
                        h, frames, gops = analyzer.analyze_sequence(
                            images, progress_callback=progress
                        )

                        if h is None:
                            results.append(
                                (item["folder"], "Cancelled")
                            )
                            break

                        self.archive_engine.build_archive(
                            item["output"], images, h, frames, gops
                        )

                        # Auto-save project
                        proj = ProjectFile.from_settings(
                            self.settings,
                            source_folder=item["folder"],
                            archive_path=item["output"]
                        )
                        auto_save_project(item["output"], proj)

                        results.append(
                            (item["folder"], "Success")
                        )
                        self.recent_manager.add(item["output"])

                    except Exception as e:
                        results.append(
                            (item["folder"], f"Error: {e}")
                        )

                return results

            def on_complete(results):
                self._update_recent_menu()
                success = sum(
                    1 for _, s in results if s == "Success"
                )
                msg = f"Batch complete: {success}/{len(results)}\n\n"
                for folder, status in results:
                    name = os.path.basename(folder)
                    msg += f"{name}: {status}\n"
                messagebox.showinfo("Batch Complete", msg)
                self.set_status("Batch build complete")

            task = ThreadedTask(
                self, batch_task,
                on_complete=on_complete,
                title="Batch Archive Builder"
            )
            task.start()

        # Buttons
        btn_frame = StyledFrame(dlg)
        btn_frame.pack(fill="x", padx=20, pady=10)

        StyledButton(
            btn_frame, text="Add Folder", variant="info",
            command=add_folder,
            tooltip="Add a source image folder to the batch queue"
        ).pack(side="left", padx=3)

        StyledButton(
            btn_frame, text="Remove Selected", variant="danger",
            command=remove_selected,
            tooltip="Remove selected entry from queue"
        ).pack(side="left", padx=3)

        StyledButton(
            btn_frame, text="Build All", variant="success",
            command=start_batch,
            tooltip="Start building archives for all queued folders"
        ).pack(side="left", padx=3)

        StyledButton(
            btn_frame, text="Close", variant="secondary",
            command=dlg.destroy
        ).pack(side="right", padx=3)
# ============================================================================
# VIRTUALIZED FRAME TABLE — Handles 100k+ frames without lag
# ============================================================================

class VirtualizedFrameTable(StyledFrame):
    """
    High-performance virtualized frame table using Treeview.
    Only renders visible rows. Handles 100k+ frames smoothly.
    Supports sorting, filtering, selection, and color-coded type indicators.
    """

    def __init__(self, parent, on_select=None, on_double_click=None, **kwargs):
        super().__init__(parent, style="white", **kwargs)

        self.on_select_callback = on_select
        self.on_double_click_callback = on_double_click
        self._all_frames: List[FrameEntry] = []
        self._filtered_indices: List[int] = []
        self._selected_index: int = -1

        # Search / filter bar
        filter_frame = StyledFrame(self)
        filter_frame.pack(fill="x", pady=(0, 5))

        StyledLabel(filter_frame, text="Search:", style="body").pack(side="left", padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_entry = StyledEntry(
            filter_frame, textvariable=self.search_var, width=20,
            tooltip="Search frames by name or index. Supports partial match."
        )
        self.search_entry.pack(side="left", padx=(0, 5))
        self.search_var.trace_add("write", self._on_search_changed)

        StyledLabel(filter_frame, text="Filter:", style="body").pack(side="left", padx=(10, 5))
        self.filter_var = tk.StringVar(value="All")
        self.filter_combo = StyledCombobox(
            filter_frame,
            values=["All", "Keyframe", "Interpolated", "Residual", "Forced KF", "Deleted"],
            width=14,
            tooltip="Filter frames by storage type."
        )
        self.filter_combo.set("All")
        self.filter_combo.pack(side="left", padx=(0, 5))
        self.filter_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)

        # Count label
        self.count_var = tk.StringVar(value="0 frames")
        self.count_label = StyledLabel(
            filter_frame, textvariable=self.count_var, style="caption",
            tooltip="Total frames matching current filter and search."
        )
        self.count_label.pack(side="right", padx=10)

        # Jump-to control
        StyledLabel(filter_frame, text="Go to:", style="body").pack(side="left", padx=(20, 5))
        self.goto_var = tk.StringVar()
        self.goto_entry = StyledEntry(
            filter_frame, textvariable=self.goto_var, width=8,
            tooltip="Jump to a specific frame index."
        )
        self.goto_entry.pack(side="left", padx=(0, 3))
        self.goto_entry.bind("<Return>", self._on_goto)

        StyledButton(
            filter_frame, text="Go", variant="info", command=self._on_goto,
            tooltip="Jump to the entered frame index."
        ).pack(side="left", padx=2)

        # Treeview with scrollbar
        tree_container = StyledFrame(self, style="white")
        tree_container.pack(fill="both", expand=True)

        columns = ("index", "name", "type", "size", "face", "body", "sim", "status")
        self.tree = ttk.Treeview(
            tree_container,
            columns=columns,
            show="headings",
            selectmode="extended",
            height=30
        )

        # Column definitions
        col_config = {
            "index": ("  #", 60, "center"),
            "name": ("  Name", 220, "w"),
            "type": ("  Type", 80, "center"),
            "size": ("  Size", 90, "e"),
            "face": ("  Face", 70, "center"),
            "body": ("  Body", 70, "center"),
            "sim": ("  Sim", 70, "center"),
            "status": ("  Status", 80, "center"),
        }

        for col_id, (heading, width, anchor) in col_config.items():
            self.tree.heading(col_id, text=heading,
                              command=lambda c=col_id: self._sort_by_column(c))
            self.tree.column(col_id, width=width, anchor=anchor, minwidth=50)

        # Tags for color coding
        self.tree.tag_configure("keyframe", foreground=COLOR_KEYFRAME)
        self.tree.tag_configure("interpolated", foreground=COLOR_INTERPOLATED)
        self.tree.tag_configure("residual", foreground=COLOR_RESIDUAL)
        self.tree.tag_configure("forced", foreground=COLOR_FORCED)
        self.tree.tag_configure("deleted", foreground=COLOR_DELETED)
        self.tree.tag_configure("even", background=COLOR_BG_WHITE)
        self.tree.tag_configure("odd", background="#F5F5F5")

        # Scrollbars
        vsb = ttk.Scrollbar(tree_container, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_container, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)

        # Bindings
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-1>", self._on_tree_double_click)

        # Sort state
        self._sort_column = "index"
        self._sort_reverse = False

        # Batch insert scheduling
        self._insert_batch_size = 5000
        self._insert_job_id = None

    def set_frames(self, frames: List[FrameEntry]):
        """Set frames data. Handles 100k+ efficiently via batch insert."""
        self._all_frames = frames
        self._apply_filters()

    def _apply_filters(self):
        """Apply current search and filter, then re-populate tree."""
        search_q = self.search_var.get().strip().lower()
        filter_val = self.filter_combo.get()

        type_map = {
            "Keyframe": FrameType.KEYFRAME,
            "Interpolated": FrameType.INTERPOLATED,
            "Residual": FrameType.RESIDUAL,
            "Forced KF": FrameType.FORCED_KEYFRAME,
            "Deleted": FrameType.DELETED,
        }

        indices = []
        for i, f in enumerate(self._all_frames):
            # Filter by type
            if filter_val != "All":
                target_type = type_map.get(filter_val)
                if target_type is not None and f.frame_type != target_type:
                    continue

            # Search by name or index
            if search_q:
                if search_q not in f.name.lower() and search_q not in str(f.index):
                    continue

            indices.append(i)

        self._filtered_indices = indices

        # Sort
        self._sort_filtered()

        # Populate tree in batches
        self._populate_tree_batched()

    def _sort_filtered(self):
        """Sort filtered indices by current sort column."""
        col = self._sort_column
        rev = self._sort_reverse

        def sort_key(idx):
            f = self._all_frames[idx]
            if col == "index":
                return f.index
            elif col == "name":
                return f.name.lower()
            elif col == "type":
                return f.frame_type
            elif col == "size":
                return f.data_size + f.residual_size
            elif col == "face":
                return f.face_score
            elif col == "body":
                return f.body_score
            elif col == "sim":
                return f.similarity_score
            elif col == "status":
                return int(f.is_deleted)
            return 0

        self._filtered_indices.sort(key=sort_key, reverse=rev)

    def _sort_by_column(self, col):
        """Sort by column header click."""
        if self._sort_column == col:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = col
            self._sort_reverse = False
        self._sort_filtered()
        self._populate_tree_batched()

    def _populate_tree_batched(self):
        """
        Populate the treeview in batches to keep UI responsive.
        For 100k frames, inserting all at once would freeze the UI.
        We insert in chunks of 5000 with scheduled callbacks.
        """
        # Cancel any pending batch job
        if self._insert_job_id is not None:
            self.after_cancel(self._insert_job_id)
            self._insert_job_id = None

        # Clear existing items
        self.tree.delete(*self.tree.get_children())

        total = len(self._filtered_indices)
        self.count_var.set(f"{total:,} frames")

        if total == 0:
            return

        self._batch_insert_pos = 0
        self._insert_next_batch()

    def _insert_next_batch(self):
        """Insert the next batch of rows into the treeview."""
        start = self._batch_insert_pos
        end = min(start + self._insert_batch_size, len(self._filtered_indices))

        for pos in range(start, end):
            idx = self._filtered_indices[pos]
            f = self._all_frames[idx]
            self._insert_frame_row(f, pos)

        self._batch_insert_pos = end

        if end < len(self._filtered_indices):
            # Schedule next batch (1ms delay to let UI breathe)
            self._insert_job_id = self.after(1, self._insert_next_batch)
        else:
            self._insert_job_id = None

    def _insert_frame_row(self, f: FrameEntry, position: int):
        """Insert a single frame row into the treeview."""
        # Type label and tag
        type_label = FRAME_TYPE_LABELS.get(f.frame_type, "?")
        type_name = FRAME_TYPE_NAMES.get(f.frame_type, "Unknown")

        tag_map = {
            FrameType.KEYFRAME: "keyframe",
            FrameType.INTERPOLATED: "interpolated",
            FrameType.RESIDUAL: "residual",
            FrameType.FORCED_KEYFRAME: "forced",
            FrameType.DELETED: "deleted",
        }
        type_tag = tag_map.get(f.frame_type, "")
        stripe_tag = "even" if position % 2 == 0 else "odd"

        # Size
        total_size = f.data_size + f.residual_size
        size_str = human_readable_size(total_size) if total_size > 0 else "---"

        # Status
        status = "Deleted" if f.is_deleted else "OK"

        values = (
            f.index,
            f.name,
            f"[{type_label}] {type_name}",
            size_str,
            f"{f.face_score:.3f}",
            f"{f.body_score:.3f}",
            f"{f.similarity_score:.3f}",
            status
        )

        self.tree.insert("", "end", iid=str(f.index), values=values,
                         tags=(type_tag, stripe_tag))

    def _on_search_changed(self, *args):
        """Debounced search — wait 300ms after typing stops."""
        if hasattr(self, "_search_debounce_id"):
            self.after_cancel(self._search_debounce_id)
        self._search_debounce_id = self.after(300, self._apply_filters)

    def _on_filter_changed(self, event=None):
        self._apply_filters()

    def _on_goto(self, event=None):
        """Jump to a specific frame index."""
        try:
            target = int(self.goto_var.get().strip())
            iid = str(target)
            if self.tree.exists(iid):
                self.tree.see(iid)
                self.tree.selection_set(iid)
                self.tree.focus(iid)
                self._selected_index = target
                if self.on_select_callback:
                    self.on_select_callback(target)
        except (ValueError, tk.TclError):
            pass

    def _on_tree_select(self, event=None):
        """Handle tree selection change."""
        selection = self.tree.selection()
        if selection:
            try:
                idx = int(selection[0])
                self._selected_index = idx
                if self.on_select_callback:
                    self.on_select_callback(idx)
            except (ValueError, IndexError):
                pass

    def _on_tree_double_click(self, event=None):
        """Handle double-click on a frame row."""
        if self._selected_index >= 0 and self.on_double_click_callback:
            self.on_double_click_callback(self._selected_index)

    def get_selected_index(self) -> int:
        return self._selected_index

    def get_selected_indices(self) -> List[int]:
        """Get all selected indices (multi-select)."""
        return [int(iid) for iid in self.tree.selection() if iid.isdigit()]

    def select_index(self, index: int):
        """Programmatically select a frame."""
        iid = str(index)
        if self.tree.exists(iid):
            self.tree.see(iid)
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            self._selected_index = index

    def refresh(self):
        """Re-render with current data and filters."""
        self._apply_filters()


# ============================================================================
# ENHANCED PROGRESS DIALOG — For 100k-scale operations
# ============================================================================

class EnhancedProgressDialog(tk.Toplevel):
    """
    Progress dialog designed for long-running 100k+ frame operations.
    Shows: message, progress bar, percentage, ETA, throughput,
    current/total count, elapsed time. Runs callback in background thread.
    """

    def __init__(self, parent, title: str = "Processing...", cancelable: bool = True):
        super().__init__(parent)

        self.title(title)
        self.geometry("550x260")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.configure(bg=COLOR_BG)

        self.cancelled = False
        self._start_time = time.time()
        self._last_update_time = time.time()
        self._last_count = 0

        # Center on parent
        self.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_x()
        py = parent.winfo_y()
        w, h = 550, 260
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        # Title
        self.msg_var = tk.StringVar(value="Initializing...")
        StyledLabel(self, textvariable=self.msg_var, style="subheading").pack(pady=(15, 5))

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        bar_frame = StyledFrame(self)
        bar_frame.pack(fill="x", padx=30, pady=5)
        self.progress_bar = ttk.Progressbar(
            bar_frame, variable=self.progress_var, maximum=100, length=490
        )
        self.progress_bar.pack(fill="x")

        # Stats grid
        stats_frame = StyledFrame(self)
        stats_frame.pack(fill="x", padx=30, pady=5)

        self.pct_var = tk.StringVar(value="0%")
        self.count_var = tk.StringVar(value="0 / 0")
        self.elapsed_var = tk.StringVar(value="Elapsed: 0s")
        self.eta_var = tk.StringVar(value="ETA: calculating...")
        self.throughput_var = tk.StringVar(value="Speed: -- frames/s")
        self.detail_var = tk.StringVar(value="")

        row1 = StyledFrame(stats_frame)
        row1.pack(fill="x")
        StyledLabel(row1, textvariable=self.pct_var, style="body").pack(side="left", padx=10)
        StyledLabel(row1, textvariable=self.count_var, style="body").pack(side="left", padx=10)
        StyledLabel(row1, textvariable=self.elapsed_var, style="caption").pack(side="right", padx=10)

        row2 = StyledFrame(stats_frame)
        row2.pack(fill="x")
        StyledLabel(row2, textvariable=self.eta_var, style="caption").pack(side="left", padx=10)
        StyledLabel(row2, textvariable=self.throughput_var, style="caption").pack(side="right", padx=10)

        row3 = StyledFrame(stats_frame)
        row3.pack(fill="x", pady=(5, 0))
        StyledLabel(row3, textvariable=self.detail_var, style="mono").pack(side="left", padx=10)

        # Cancel button
        if cancelable:
            btn_frame = StyledFrame(self)
            btn_frame.pack(pady=10)
            StyledButton(
                btn_frame, text="Cancel", variant="danger",
                command=self._on_cancel,
                tooltip="Cancel the current operation. Partial results may be lost."
            ).pack()

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _on_cancel(self):
        self.cancelled = True

    def is_cancelled(self) -> bool:
        return self.cancelled

    def update_progress(self, message: str, current: int, total: int,
                        detail: str = ""):
        """
        Update progress with full stats.
        Automatically computes %, ETA, throughput, elapsed.
        """
        if total <= 0:
            return

        pct = (current / total) * 100
        elapsed = time.time() - self._start_time

        # Throughput
        now = time.time()
        dt = now - self._last_update_time
        if dt > 0.5:
            speed = (current - self._last_count) / dt
            self._last_update_time = now
            self._last_count = current
        else:
            speed = current / elapsed if elapsed > 0 else 0

        # ETA
        if speed > 0:
            remaining = (total - current) / speed
            eta_str = self._format_time(remaining)
        else:
            eta_str = "calculating..."

        self.msg_var.set(message)
        self.progress_var.set(pct)
        self.pct_var.set(f"{pct:.1f}%")
        self.count_var.set(f"{current:,} / {total:,}")
        self.elapsed_var.set(f"Elapsed: {self._format_time(elapsed)}")
        self.eta_var.set(f"ETA: {eta_str}")
        self.throughput_var.set(f"Speed: {speed:.1f} frames/s")
        if detail:
            self.detail_var.set(detail)

        self.update()

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m}m {s}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h {m}m"

    def close(self):
        self.grab_release()
        self.destroy()


# ============================================================================
# THREADED TASK RUNNER — Non-blocking background operations
# ============================================================================

class ThreadedTask:
    """
    Runs a heavy operation in a background thread while keeping the UI responsive.
    Uses after() polling to update progress dialog from the main thread.
    """

    def __init__(self, app: tk.Tk, task_fn, on_complete=None, on_error=None,
                 title: str = "Processing...", cancelable: bool = True):
        self.app = app
        self.task_fn = task_fn
        self.on_complete = on_complete
        self.on_error = on_error

        self.progress_dialog = EnhancedProgressDialog(app, title=title, cancelable=cancelable)

        self._result = None
        self._error = None
        self._done = False
        self._thread = None

    def start(self):
        """Start the background task."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._poll()

    def _run(self):
        try:
            self._result = self.task_fn(self.progress_dialog)
        except Exception as e:
            self._error = e
            traceback.print_exc()
        finally:
            self._done = True

    def _poll(self):
        if self._done:
            self.progress_dialog.close()
            if self._error:
                if self.on_error:
                    self.on_error(self._error)
                else:
                    messagebox.showerror("Error", str(self._error))
            elif self.on_complete:
                self.on_complete(self._result)
        else:
            self.app.after(100, self._poll)


# ============================================================================
# BUILDER TAB — Full implementation for 100k+ images
# ============================================================================

class BuilderTab(StyledFrame):
    """
    Archive Builder Tab.
    Handles: folder scanning, sequence analysis, compression plan preview,
    archive building. All heavy ops are threaded for responsiveness.
    """

    def __init__(self, parent, app: 'ImgArchiveStudioApp', **kwargs):
        super().__init__(parent, **kwargs)
        self.app = app
        self.source_folder: Optional[str] = None
        self.source_images: List[str] = []
        self.analysis_header: Optional[ArchiveHeader] = None
        self.analysis_frames: List[FrameEntry] = []
        self.analysis_gops: List[GOPEntry] = []

        self._build_ui()

    def _build_ui(self):
        # ---- Top: Source & Output paths ----
        paths_frame = StyledLabelFrame(self, text="Source & Output",
                                       tooltip="Configure source image folder and archive output path.")
        paths_frame.pack(fill="x", padx=10, pady=5)

        # Source row
        src_row = StyledFrame(paths_frame)
        src_row.pack(fill="x", pady=3)

        StyledLabel(src_row, text="Source Folder:", style="body",
                    tooltip="Folder containing sequential images to archive.").pack(side="left")
        self.source_var = tk.StringVar()
        self.source_entry = StyledEntry(
            src_row, textvariable=self.source_var, width=60,
            tooltip="Path to folder containing source images (PNG, JPG, WEBP, BMP, TIFF)."
        )
        self.source_entry.pack(side="left", padx=5)
        StyledButton(
            src_row, text="Browse", variant="info", command=self._browse_source,
            tooltip="Browse for source image folder."
        ).pack(side="left", padx=3)
        StyledButton(
            src_row, text="Scan", variant="primary", command=self._scan_folder,
            tooltip="Scan the source folder and list all supported images."
        ).pack(side="left", padx=3)

        # Output row
        out_row = StyledFrame(paths_frame)
        out_row.pack(fill="x", pady=3)

        StyledLabel(out_row, text="Output Archive:", style="body",
                    tooltip="Path for the output .iarc archive file.").pack(side="left")
        self.output_var = tk.StringVar()
        self.output_entry = StyledEntry(
            out_row, textvariable=self.output_var, width=60,
            tooltip="Output .iarc file path. Will be created or overwritten."
        )
        self.output_entry.pack(side="left", padx=5)
        StyledButton(
            out_row, text="Browse", variant="info", command=self._browse_output,
            tooltip="Choose output location for the archive file."
        ).pack(side="left", padx=3)

        # ---- Middle: Split pane (Frame List Left, Options Right) ----
        middle_frame = StyledFrame(self)
        middle_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Left: Frame list (60% width)
        left_pane = StyledLabelFrame(middle_frame, text="Source Sequence",
                                     tooltip="List of source images in the selected folder.")
        left_pane.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.source_frame_table = VirtualizedFrameTable(
            left_pane,
            on_select=self._on_source_frame_select,
            on_double_click=self._on_source_frame_preview
        )
        self.source_frame_table.pack(fill="both", expand=True)

        # Right: Build options (40% width)
        right_pane = StyledLabelFrame(middle_frame, text="Build Options",
                                      tooltip="Configure compression and AI analysis settings.")
        right_pane.pack(side="right", fill="y", padx=(5, 0))
        right_pane.config(width=380)
        right_pane.pack_propagate(False)

        right_scroll = ScrollableFrame(right_pane)
        right_scroll.pack(fill="both", expand=True)
        opts = right_scroll.scrollable_frame

        # -- Compression settings --
        comp_frame = StyledLabelFrame(opts, text="Compression",
                                      tooltip="Image compression codec and quality settings.")
        comp_frame.pack(fill="x", pady=5, padx=5)

        codec_row = StyledFrame(comp_frame)
        codec_row.pack(fill="x", pady=2)
        StyledLabel(codec_row, text="Codec:", style="body").pack(side="left")
        self.codec_var = tk.StringVar(value="WebP Lossy")
        StyledCombobox(
            codec_row, values=["WebP Lossy", "JPEG"],
            width=14, tooltip="Image codec for keyframe storage. WebP recommended."
        ).pack(side="left", padx=5)

        qual_row = StyledFrame(comp_frame)
        qual_row.pack(fill="x", pady=2)
        StyledLabel(qual_row, text="Quality:", style="body").pack(side="left")
        self.quality_var = tk.IntVar(value=self.app.settings.default_quality)
        self.quality_scale = StyledScale(
            qual_row, from_=50, to=100, orient="horizontal",
            variable=self.quality_var,
            tooltip="Compression quality (50-100). Higher = better quality, larger file."
        )
        self.quality_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.quality_label = StyledLabel(qual_row, text="92", style="mono")
        self.quality_label.pack(side="right", padx=5)
        self.quality_var.trace_add("write", lambda *a: self.quality_label.config(
            text=str(self.quality_var.get())))

        # -- Sequence settings --
        seq_frame = StyledLabelFrame(opts, text="Sequence Handling",
                                     tooltip="GOP size and keyframe selection strategy.")
        seq_frame.pack(fill="x", pady=5, padx=5)

        gop_row = StyledFrame(seq_frame)
        gop_row.pack(fill="x", pady=2)
        StyledLabel(gop_row, text="GOP Size:", style="body").pack(side="left")
        self.gop_var = tk.IntVar(value=self.app.settings.default_gop_size)
        StyledScale(
            gop_row, from_=4, to=48, orient="horizontal",
            variable=self.gop_var,
            tooltip="Group of Pictures size. Smaller = more keyframes, better quality, larger archive. Larger = fewer keyframes, more interpolation."
        ).pack(side="left", fill="x", expand=True, padx=5)
        self.gop_label = StyledLabel(gop_row, text="12", style="mono")
        self.gop_label.pack(side="right", padx=5)
        self.gop_var.trace_add("write", lambda *a: self.gop_label.config(
            text=str(self.gop_var.get())))

        sim_row = StyledFrame(seq_frame)
        sim_row.pack(fill="x", pady=2)
        StyledLabel(sim_row, text="Similarity Threshold:", style="body").pack(side="left")
        self.sim_var = tk.DoubleVar(value=self.app.settings.similarity_threshold)
        StyledScale(
            sim_row, from_=0.80, to=0.99, orient="horizontal",
            variable=self.sim_var, resolution=0.01,
            tooltip="Minimum SSIM similarity. Frames below this get stored as keyframes."
        ).pack(side="left", fill="x", expand=True, padx=5)

        # -- AI Feature toggles --
        ai_frame = StyledLabelFrame(opts, text="AI Analysis",
                                    tooltip="Enable/disable AI-powered analysis features.")
        ai_frame.pack(fill="x", pady=5, padx=5)

        self.face_safe_var = tk.BooleanVar(value=self.app.settings.face_safe_mode)
        StyledCheckbox(
            ai_frame, text="Face-Safe Mode", variable=self.face_safe_var,
            tooltip="Detect faces and prevent quality loss on facial regions. Uses InsightFace + face parsing."
        ).pack(anchor="w", pady=1)

        self.body_safe_var = tk.BooleanVar(value=self.app.settings.body_safe_mode)
        StyledCheckbox(
            ai_frame, text="Body-Safe Mode", variable=self.body_safe_var,
            tooltip="Detect human bodies and prevent quality loss on body regions. Uses YOLO segmentation + pose."
        ).pack(anchor="w", pady=1)

        self.residual_var = tk.BooleanVar(value=self.app.settings.keep_residuals)
        StyledCheckbox(
            ai_frame, text="Keep Residuals", variable=self.residual_var,
            tooltip="Store residual correction patches for interpolated frames. Improves quality at cost of size."
        ).pack(anchor="w", pady=1)

        self.scene_cut_var = tk.BooleanVar(value=self.app.settings.use_scene_cut_detection)
        StyledCheckbox(
            ai_frame, text="Scene Cut Detection", variable=self.scene_cut_var,
            tooltip="Detect hard scene cuts and force keyframe boundaries. Uses histogram analysis."
        ).pack(anchor="w", pady=1)

        self.flow_var = tk.BooleanVar(value=self.app.settings.use_optical_flow)
        StyledCheckbox(
            ai_frame, text="Optical Flow Analysis (RAFT)", variable=self.flow_var,
            tooltip="Use RAFT optical flow for motion scoring. More accurate but slower."
        ).pack(anchor="w", pady=1)

        self.identity_var = tk.BooleanVar(value=self.app.settings.use_identity_check)
        StyledCheckbox(
            ai_frame, text="Identity Consistency Check", variable=self.identity_var,
            tooltip="Verify face identity is consistent across GOP. Prevents person-change frames from being interpolated."
        ).pack(anchor="w", pady=1)

        self.downscale_var = tk.BooleanVar(value=self.app.settings.use_archive_downscale)
        StyledCheckbox(
            ai_frame, text="Archive Downscale (store reduced)", variable=self.downscale_var,
            tooltip="Store keyframes at reduced resolution. Requires upscaler on extraction. Toggle in Settings."
        ).pack(anchor="w", pady=1)

        self.depth_var = tk.BooleanVar(value=self.app.settings.use_depth_aware)
        StyledCheckbox(
            ai_frame, text="Depth-Aware Compression", variable=self.depth_var,
            tooltip="Use Depth Anything for depth-guided residual placement. Toggle in Settings."
        ).pack(anchor="w", pady=1)

        # Archive mode
        archmode_frame = StyledLabelFrame(
            opts, text="Archive Mode",
            tooltip="Select based on your image collection type."
        )
        archmode_frame.pack(fill="x", pady=5, padx=5)

        self.archmode_var = tk.StringVar(
            value=self.app.settings.archive_mode
        )

        rb_seq = tk.Radiobutton(
            archmode_frame,
            text="Sequence Mode — Video frames, sequential images",
            variable=self.archmode_var, value="sequence",
            font=("Segoe UI", 9), bg=COLOR_BG, fg=COLOR_TEXT,
            activebackground=COLOR_BG, selectcolor=COLOR_BG_WHITE
        )
        rb_seq.pack(anchor="w", pady=1)
        create_tooltip(
            rb_seq,
            "Use for video frames or sequential screenshots.\n"
            "Frames must be in order. RIFE interpolates between "
            "consecutive similar frames."
        )

        rb_gal = tk.Radiobutton(
            archmode_frame,
            text="Gallery Mode — Mixed images, grouped by face",
            variable=self.archmode_var, value="gallery",
            font=("Segoe UI", 9), bg=COLOR_BG, fg=COLOR_TEXT,
            activebackground=COLOR_BG, selectcolor=COLOR_BG_WHITE
        )
        rb_gal.pack(anchor="w", pady=1)
        create_tooltip(
            rb_gal,
            "Use for collections of different images per person.\n"
            "Images are auto-grouped by face identity, sorted by\n"
            "visual similarity within each group, then compressed."
        )

        # Compression mode
        mode_frame = StyledLabelFrame(opts, text="Compression Mode",
                                      tooltip="Overall compression vs quality trade-off.")
        mode_frame.pack(fill="x", pady=5, padx=5)

        self.mode_var = tk.StringVar(value=self.app.settings.compression_mode)
        for mode, desc in [
            ("max_compression", "Max Compression — Smallest file, more GPU on extract"),
            ("balanced", "Balanced — Good compression with fast extraction"),
            ("fast_access", "Fast Access — Larger file, minimal decode cost"),
        ]:
            rb = tk.Radiobutton(
                mode_frame, text=desc, variable=self.mode_var, value=mode,
                font=("Segoe UI", 9), bg=COLOR_BG, fg=COLOR_TEXT,
                activebackground=COLOR_BG, selectcolor=COLOR_BG_WHITE
            )
            rb.pack(anchor="w", pady=1)
            create_tooltip(rb, f"Compression mode: {mode}")

        # ---- Bottom: Analysis results & Build controls ----
        bottom_frame = StyledFrame(self)
        bottom_frame.pack(fill="x", padx=10, pady=5)

        # Analysis results panel
        self.analysis_frame = StyledLabelFrame(
            bottom_frame, text="Analysis Results",
            tooltip="Results of sequence analysis. Run 'Analyze Sequence' first."
        )
        self.analysis_frame.pack(fill="x", pady=5)

        self.analysis_text = StyledText(
            self.analysis_frame, height=5,
            tooltip="Compression plan summary and estimated savings."
        )
        self.analysis_text.pack(fill="x", padx=5, pady=5)
        self.analysis_text.insert("1.0", "No analysis performed yet. Click 'Analyze Sequence' to start.")
        self.analysis_text.config(state="disabled")

        # Build controls row
        build_row = StyledFrame(bottom_frame)
        build_row.pack(fill="x", pady=5)

        StyledButton(
            build_row, text="🔍 Analyze Sequence", variant="info",
            command=self.analyze_sequence,
            tooltip="Analyze source images: compute similarity, detect faces/bodies, plan compression. Required before build."
        ).pack(side="left", padx=5)

        StyledButton(
            build_row, text="📋 Preview Plan", variant="primary",
            command=self._preview_plan,
            tooltip="Show the frame-by-frame compression plan before building."
        ).pack(side="left", padx=5)

        StyledButton(
            build_row, text="🔨 Build Archive", variant="success",
            command=self.build_archive,
            tooltip="Build the .iarc archive using the analyzed compression plan."
        ).pack(side="left", padx=5)

        # Build log
        log_label_frame = StyledLabelFrame(bottom_frame, text="Build Log",
                                            tooltip="Detailed log of build operations.")
        log_label_frame.pack(fill="x", pady=5)

        self.log_text = StyledText(
            log_label_frame, height=4,
            tooltip="Live build output and diagnostic messages."
        )
        self.log_text.pack(fill="x", padx=5, pady=5)

    def log(self, message: str):
        """Append message to build log."""
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def set_source_folder(self, folder: str):
        """Set source folder and auto-scan."""
        self.source_folder = folder
        self.source_var.set(folder)
        self._scan_folder()

    def _browse_source(self):
        folder = filedialog.askdirectory(title="Select Source Image Folder")
        if folder:
            self.set_source_folder(folder)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save Archive As",
            defaultextension=ARCHIVE_EXTENSION,
            filetypes=[("ImgArchive Files", "*.iarc"), ("All Files", "*.*")]
        )
        if path:
            self.output_var.set(path)

    def _scan_folder(self):
        """Scan source folder for images — threaded for 100k folders."""
        folder = self.source_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Warning", "Please select a valid source folder.")
            return

        self.app.set_status(f"Scanning {folder}...")
        self.log(f"Scanning folder: {folder}")

        def scan_task(progress: EnhancedProgressDialog):
            progress.update_progress("Scanning folder for images...", 0, 1)
            files = []
            all_files = sorted(os.listdir(folder))
            total_files = len(all_files)

            for i, f in enumerate(all_files):
                if progress.is_cancelled():
                    return []
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    files.append(os.path.join(folder, f))
                if i % 1000 == 0:
                    progress.update_progress(
                        f"Scanning: found {len(files):,} images...",
                        i, total_files,
                        detail=f"Checking: {f}"
                    )

            return files

        def on_complete(result):
            self.source_images = result
            count = len(result)
            self.log(f"Found {count:,} images")
            self.app.set_status(f"Found {count:,} images in source folder")

            # Create lightweight FrameEntry list for display (no analysis yet)
            display_frames = []
            for i, path in enumerate(result):
                display_frames.append(FrameEntry(
                    index=i,
                    name=os.path.basename(path),
                    frame_type=FrameType.KEYFRAME,
                    width=0,
                    height=0,
                ))
            self.source_frame_table.set_frames(display_frames)

            # Auto-suggest output path
            if not self.output_var.get():
                base = os.path.basename(folder.rstrip("/\\"))
                self.output_var.set(os.path.join(folder, f"{base}{ARCHIVE_EXTENSION}"))

        def on_error(err):
            messagebox.showerror("Scan Error", str(err))
            self.log(f"Scan error: {err}")

        task = ThreadedTask(
            self.app, scan_task,
            on_complete=on_complete, on_error=on_error,
            title="Scanning Folder"
        )
        task.start()

    def _sync_settings(self):
        """Sync UI options to app settings before analysis/build."""
        s = self.app.settings
        s.default_quality = self.quality_var.get()
        s.default_gop_size = self.gop_var.get()
        s.similarity_threshold = self.sim_var.get()
        s.face_safe_mode = self.face_safe_var.get()
        s.body_safe_mode = self.body_safe_var.get()
        s.keep_residuals = self.residual_var.get()
        s.use_scene_cut_detection = self.scene_cut_var.get()
        s.use_optical_flow = self.flow_var.get()
        s.use_identity_check = self.identity_var.get()
        s.use_archive_downscale = self.downscale_var.get()
        s.use_depth_aware = self.depth_var.get()
        s.compression_mode = self.mode_var.get()
        s.archive_mode = self.archmode_var.get()

    def analyze_sequence(self):
        """Run fast multi-phase analysis — optimized for 45k-100k+ images."""
        if not self.source_images:
            messagebox.showwarning("Warning", "No source images. Scan a folder first.")
            return

        self._sync_settings()
        self.log("Starting fast multi-phase analysis...")
        self.log(f"Total images: {len(self.source_images):,}")
        self.app.set_status("Analyzing sequence (fast mode)...")

        image_paths = self.source_images

        # def analyze_task(progress: EnhancedProgressDialog):
        #     if self.app.settings.archive_mode == "gallery":
        #         analyzer = GalleryAnalyzer(
        #             self.app.model_manager, self.app.settings
        #         )
        #     else:
        #         analyzer = FastAnalyzer(
        #             self.app.model_manager, self.app.settings
        #         )
        #     result = analyzer.analyze_sequence(
        #         image_paths, progress_callback=progress
        #     )
        #     return result
        analyzer_ref = [None]

        def analyze_task(progress: EnhancedProgressDialog):
            if self.app.settings.archive_mode == "gallery":
                analyzer = GalleryAnalyzer(
                    self.app.model_manager, self.app.settings
                )
            else:
                analyzer = FastAnalyzer(
                    self.app.model_manager, self.app.settings
                )
            analyzer_ref[0] = analyzer
            result = analyzer.analyze_sequence(
                image_paths, progress_callback=progress
            )
            return result

        def on_complete(result):
            if result is None:
                self.log("Analysis cancelled.")
                return

            header, frames, gops = result
            self.analysis_header = header
            self.analysis_frames = frames
            self.analysis_gops = gops
            # Keep analyzer reference for gallery mode sorted paths
            if self.app.settings.archive_mode == "gallery":
                self._last_analyzer = analyzer_ref[0]

            self.source_frame_table.set_frames(frames)

            total_kf = header.keyframe_count + header.forced_keyframe_count
            est_kf_size = total_kf * 180 * 1024
            est_res_size = header.residual_count * 18 * 1024
            est_total = est_kf_size + est_res_size
            reduction = (1.0 - est_total / max(1, header.original_total_bytes)) * 100

            analysis_msg = (
                f"{'=' * 50}\n"
                f"ANALYSIS RESULTS\n"
                f"{'=' * 50}\n"
                f"Total Images:         {header.total_frames:,}\n"
                f"Keyframes:            {header.keyframe_count:,}\n"
                f"Forced Keyframes:     {header.forced_keyframe_count:,}\n"
                f"Interpolated:         {header.interpolated_count:,}\n"
                f"Residual:             {header.residual_count:,}\n"
                f"GOPs:                 {header.gop_count:,}\n"
                f"{'=' * 50}\n"
                f"Original Size:        {human_readable_size(header.original_total_bytes)}\n"
                f"Estimated Archive:    {human_readable_size(est_total)}\n"
                f"Estimated Reduction:  {reduction:.1f}%\n"
                f"Resolution:           {header.original_width}x{header.original_height}\n"
                f"{'=' * 50}\n"
                f"Keyframe Ratio:       "
                f"{(total_kf / max(1, header.total_frames)) * 100:.1f}%\n"
                f"Interpolation Ratio:  "
                f"{(header.interpolated_count / max(1, header.total_frames)) * 100:.1f}%\n"
            )

            self.analysis_text.config(state="normal")
            self.analysis_text.delete("1.0", "end")
            self.analysis_text.insert("1.0", analysis_msg)
            self.analysis_text.config(state="disabled")

            self.log(
                f"Analysis complete: {header.total_frames:,} frames, "
                f"{total_kf:,} keyframes "
                f"({(total_kf / max(1, header.total_frames)) * 100:.1f}%), "
                f"{header.interpolated_count:,} interpolated, "
                f"{header.residual_count:,} residual"
            )
            self.log(f"Estimated reduction: {reduction:.1f}%")
            self.app.set_status("Analysis complete")

        def on_error(err):
            self.log(f"Analysis error: {err}")
            traceback.print_exc()
            messagebox.showerror("Analysis Error", str(err))

        task = ThreadedTask(
            self.app, analyze_task,
            on_complete=on_complete, on_error=on_error,
            title="Analyzing Sequence (Fast Mode)"
        )
        task.start()

    def build_archive(self):
        """Build the .iarc archive from analyzed data — threaded for 100k."""
        if not self.analysis_frames:
            messagebox.showwarning("Warning", "Run 'Analyze Sequence' first.")
            return

        output_path = self.output_var.get().strip()
        if not output_path:
            messagebox.showwarning("Warning", "Please specify an output archive path.")
            return

        if os.path.exists(output_path):
            if not messagebox.askyesno("Confirm Overwrite",
                                       f"File exists:\n{output_path}\n\nOverwrite?"):
                return

        self.log(f"Building archive: {output_path}")
        self.app.set_status("Building archive...")

        header = self.analysis_header
        frames = self.analysis_frames
        gops = self.analysis_gops
        image_paths = self.source_images
        # If gallery mode, use sorted paths from analyzer
        if hasattr(self, 'analysis_frames') and self.analysis_frames:
            analyzer_obj = getattr(self, '_last_analyzer', None)
            if analyzer_obj and hasattr(analyzer_obj, '_sorted_paths'):
                image_paths = analyzer_obj._sorted_paths
        quality = header.compression_quality
        use_downscale = header.archive_downscale
        downscale = header.downscale_factor
        settings = self.app.settings

        def build_task(progress: EnhancedProgressDialog):
            total = len(frames)
            engine = self.app.archive_engine

            data_blobs = {}
            residual_blobs = {}

            for i, f in enumerate(frames):
                if progress.is_cancelled():
                    return None

                if f.frame_type in (FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME):
                    img = load_image_file(image_paths[i])
                    if use_downscale and downscale < 1.0:
                        img = resize_image(img, downscale)
                    blob = image_to_webp_bytes(img, quality)
                    data_blobs[i] = blob
                    f.data_size = len(blob)
                    f.checksum = compute_checksum(blob)
                    del img

                elif f.frame_type == FrameType.RESIDUAL:
                    try:
                        original = load_image_file(image_paths[i])
                        img_a = load_image_file(image_paths[f.parent_keyframe_a])
                        img_b = load_image_file(image_paths[f.parent_keyframe_b])

                        if use_downscale and downscale < 1.0:
                            img_a = resize_image(img_a, downscale)
                            img_b = resize_image(img_b, downscale)
                            original = resize_image(original, downscale)

                        interpolated = engine.models.rife_interpolate(
                            img_a, img_b, f.interpolation_timestep
                        )
                        residual = cv2.subtract(original, interpolated)
                        res_blob = image_to_webp_bytes(residual, max(50, quality - 20))
                        residual_blobs[i] = res_blob
                        f.residual_size = len(res_blob)

                        del original, img_a, img_b, interpolated, residual
                    except Exception:
                        # Fallback: store as keyframe
                        img = load_image_file(image_paths[i])
                        if use_downscale and downscale < 1.0:
                            img = resize_image(img, downscale)
                        blob = image_to_webp_bytes(img, quality)
                        data_blobs[i] = blob
                        f.frame_type = FrameType.KEYFRAME
                        f.data_size = len(blob)
                        f.checksum = compute_checksum(blob)
                        del img

                progress.update_progress(
                    "Encoding frames...", i + 1, total,
                    detail=f"Frame {f.name} → {FRAME_TYPE_LABELS.get(f.frame_type, '?')}"
                )

            if progress.is_cancelled():
                return None

            # Assign data offsets
            progress.update_progress("Writing archive...", total, total, detail="Computing offsets...")
            current_offset = 0
            for i, f in enumerate(frames):
                if i in data_blobs:
                    f.data_offset = current_offset
                    f.data_size = len(data_blobs[i])
                    current_offset += f.data_size
                else:
                    f.data_offset = 0
                    if f.frame_type not in (FrameType.RESIDUAL,):
                        f.data_size = 0

                if i in residual_blobs:
                    f.residual_offset = current_offset
                    f.residual_size = len(residual_blobs[i])
                    current_offset += f.residual_size
                else:
                    f.residual_offset = 0
                    if f.frame_type != FrameType.RESIDUAL:
                        f.residual_size = 0

            # Serialize
            header.created_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            header_json = json.dumps(header.to_dict(), separators=(",", ":")).encode("utf-8")
            index_data = [f.to_dict() for f in frames]
            index_json = json.dumps(index_data, separators=(",", ":")).encode("utf-8")
            header.index_size = len(index_json)

            # Write file
            with open(output_path, "wb") as fp:
                fp.write(ARCHIVE_MAGIC)
                fp.write(struct.pack("<I", ARCHIVE_FORMAT_VERSION))
                fp.write(struct.pack("<Q", len(header_json)))
                fp.write(struct.pack("<Q", len(index_json)))
                fp.write(header_json)
                fp.write(index_json)

                for i in range(total):
                    if i in data_blobs:
                        fp.write(data_blobs[i])
                    if i in residual_blobs:
                        fp.write(residual_blobs[i])

                # Footer — write placeholder
                fp.write(b"\x00" * 64)
                archive_size = fp.tell()

            # File is now closed. Reopen as read+write to patch checksum.
            with open(output_path, "r+b") as fp:
                fp.seek(0)
                all_data = fp.read(archive_size - 64)
                checksum = compute_checksum(all_data)
                fp.seek(archive_size - 64)
                fp.write(checksum.encode("ascii")[:64].ljust(64, b"\x00"))

            header.archive_total_bytes = archive_size
            if header.original_total_bytes > 0:
                header.reduction_percent = round(
                    (1.0 - archive_size / header.original_total_bytes) * 100, 1
                )

            return output_path, archive_size

        def on_complete(result):
            if result is None:
                self.log("Build cancelled.")
                return
            path, size = result
            self.log(f"Archive built successfully: {path}")
            self.log(f"Archive size: {human_readable_size(size)}")
            self.app.set_status(f"Archive built: {human_readable_size(size)}")
            messagebox.showinfo("Build Complete",
                                f"Archive built successfully!\n\n"
                                f"Path: {path}\n"
                                f"Size: {human_readable_size(size)}")
            # Auto-save .ias project file alongside archive
            try:
                proj = ProjectFile.from_settings(
                    self.app.settings,
                    source_folder=self.source_var.get().strip(),
                    archive_path=path,
                    audio_file=self.app.project.audio_file
                        if self.app.project else ""
                )
                auto_save_project(path, proj)
                self.app.project = proj
                self.app.project_path = os.path.splitext(path)[0] + ".ias"
                self.log(f"Project saved: {self.app.project_path}")
            except Exception as e:
                self.log(f"Warning: Could not save .ias project: {e}")

            # Add to recent files
            self.app.recent_manager.add(path)
            self.app._update_recent_menu()

        def on_error(err):
            self.log(f"Build error: {err}")
            messagebox.showerror("Build Error", str(err))

        task = ThreadedTask(
            self.app, build_task,
            on_complete=on_complete, on_error=on_error,
            title="Building Archive"
        )
        task.start()

    def _preview_plan(self):
        """Show compression plan in the frame table."""
        if not self.analysis_frames:
            messagebox.showinfo("Preview", "Run 'Analyze Sequence' first.")
            return
        self.source_frame_table.set_frames(self.analysis_frames)
        self.log("Compression plan loaded into frame table. Use filters to inspect.")

    def _on_source_frame_select(self, index: int):
        """Handle frame selection in source table."""
        if index < 0:
            return
        if index < len(self.source_images):
            self.app.set_status(f"Selected: {os.path.basename(self.source_images[index])}")

    def _on_source_frame_preview(self, index: int):
        """Handle double-click preview in source table."""
        if index < 0 or index >= len(self.source_images):
            return
        try:
            img = load_image_file(self.source_images[index])
            preview_win = tk.Toplevel(self.app)
            preview_win.title(f"Preview: {os.path.basename(self.source_images[index])}")
            preview_win.configure(bg=COLOR_BG)
            center_window(preview_win, self.app, 900, 700)
            # center_window(dlg, self)


            canvas = ImagePreviewCanvas(
                preview_win,
                tooltip="Image preview. Scroll to zoom, drag to pan, double-click to fit."
            )
            canvas.pack(fill="both", expand=True, padx=10, pady=10)
            canvas.set_image(img)
            canvas.after(100, canvas.fit)
        except Exception as e:
            messagebox.showerror("Preview Error", str(e))


# ============================================================================
# VIEWER TAB — Full implementation for 100k+ archives
# ============================================================================

class ViewerTab(StyledFrame):
    """
    Archive Viewer Tab.
    Handles: opening archives, listing frames, on-the-fly decode,
    preview, extraction, deletion, quality checks.
    All heavy ops threaded. Frame list virtualized for 100k+.
    """

    def __init__(self, parent, app: 'ImgArchiveStudioApp', **kwargs):
        super().__init__(parent, **kwargs)
        self.app = app
        self._header: Optional[ArchiveHeader] = None
        self._frames: List[FrameEntry] = []
        self._gops: List[GOPEntry] = []
        self._current_decoded: Optional[np.ndarray] = None
        self._decode_cache: OrderedDict = OrderedDict()
        self._cache_max = 50  # Keep last 50 decoded frames in memory

        self._build_ui()

    def _build_ui(self):
        # ---- Top: Archive path ----
        path_frame = StyledLabelFrame(self, text="Archive",
                                      tooltip="Open an .iarc archive file to view and extract frames.")
        path_frame.pack(fill="x", padx=10, pady=5)

        path_row = StyledFrame(path_frame)
        path_row.pack(fill="x")

        StyledLabel(path_row, text="Archive:", style="body").pack(side="left")
        self.archive_var = tk.StringVar()
        self.archive_entry = StyledEntry(
            path_row, textvariable=self.archive_var, width=70,
            tooltip="Path to the .iarc archive file."
        )
        self.archive_entry.pack(side="left", padx=5)

        StyledButton(
            path_row, text="Open", variant="primary",
            command=self._browse_archive,
            tooltip="Browse for an .iarc archive file."
        ).pack(side="left", padx=3)

        StyledButton(
            path_row, text="Reload", variant="info",
            command=self._reload_archive,
            tooltip="Reload the currently open archive."
        ).pack(side="left", padx=3)

        # ---- Main: Three-pane split (List | Preview | Info) ----
        main_frame = StyledFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Left: Frame list
        left_pane = StyledLabelFrame(main_frame, text="Frame Index",
                                     tooltip="List of all frames in the archive. Click to select, double-click to decode.")
        left_pane.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.viewer_frame_table = VirtualizedFrameTable(
            left_pane,
            on_select=self._on_frame_select,
            on_double_click=self._on_frame_decode
        )
        self.viewer_frame_table.pack(fill="both", expand=True)

        # Center: Preview
        center_pane = StyledLabelFrame(main_frame, text="Preview",
                                       tooltip="Decoded frame preview. Double-click a frame to decode. Scroll to zoom.")
        center_pane.pack(side="left", fill="both", expand=True, padx=5)
        center_pane.config(width=500)

        self.preview_canvas = ImagePreviewCanvas(
            center_pane,
            tooltip="Image preview. Scroll to zoom, drag to pan, double-click to fit."
        )
        self.preview_canvas.pack(fill="both", expand=True)

        # Preview controls
        preview_controls = StyledFrame(center_pane)
        preview_controls.pack(fill="x", pady=5)

        StyledButton(
            preview_controls, text="◀ Prev", variant="secondary",
            command=self._prev_frame,
            tooltip="Go to previous frame."
        ).pack(side="left", padx=2)

        StyledButton(
            preview_controls, text="Next ▶", variant="secondary",
            command=self._next_frame,
            tooltip="Go to next frame."
        ).pack(side="left", padx=2)

        StyledButton(
            preview_controls, text="🔍 Decode", variant="primary",
            command=self._decode_current,
            tooltip="Decode and display the selected frame. For interpolated frames, this runs RIFE on GPU."
        ).pack(side="left", padx=2)

        StyledButton(
            preview_controls, text="Fit", variant="info",
            command=lambda: self.preview_canvas.fit(),
            tooltip="Fit image to preview area."
        ).pack(side="left", padx=2)

        StyledButton(
            preview_controls, text="100%", variant="info",
            command=lambda: self.preview_canvas.zoom_100(),
            tooltip="View at 100% zoom."
        ).pack(side="left", padx=2)

        self.decode_time_var = tk.StringVar(value="")
        StyledLabel(
            preview_controls, textvariable=self.decode_time_var, style="caption",
            tooltip="Time taken to decode the current frame."
        ).pack(side="right", padx=10)

        # Right: Info panel
        right_pane = StyledLabelFrame(main_frame, text="Info",
                                      tooltip="Archive and selected frame information.")
        right_pane.pack(side="right", fill="y", padx=(5, 0))
        right_pane.config(width=280)
        right_pane.pack_propagate(False)

        # Archive info
        arch_info_label = StyledLabelFrame(right_pane, text="Archive Info",
                                           tooltip="Summary of the loaded archive.")
        arch_info_label.pack(fill="x", pady=5)

        self.archive_info_text = StyledText(
            arch_info_label, height=8,
            tooltip="Archive metadata: frame counts, codec, quality, features used."
        )
        self.archive_info_text.pack(fill="x", padx=5, pady=5)
        self.archive_info_text.config(state="disabled")

        # Frame detail
        frame_info_label = StyledLabelFrame(right_pane, text="Frame Detail",
                                            tooltip="Details of the currently selected frame.")
        frame_info_label.pack(fill="x", pady=5)

        self.frame_info_text = StyledText(
            frame_info_label, height=10,
            tooltip="Selected frame details: type, storage mode, parents, scores."
        )
        self.frame_info_text.pack(fill="x", padx=5, pady=5)
        self.frame_info_text.config(state="disabled")

        # ---- Bottom: Action buttons ----
        action_frame = StyledFrame(self)
        action_frame.pack(fill="x", padx=10, pady=5)

        StyledButton(
            action_frame, text="📤 Extract Selected", variant="warning",
            command=self.extract_selected,
            tooltip="Extract the selected frame(s) to disk as PNG/WebP."
        ).pack(side="left", padx=3)

        StyledButton(
            action_frame, text="📤 Extract Range", variant="warning",
            command=self._extract_range,
            tooltip="Extract a range of frames (by index) to a folder."
        ).pack(side="left", padx=3)

        StyledButton(
            action_frame, text="📤 Extract All", variant="warning",
            command=self.extract_all,
            tooltip="Extract all frames from the archive to a folder. Uses GPU for interpolated frames."
        ).pack(side="left", padx=3)

        StyledButton(
            action_frame, text="📤 Export KFs Only", variant="info",
            command=self._export_keyframes_only,
            tooltip="Export only keyframes (no interpolated frames). Fast, no GPU needed."
        ).pack(side="left", padx=3)

        ttk.Separator(action_frame, orient="vertical").pack(side="left", fill="y", padx=10)

        StyledButton(
            action_frame, text="🗑 Delete Selected", variant="danger",
            command=self._delete_selected,
            tooltip="Mark selected frame(s) as deleted. Can be restored later."
        ).pack(side="left", padx=3)

        StyledButton(
            action_frame, text="♻ Restore", variant="success",
            command=self._restore_selected,
            tooltip="Restore a previously deleted frame."
        ).pack(side="left", padx=3)

        StyledButton(
            action_frame, text="✅ Verify", variant="info",
            command=self.verify_archive,
            tooltip="Verify archive integrity: check for corrupted or missing data."
        ).pack(side="left", padx=3)

        StyledButton(
            action_frame, text="📦 Compact Archive", variant="primary",
            command=self._compact_archive,
            tooltip="Remove deleted frames and rebuild a clean, smaller archive.\n"
                    "Creates a backup before modifying. Reduces file size."
        ).pack(side="left", padx=3)

        StyledButton(
            action_frame, text="🔧 Repair Archive", variant="danger",
            command=self._repair_archive,
            tooltip="Scan for corrupted data, broken references, and orphaned frames.\n"
                    "Automatically fixes what it can and reports unrecoverable issues."
        ).pack(side="left", padx=3)

    # ------------------------------------------------------------------
    # Archive loading
    # ------------------------------------------------------------------

    def open_archive(self, path: str):
        """Open an archive file and populate the viewer."""
        self.archive_var.set(path)
        self.app.set_status(f"Opening archive: {path}")

        def open_task(progress: EnhancedProgressDialog):
            progress.update_progress("Opening archive...", 0, 1)
            engine = self.app.archive_engine
            header, frames, gops = engine.open_archive(path)
            return header, frames, gops

        def on_complete(result):
            header, frames, gops = result
            self._header = header
            self._frames = frames
            self._gops = gops
            self._decode_cache.clear()

            # Populate frame table
            self.viewer_frame_table.set_frames(frames)

            # Update archive info
            info = (
                f"Frames:    {header.total_frames:,}\n"
                f"Keyframes: {header.keyframe_count + header.forced_keyframe_count:,}\n"
                f"Interp:    {header.interpolated_count:,}\n"
                f"Residual:  {header.residual_count:,}\n"
                f"GOPs:      {header.gop_count:,}\n"
                f"Codec:     {header.compression_codec}\n"
                f"Quality:   {header.compression_quality}\n"
                f"Size:      {human_readable_size(header.archive_total_bytes)}\n"
                f"Resolution:{header.original_width}x{header.original_height}\n"
                f"Created:   {header.created_timestamp}\n"
            )

            self.archive_info_text.config(state="normal")
            self.archive_info_text.delete("1.0", "end")
            self.archive_info_text.insert("1.0", info)
            self.archive_info_text.config(state="disabled")

            # Track in recent files
            self.app.recent_manager.add(path)
            self.app._update_recent_menu()

            # Auto-load .ias project if present
            ias = find_ias_for_iarc(path)
            if ias:
                self.app.project = ProjectFile.load(ias)
                self.app.project_path = ias

            self.app.set_status(f"Archive loaded: {header.total_frames:,} frames")
            self.app.archive_info_var.set(
                f"Archive: {os.path.basename(path)} | "
                f"{header.total_frames:,} frames | "
                f"{human_readable_size(header.archive_total_bytes)}"
            )

        def on_error(err):
            messagebox.showerror("Open Error", str(err))

        task = ThreadedTask(
            self.app, open_task,
            on_complete=on_complete, on_error=on_error,
            title="Opening Archive"
        )
        task.start()

    def _browse_archive(self):
        path = filedialog.askopenfilename(
            title="Open Archive",
            filetypes=[("ImgArchive Files", "*.iarc"), ("All Files", "*.*")]
        )
        if path:
            self.open_archive(path)

    def _reload_archive(self):
        path = self.archive_var.get().strip()
        if path and os.path.exists(path):
            self.open_archive(path)

    # ------------------------------------------------------------------
    # Frame selection, decoding, navigation
    # ------------------------------------------------------------------

    def _on_frame_select(self, index: int):
        """Handle frame selection — show detail info."""
        if index < 0 or index >= len(self._frames):
            return

        f = self._frames[index]
        type_name = FRAME_TYPE_NAMES.get(f.frame_type, "Unknown")
        total_size = f.data_size + f.residual_size

        detail = (
            f"Frame:       {f.index}\n"
            f"Name:        {f.name}\n"
            f"Type:        [{FRAME_TYPE_LABELS.get(f.frame_type, '?')}] {type_name}\n"
            f"Size:        {human_readable_size(total_size)}\n"
            f"GOP:         {f.gop_id}\n"
            f"Similarity:  {f.similarity_score:.4f}\n"
            f"Face Score:  {f.face_score:.4f}\n"
            f"Body Score:  {f.body_score:.4f}\n"
            f"Motion:      {f.motion_score:.4f}\n"
            f"Scene Cut:   {'Yes' if f.scene_cut else 'No'}\n"
            f"Parents:     K({f.parent_keyframe_a}) → K({f.parent_keyframe_b})\n"
            f"Timestep:    {f.interpolation_timestep:.4f}\n"
            f"Deleted:     {'Yes' if f.is_deleted else 'No'}\n"
        )

        self.frame_info_text.config(state="normal")
        self.frame_info_text.delete("1.0", "end")
        self.frame_info_text.insert("1.0", detail)
        self.frame_info_text.config(state="disabled")

        # Show cached decode if available
        if index in self._decode_cache:
            self.preview_canvas.set_image(self._decode_cache[index])
            self.preview_canvas.after(50, self.preview_canvas.fit)

        # Update frame counter in toolbar
        total = len(self._frames)
        self.app.frame_counter_var.set(
            f"Frame: {index + 1:,} / {total:,}"
        )

    def _on_frame_decode(self, index: int):
        """Handle double-click: decode and display frame."""
        self._decode_frame(index)

    def _decode_current(self):
        """Decode the currently selected frame."""
        index = self.viewer_frame_table.get_selected_index()
        if index >= 0:
            self._decode_frame(index)

    def _decode_frame(self, index: int):
        """Decode a frame in a background thread and display it."""
        if index < 0 or index >= len(self._frames):
            return

        # Check cache first
        if index in self._decode_cache:
            self.preview_canvas.set_image(self._decode_cache[index])
            self.preview_canvas.after(50, self.preview_canvas.fit)
            self.decode_time_var.set("(cached)")
            return

        self.app.set_status(f"Decoding frame {index}...")
        self.decode_time_var.set("Decoding...")

        def decode_task(progress: EnhancedProgressDialog):
            progress.update_progress(f"Decoding frame {index}...", 0, 1)
            start = time.time()
            img = self.app.archive_engine.extract_frame(index)

            # Apply optional post-processing
            if self.app.settings.gfpgan_mode in ("Preview", "Both"):
                img = self.app.model_manager.enhance_face(img)

            elapsed = time.time() - start
            return img, elapsed

        def on_complete(result):
            img, elapsed = result

            # Cache management
            self._decode_cache[index] = img
            if len(self._decode_cache) > self._cache_max:
                self._decode_cache.popitem(last=False)

            self.preview_canvas.set_image(img)
            self.preview_canvas.after(50, self.preview_canvas.fit)
            self.decode_time_var.set(f"Decoded in {elapsed * 1000:.0f} ms")
            self.app.set_status(f"Frame {index} decoded in {elapsed * 1000:.0f} ms")

        def on_error(err):
            self.decode_time_var.set("Decode failed")
            messagebox.showerror("Decode Error", str(err))

        task = ThreadedTask(
            self.app, decode_task,
            on_complete=on_complete, on_error=on_error,
            title=f"Decoding Frame {index}"
        )
        task.start()

    # def _prev_frame(self):
    #     idx = self.viewer_frame_table.get_selected_index()
    #     if idx > 0:
    #         self.viewer_frame_table.select_index(idx - 1)
    #         self._on_frame_select(idx - 1)
    #
    # def _next_frame(self):
    #     idx = self.viewer_frame_table.get_selected_index()
    #     if idx < len(self._frames) - 1:
    #         self.viewer_frame_table.select_index(idx + 1)
    #         self._on_frame_select(idx + 1)

    def _prev_frame(self):
        idx = self.viewer_frame_table.get_selected_index()
        if idx > 0:
            new_idx = idx - 1
            self.viewer_frame_table.select_index(new_idx)
            self._on_frame_select(new_idx)
            self._decode_frame(new_idx)

    def _next_frame(self):
        idx = self.viewer_frame_table.get_selected_index()
        if idx < len(self._frames) - 1:
            new_idx = idx + 1
            self.viewer_frame_table.select_index(new_idx)
            self._on_frame_select(new_idx)
            self._decode_frame(new_idx)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_selected(self):
        """Extract selected frame(s) to disk."""
        indices = self.viewer_frame_table.get_selected_indices()
        if not indices:
            messagebox.showinfo("Extract", "Select one or more frames first.")
            return

        if len(indices) == 1:
            path = filedialog.asksaveasfilename(
                title="Save Frame As",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("WebP", "*.webp"), ("JPEG", "*.jpg")]
            )
            if path:
                try:
                    self.app.archive_engine.extract_frame_to_file(indices[0], path)
                    messagebox.showinfo("Extracted", f"Frame saved to:\n{path}")
                except Exception as e:
                    messagebox.showerror("Error", str(e))
        else:
            folder = filedialog.askdirectory(title="Select Output Folder")
            if folder:
                self._extract_indices(indices, folder)

    def _extract_range(self):
        """Extract a range of frames."""
        if not self._frames:
            return

        dialog = tk.Toplevel(self.app)
        dialog.title("Extract Range")
        dialog.transient(self.app)
        dialog.configure(bg=COLOR_BG)
        center_window(dialog, self.app, 300, 180)
        # dialog.configure(bg=COLOR_BG)

        StyledLabel(dialog, text="Start Index:", style="body").pack(pady=(10, 2))
        start_var = tk.IntVar(value=0)
        StyledEntry(dialog, textvariable=start_var, width=10).pack()

        StyledLabel(dialog, text="End Index:", style="body").pack(pady=(10, 2))
        end_var = tk.IntVar(value=len(self._frames) - 1)
        StyledEntry(dialog, textvariable=end_var, width=10).pack()

        def do_extract():
            dialog.destroy()
            folder = filedialog.askdirectory(title="Select Output Folder")
            if folder:
                indices = list(range(start_var.get(), end_var.get() + 1))
                self._extract_indices(indices, folder)

        StyledButton(dialog, text="Extract", variant="success", command=do_extract).pack(pady=15)

    def extract_all(self):
        """Extract all frames."""
        if not self._frames:
            messagebox.showinfo("Extract All", "No archive loaded.")
            return

        folder = filedialog.askdirectory(title="Select Output Folder for All Frames")
        if folder:
            indices = list(range(len(self._frames)))
            self._extract_indices(indices, folder)

    def _export_keyframes_only(self):
        """Export only keyframes (fast, no GPU needed)."""
        if not self._frames:
            return

        folder = filedialog.askdirectory(title="Select Output Folder for Keyframes")
        if not folder:
            return

        indices = [f.index for f in self._frames
                   if f.frame_type in (FrameType.KEYFRAME, FrameType.FORCED_KEYFRAME)
                   and not f.is_deleted]
        self._extract_indices(indices, folder)

    def _extract_indices(self, indices: List[int], output_folder: str):
        """Extract a list of frame indices to a folder — threaded."""
        total = len(indices)

        def extract_task(progress: EnhancedProgressDialog):
            extracted = 0
            errors = 0

            for pos, idx in enumerate(indices):
                if progress.is_cancelled():
                    break

                f = self._frames[idx]
                if f.is_deleted:
                    continue

                out_path = os.path.join(output_folder, f"{f.name}")
                if not out_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    out_path += ".png"

                try:
                    self.app.archive_engine.extract_frame_to_file(idx, out_path)
                    extracted += 1
                except Exception:
                    errors += 1

                progress.update_progress(
                    "Extracting frames...", pos + 1, total,
                    detail=f"Frame {f.name} → {out_path}"
                )

            return extracted, errors

        def on_complete(result):
            extracted, errors = result
            msg = f"Extracted {extracted:,} frames to:\n{output_folder}"
            if errors > 0:
                msg += f"\n\n{errors} frames had errors."
            messagebox.showinfo("Extraction Complete", msg)
            self.app.set_status(f"Extracted {extracted:,} frames")

        task = ThreadedTask(
            self.app, extract_task,
            on_complete=on_complete,
            title="Extracting Frames"
        )
        task.start()

    # ------------------------------------------------------------------
    # Delete / Restore
    # ------------------------------------------------------------------

    def _delete_selected(self):
        """Mark selected frames as deleted."""
        indices = self.viewer_frame_table.get_selected_indices()
        if not indices:
            return

        if messagebox.askyesno("Confirm Delete",
                               f"Mark {len(indices)} frame(s) as deleted?"):
            for idx in indices:
                self.app.archive_engine.mark_deleted(idx)
            self.app.archive_engine.save_index_changes()
            self.viewer_frame_table.refresh()
            self.app.set_status(f"Deleted {len(indices)} frame(s)")

    def _restore_selected(self):
        """Restore deleted frames."""
        indices = self.viewer_frame_table.get_selected_indices()
        if not indices:
            return

        for idx in indices:
            self.app.archive_engine.restore_frame(idx, FrameType.KEYFRAME)
        self.app.archive_engine.save_index_changes()
        self.viewer_frame_table.refresh()
        self.app.set_status(f"Restored {len(indices)} frame(s)")

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def verify_archive(self):
        """Verify archive integrity."""
        if not self._frames:
            messagebox.showinfo("Verify", "No archive loaded.")
            return

        results = self.app.archive_engine.verify_integrity()
        if results["valid"]:
            messagebox.showinfo("Integrity Check",
                                f"Archive is valid.\n"
                                f"Total frames: {results['total_frames']:,}")
        else:
            messagebox.showwarning("Integrity Check",
                                   f"Issues found!\n"
                                   f"Corrupted: {len(results['corrupted_frames'])}\n"
                                   f"Missing data: {len(results['missing_data'])}")


    # ----------------------------------------------------------------------
    # Compact and Repair
    # ----------------------------------------------------------------------

    def _compact_archive(self):
        """Compact the current archive — remove deleted, rebuild clean."""
        if not self._frames:
            messagebox.showinfo("Compact", "No archive loaded.")
            return

        deleted_count = sum(1 for f in self._frames if f.is_deleted)
        if deleted_count == 0:
            messagebox.showinfo(
                "Compact",
                "No deleted frames found. Archive is already clean."
            )
            return

        if not messagebox.askyesno(
            "Confirm Compact",
            f"This will permanently remove {deleted_count} deleted frame(s) "
            f"and rebuild the archive.\n\n"
            f"A backup will be created before modifying.\n\n"
            f"Continue?"
        ):
            return

        self.app.set_status("Compacting archive...")

        def compact_task(progress: EnhancedProgressDialog):
            def cb(msg, current, total):
                progress.update_progress(msg, current, total)

            result_path = self.app.archive_engine.compact_archive(
                output_path=None,
                progress_callback=cb
            )
            return result_path

        def on_complete(result_path):
            # Reload the archive
            self._header = self.app.archive_engine.get_header()
            self._frames = self.app.archive_engine.list_frames()
            self._gops = self.app.archive_engine.get_gops()
            self._decode_cache.clear()

            self.viewer_frame_table.set_frames(self._frames)
            self._update_archive_info()

            new_size = human_readable_size(
                self._header.archive_total_bytes
            )
            messagebox.showinfo(
                "Compact Complete",
                f"Archive compacted successfully.\n\n"
                f"Deleted frames removed: {deleted_count}\n"
                f"New size: {new_size}\n"
                f"Path: {result_path}"
            )
            self.app.set_status(
                f"Compact complete. New size: {new_size}"
            )

        def on_error(err):
            messagebox.showerror("Compact Error", str(err))
            self.app.set_status("Compact failed")

        task = ThreadedTask(
            self.app, compact_task,
            on_complete=on_complete, on_error=on_error,
            title="Compacting Archive"
        )
        task.start()

    def _repair_archive(self):
        """Repair the current archive — fix corrupted and orphaned frames."""
        if not self._frames:
            messagebox.showinfo("Repair", "No archive loaded.")
            return

        if not messagebox.askyesno(
            "Confirm Repair",
            "This will scan the entire archive for:\n"
            "• Corrupted keyframe data\n"
            "• Corrupted residual patches\n"
            "• Orphaned interpolated frames\n"
            "• Broken GOP chains\n\n"
            "Fixes will be applied automatically where possible.\n"
            "A backup is created before any modifications.\n\n"
            "Continue?"
        ):
            return

        self.app.set_status("Repairing archive...")

        def repair_task(progress: EnhancedProgressDialog):
            def cb(msg, current, total):
                progress.update_progress(msg, current, total)

            report = self.app.archive_engine.repair_archive(
                output_path=None,
                progress_callback=cb
            )
            return report

        def on_complete(report):
            # Reload archive after repair
            self._header = self.app.archive_engine.get_header()
            self._frames = self.app.archive_engine.list_frames()
            self._gops = self.app.archive_engine.get_gops()
            self._decode_cache.clear()

            self.viewer_frame_table.set_frames(self._frames)
            self._update_archive_info()

            # Build report message
            msg = (
                f"Repair Results\n"
                f"{'=' * 40}\n"
                f"Total frames scanned:    {report['total_frames']}\n"
                f"Corrupted keyframes:     "
                f"{len(report['corrupted_keyframes'])}\n"
                f"Corrupted residuals:     "
                f"{len(report['corrupted_residuals'])}\n"
                f"Orphaned frames:         "
                f"{len(report['orphaned_frames'])}\n"
                f"{'=' * 40}\n"
                f"Frames fixed:            "
                f"{len(report['fixed_frames'])}\n"
                f"Promoted to keyframe:    "
                f"{len(report['promoted_to_keyframe'])}\n"
                f"Demoted to interpolated: "
                f"{len(report['demoted_to_interpolated'])}\n"
                f"Unrecoverable:           "
                f"{len(report['unrecoverable'])}\n"
                f"{'=' * 40}\n"
                f"Archive rebuilt:          "
                f"{'Yes' if report['repaired'] else 'No'}\n"
            )

            if report["unrecoverable"]:
                msg += (
                    f"\nUnrecoverable frame indices:\n"
                    f"{report['unrecoverable'][:20]}"
                )
                if len(report["unrecoverable"]) > 20:
                    msg += f"\n... and {len(report['unrecoverable']) - 20} more"

            if not report["corrupted_keyframes"] and \
               not report["corrupted_residuals"] and \
               not report["orphaned_frames"]:
                messagebox.showinfo("Repair Complete", "No issues found!\n\nArchive is healthy.")
            elif report["repaired"]:
                messagebox.showinfo("Repair Complete", msg)
            else:
                messagebox.showwarning("Repair Report", msg)

            self.app.set_status("Repair complete")

        def on_error(err):
            messagebox.showerror("Repair Error", str(err))
            self.app.set_status("Repair failed")

        task = ThreadedTask(
            self.app, repair_task,
            on_complete=on_complete, on_error=on_error,
            title="Repairing Archive"
        )
        task.start()

    def _update_archive_info(self):
        """Refresh the archive info panel after compact/repair."""
        if not self._header:
            return
        header = self._header
        info = (
            f"Frames:    {header.total_frames:,}\n"
            f"Keyframes: {header.keyframe_count + header.forced_keyframe_count:,}\n"
            f"Interp:    {header.interpolated_count:,}\n"
            f"Residual:  {header.residual_count:,}\n"
            f"Codec:     {header.compression_codec}\n"
            f"Quality:   {header.compression_quality}\n"
            f"Size:      {human_readable_size(header.archive_total_bytes)}\n"
            f"Resolution:{header.original_width}x{header.original_height}\n"
            f"Created:   {header.created_timestamp}\n"
        )
        self.archive_info_text.config(state="normal")
        self.archive_info_text.delete("1.0", "end")
        self.archive_info_text.insert("1.0", info)
        self.archive_info_text.config(state="disabled")

# ============================================================================
# SETTINGS TAB — Models, Device, Archive Defaults, Feature Toggles
# ============================================================================

class SettingsTab(StyledFrame):
    """
    Models / Settings Tab.
    Configure all model paths, device preferences, archive defaults,
    feature toggles, and safety thresholds.
    """

    def __init__(self, parent, app: 'ImgArchiveStudioApp', **kwargs):
        super().__init__(parent, **kwargs)
        self.app = app
        self._build_ui()
        self._load_from_settings()

    def _build_ui(self):
        # Main scrollable area (settings can be long)
        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True, padx=10, pady=5)
        container = scroll.scrollable_frame

        # ================================================================
        # SECTION 1: RIFE / Optical Flow Models
        # ================================================================
        rife_frame = StyledLabelFrame(
            container, text="🧠 RIFE / Optical Flow Models",
            tooltip="Configure paths to RIFE interpolation and RAFT optical flow models."
        )
        rife_frame.pack(fill="x", pady=5, padx=5)

        # RIFE model dir
        rife_row = StyledFrame(rife_frame)
        rife_row.pack(fill="x", pady=3)
        StyledLabel(rife_row, text="RIFE Model Directory:", style="body",
                    tooltip="Directory containing RIFE model files (e.g., flownet.pkl).").pack(side="left")
        self.rife_dir_var = tk.StringVar()
        StyledEntry(
            rife_row, textvariable=self.rife_dir_var, width=55,
            tooltip="Path to RIFE model directory. Must contain model files."
        ).pack(side="left", padx=5)
        StyledButton(
            rife_row, text="...", variant="secondary",
            command=lambda: self._browse_dir(self.rife_dir_var),
            tooltip="Browse for RIFE model directory."
        ).pack(side="left", padx=2)
        StyledButton(
            rife_row, text="Test", variant="info",
            command=self._test_rife,
            tooltip="Test loading the RIFE model. Verifies the directory is valid."
        ).pack(side="left", padx=2)

        # RAFT weights
        raft_row = StyledFrame(rife_frame)
        raft_row.pack(fill="x", pady=3)
        StyledLabel(raft_row, text="RAFT Model:", style="body",
                    tooltip="Select RAFT optical flow model variant.").pack(side="left")
        self.raft_var = tk.StringVar()
        self.raft_combo = StyledCombobox(
            raft_row, values=list(RAFT_WEIGHT_OPTIONS.keys()), width=15,
            tooltip="RAFT variant. 'raft-things' is recommended for general use.\n"
                    "• raft-things: Best for diverse scenes\n"
                    "• raft-sintel: Best for synthetic/animated content\n"
                    "• raft-small: Fastest, lowest quality"
        )
        self.raft_combo.pack(side="left", padx=5)
        StyledButton(
            raft_row, text="Test", variant="info",
            command=self._test_raft,
            tooltip="Test loading the selected RAFT model."
        ).pack(side="left", padx=2)

        # ================================================================
        # SECTION 2: Scene Cut Detection
        # ================================================================
        transnet_frame = StyledLabelFrame(
            container, text="🎬 Scene Cut Detection",
            tooltip="TransNetV2 model for detecting hard scene cuts in image sequences."
        )
        transnet_frame.pack(fill="x", pady=5, padx=5)

        tn_row = StyledFrame(transnet_frame)
        tn_row.pack(fill="x", pady=3)
        StyledLabel(tn_row, text="TransNetV2 Weights:", style="body",
                    tooltip="Path to TransNetV2 PyTorch weights file.").pack(side="left")
        self.transnet_var = tk.StringVar()
        StyledEntry(
            tn_row, textvariable=self.transnet_var, width=55,
            tooltip="TransNetV2 model weights (.pth file)."
        ).pack(side="left", padx=5)
        StyledButton(
            tn_row, text="...", variant="secondary",
            command=lambda: self._browse_file(self.transnet_var, [("PyTorch weights", "*.pth")]),
            tooltip="Browse for TransNetV2 weights file."
        ).pack(side="left", padx=2)

        sc_sens_row = StyledFrame(transnet_frame)
        sc_sens_row.pack(fill="x", pady=3)
        StyledLabel(sc_sens_row, text="Scene Cut Sensitivity:", style="body",
                    tooltip="Higher = more sensitive, detects subtle cuts. Lower = only hard cuts.").pack(side="left")
        self.scene_cut_sens_var = tk.DoubleVar()
        StyledScale(
            sc_sens_row, from_=0.1, to=0.9, orient="horizontal",
            variable=self.scene_cut_sens_var, resolution=0.05,
            tooltip="Scene cut detection sensitivity (0.1-0.9).\n"
                    "0.1 = only very obvious cuts\n"
                    "0.5 = balanced\n"
                    "0.9 = catches subtle transitions"
        ).pack(side="left", fill="x", expand=True, padx=5)

        # ================================================================
        # SECTION 3: Face Analysis Models
        # ================================================================
        face_frame = StyledLabelFrame(
            container, text="👤 Face Analysis Models",
            tooltip="Models for face detection, parsing, and identity consistency."
        )
        face_frame.pack(fill="x", pady=5, padx=5)

        # Face parse (BiSeNet)
        fp_row = StyledFrame(face_frame)
        fp_row.pack(fill="x", pady=3)
        StyledLabel(fp_row, text="Face Parsing (BiSeNet):", style="body",
                    tooltip="BiSeNet face parsing model for segmenting face regions.").pack(side="left")
        self.face_parse_var = tk.StringVar()
        StyledEntry(
            fp_row, textvariable=self.face_parse_var, width=55,
            tooltip="Path to 79999_iter.pth — BiSeNet face parsing weights."
        ).pack(side="left", padx=5)
        StyledButton(
            fp_row, text="...", variant="secondary",
            command=lambda: self._browse_file(self.face_parse_var, [("PyTorch weights", "*.pth")]),
            tooltip="Browse for BiSeNet face parsing weights."
        ).pack(side="left", padx=2)

        # ArcFace identity
        af_row = StyledFrame(face_frame)
        af_row.pack(fill="x", pady=3)
        StyledLabel(af_row, text="ArcFace Identity (w600k):", style="body",
                    tooltip="ArcFace model for face identity embedding and consistency checks.").pack(side="left")
        self.arcface_var = tk.StringVar()
        StyledEntry(
            af_row, textvariable=self.arcface_var, width=55,
            tooltip="Path to w600k_r50.onnx — ArcFace identity model."
        ).pack(side="left", padx=5)
        StyledButton(
            af_row, text="...", variant="secondary",
            command=lambda: self._browse_file(self.arcface_var, [("ONNX", "*.onnx")]),
            tooltip="Browse for ArcFace ONNX model."
        ).pack(side="left", padx=2)

        # Face similarity threshold
        fst_row = StyledFrame(face_frame)
        fst_row.pack(fill="x", pady=3)
        StyledLabel(fst_row, text="Face Similarity Threshold:", style="body",
                    tooltip="Minimum face-region SSIM. Frames below this are forced as keyframes.").pack(side="left")
        self.face_thresh_var = tk.DoubleVar()
        StyledScale(
            fst_row, from_=0.85, to=0.99, orient="horizontal",
            variable=self.face_thresh_var, resolution=0.01,
            tooltip="Face similarity threshold (0.85-0.99).\n"
                    "Higher = stricter face protection, more keyframes.\n"
                    "Lower = more compression, slight face risk."
        ).pack(side="left", fill="x", expand=True, padx=5)

        # Identity mismatch threshold
        idt_row = StyledFrame(face_frame)
        idt_row.pack(fill="x", pady=3)
        StyledLabel(idt_row, text="Identity Mismatch Threshold:", style="body",
                    tooltip="Below this cosine similarity, frames are considered different people.").pack(side="left")
        self.identity_thresh_var = tk.DoubleVar()
        StyledScale(
            idt_row, from_=0.3, to=0.9, orient="horizontal",
            variable=self.identity_thresh_var, resolution=0.05,
            tooltip="Identity consistency threshold.\n"
                    "0.6 = default, catches most person changes.\n"
                    "Lower = more tolerant.\n"
                    "Higher = stricter."
        ).pack(side="left", fill="x", expand=True, padx=5)

        # GFPGAN
        gf_row = StyledFrame(face_frame)
        gf_row.pack(fill="x", pady=3)
        StyledLabel(gf_row, text="GFPGAN Weights:", style="body",
                    tooltip="GFPGAN face enhancement model. Used for optional post-decode face repair.").pack(side="left")
        self.gfpgan_var = tk.StringVar()
        StyledEntry(
            gf_row, textvariable=self.gfpgan_var, width=55,
            tooltip="Path to GFPGANv1.4.pth."
        ).pack(side="left", padx=5)
        StyledButton(
            gf_row, text="...", variant="secondary",
            command=lambda: self._browse_file(self.gfpgan_var, [("PyTorch weights", "*.pth")]),
            tooltip="Browse for GFPGAN weights."
        ).pack(side="left", padx=2)

        gf_mode_row = StyledFrame(face_frame)
        gf_mode_row.pack(fill="x", pady=3)
        StyledLabel(gf_mode_row, text="GFPGAN Mode:", style="body",
                    tooltip="When to apply GFPGAN face enhancement.").pack(side="left")
        self.gfpgan_mode_var = tk.StringVar()
        StyledCombobox(
            gf_mode_row, values=["Off", "Preview", "Extraction", "Both"],
            width=14,
            tooltip="GFPGAN application mode:\n"
                    "• Off: No face enhancement\n"
                    "• Preview: Enhance only in viewer preview\n"
                    "• Extraction: Enhance only when extracting to disk\n"
                    "• Both: Always enhance"
        ).pack(side="left", padx=5)

        # ================================================================
        # SECTION 4: Body Analysis Models
        # ================================================================
        body_frame = StyledLabelFrame(
            container, text="🏃 Body Analysis Models",
            tooltip="Models for human body segmentation and pose estimation."
        )
        body_frame.pack(fill="x", pady=5, padx=5)

        # YOLO Segmentation
        ys_row = StyledFrame(body_frame)
        ys_row.pack(fill="x", pady=3)
        StyledLabel(ys_row, text="YOLO Segmentation:", style="body",
                    tooltip="YOLO11x-seg model for human instance segmentation.").pack(side="left")
        self.yolo_seg_var = tk.StringVar()
        StyledEntry(
            ys_row, textvariable=self.yolo_seg_var, width=55,
            tooltip="Path to yolo11x-seg.onnx."
        ).pack(side="left", padx=5)
        StyledButton(
            ys_row, text="...", variant="secondary",
            command=lambda: self._browse_file(self.yolo_seg_var, [("ONNX", "*.onnx")]),
            tooltip="Browse for YOLO segmentation ONNX model."
        ).pack(side="left", padx=2)

        # YOLO Pose
        yp_row = StyledFrame(body_frame)
        yp_row.pack(fill="x", pady=3)
        StyledLabel(yp_row, text="YOLO Pose:", style="body",
                    tooltip="YOLO11x-pose model for body keypoint detection.").pack(side="left")
        self.yolo_pose_var = tk.StringVar()
        StyledEntry(
            yp_row, textvariable=self.yolo_pose_var, width=55,
            tooltip="Path to yolo11x-pose.onnx."
        ).pack(side="left", padx=5)
        StyledButton(
            yp_row, text="...", variant="secondary",
            command=lambda: self._browse_file(self.yolo_pose_var, [("ONNX", "*.onnx")]),
            tooltip="Browse for YOLO pose ONNX model."
        ).pack(side="left", padx=2)

        # DeepLab fallback
        dl_row = StyledFrame(body_frame)
        dl_row.pack(fill="x", pady=3)
        StyledLabel(dl_row, text="DeepLab Human (fallback):", style="body",
                    tooltip="DeepLabV3+ human segmentation — lightweight fallback.").pack(side="left")
        self.deeplab_var = tk.StringVar()
        StyledEntry(
            dl_row, textvariable=self.deeplab_var, width=55,
            tooltip="Path to deeplabv3p-resnet50-human.onnx."
        ).pack(side="left", padx=5)
        StyledButton(
            dl_row, text="...", variant="secondary",
            command=lambda: self._browse_file(self.deeplab_var, [("ONNX", "*.onnx")]),
            tooltip="Browse for DeepLab human segmentation ONNX model."
        ).pack(side="left", padx=2)

        # Body similarity threshold
        bst_row = StyledFrame(body_frame)
        bst_row.pack(fill="x", pady=3)
        StyledLabel(bst_row, text="Body Similarity Threshold:", style="body",
                    tooltip="Minimum body-region SSIM. Frames below this are forced as keyframes.").pack(side="left")
        self.body_thresh_var = tk.DoubleVar()
        StyledScale(
            bst_row, from_=0.80, to=0.99, orient="horizontal",
            variable=self.body_thresh_var, resolution=0.01,
            tooltip="Body similarity threshold (0.80-0.99).\n"
                    "Higher = stricter body protection.\n"
                    "Lower = more compression."
        ).pack(side="left", fill="x", expand=True, padx=5)

        # ================================================================
        # SECTION 5: Depth Analysis (Toggle)
        # ================================================================
        depth_frame = StyledLabelFrame(
            container, text="🌊 Depth Analysis (Optional)",
            tooltip="Depth Anything model for depth-guided compression. Toggle on/off."
        )
        depth_frame.pack(fill="x", pady=5, padx=5)

        self.depth_enabled_var = tk.BooleanVar()
        StyledCheckbox(
            depth_frame, text="Enable Depth-Aware Compression", variable=self.depth_enabled_var,
            tooltip="Use Depth Anything V2 to guide residual placement.\n"
                    "High-depth-change zones get stronger residuals.\n"
                    "Adds build time but can improve quality."
        ).pack(anchor="w", pady=3)

        depth_model_row = StyledFrame(depth_frame)
        depth_model_row.pack(fill="x", pady=3)
        StyledLabel(depth_model_row, text="Depth Model:", style="body",
                    tooltip="Select Depth Anything V2 model variant.").pack(side="left")
        self.depth_model_var = tk.StringVar()
        StyledCombobox(
            depth_model_row, values=list(DEPTH_WEIGHT_OPTIONS.keys()), width=10,
            tooltip="Depth model variant:\n"
                    "• vits: Fastest, smallest\n"
                    "• vitb: Balanced (recommended)\n"
                    "• vitl: Best quality, slowest"
        ).pack(side="left", padx=5)

        # ================================================================
        # SECTION 6: Upscaler Models (Toggle)
        # ================================================================
        upscale_frame = StyledLabelFrame(
            container, text="🔎 Upscaler Models (Optional)",
            tooltip="Super-resolution models for optional post-decode upscaling."
        )
        upscale_frame.pack(fill="x", pady=5, padx=5)

        self.downscale_enabled_var = tk.BooleanVar()
        StyledCheckbox(
            upscale_frame, text="Enable Archive Downscale + Upscale on Extract",
            variable=self.downscale_enabled_var,
            tooltip="Store keyframes at reduced resolution inside the archive.\n"
                    "On extraction, upscale back to original using RealESRGAN or SwinIR.\n"
                    "Can reduce archive size by 50-70% beyond RIFE alone."
        ).pack(anchor="w", pady=3)

        ds_row = StyledFrame(upscale_frame)
        ds_row.pack(fill="x", pady=3)
        StyledLabel(ds_row, text="Downscale Factor:", style="body",
                    tooltip="How much to reduce keyframe resolution.").pack(side="left")
        self.downscale_factor_var = tk.DoubleVar()
        StyledScale(
            ds_row, from_=0.25, to=0.75, orient="horizontal",
            variable=self.downscale_factor_var, resolution=0.05,
            tooltip="Downscale factor (0.25-0.75).\n"
                    "0.50 = store at 50% resolution (4x fewer pixels)\n"
                    "0.75 = store at 75% resolution (milder reduction)"
        ).pack(side="left", fill="x", expand=True, padx=5)

        up_row = StyledFrame(upscale_frame)
        up_row.pack(fill="x", pady=3)
        StyledLabel(up_row, text="Upscaler:", style="body",
                    tooltip="Select upscaler model for extraction.").pack(side="left")
        self.upscaler_var = tk.StringVar()
        self.upscaler_combo = StyledCombobox(
            up_row, values=list(UPSCALER_OPTIONS.keys()), width=20,
            tooltip="Upscaler model:\n"
                    "• Off: No upscaling\n"
                    "• RealESRGAN x2/x4/x4+/x8: General purpose upscaler\n"
                    "• SwinIR x4: Better for fine detail and faces"
        )
        self.upscaler_combo.pack(side="left", padx=5)

        # ================================================================
        # SECTION 7: Runtime / Device Settings
        # ================================================================
        device_frame = StyledLabelFrame(
            container, text="⚡ Runtime / Device",
            tooltip="GPU/CPU, precision, batch size, and cache settings."
        )
        device_frame.pack(fill="x", pady=5, padx=5)

        dev_row = StyledFrame(device_frame)
        dev_row.pack(fill="x", pady=3)
        StyledLabel(dev_row, text="Device:", style="body",
                    tooltip="Compute device for all AI operations.").pack(side="left")
        self.device_var = tk.StringVar()
        self.device_combo = StyledCombobox(
            dev_row, values=["cuda", "cpu"], width=8,
            tooltip="Device selection:\n"
                    "• cuda: GPU (default, required for RIFE)\n"
                    "• cpu: CPU only (very slow for AI ops)"
        )
        self.device_combo.pack(side="left", padx=5)

        prec_row = StyledFrame(device_frame)
        prec_row.pack(fill="x", pady=3)
        StyledLabel(prec_row, text="GPU Precision:", style="body",
                    tooltip="Floating point precision for GPU operations.").pack(side="left")
        self.precision_var = tk.StringVar()
        self.precision_combo = StyledCombobox(
            prec_row, values=["fp16", "fp32"], width=8,
            tooltip="GPU precision:\n"
                    "• fp16: Half precision (faster, less VRAM, recommended)\n"
                    "• fp32: Full precision (slower, more VRAM, slightly better quality)"
        )
        self.precision_combo.pack(side="left", padx=5)

        batch_row = StyledFrame(device_frame)
        batch_row.pack(fill="x", pady=3)
        StyledLabel(batch_row, text="Decode Batch Size:", style="body",
                    tooltip="Number of frames to decode simultaneously.").pack(side="left")
        self.batch_var = tk.IntVar()
        StyledScale(
            batch_row, from_=1, to=8, orient="horizontal",
            variable=self.batch_var, resolution=1,
            tooltip="Batch size for decode operations.\n"
                    "1 = sequential (safest)\n"
                    "Higher = faster but uses more VRAM"
        ).pack(side="left", fill="x", expand=True, padx=5)

        cache_row = StyledFrame(device_frame)
        cache_row.pack(fill="x", pady=3)
        StyledLabel(cache_row, text="Temp Cache Dir:", style="body",
                    tooltip="Directory for temporary decode cache files.").pack(side="left")
        self.cache_dir_var = tk.StringVar()
        StyledEntry(
            cache_row, textvariable=self.cache_dir_var, width=55,
            tooltip="Temporary directory for intermediate files during build/extract."
        ).pack(side="left", padx=5)
        StyledButton(
            cache_row, text="...", variant="secondary",
            command=lambda: self._browse_dir(self.cache_dir_var),
            tooltip="Browse for temp cache directory."
        ).pack(side="left", padx=2)

        # GPU info
        gpu_info_row = StyledFrame(device_frame)
        gpu_info_row.pack(fill="x", pady=3)
        self.gpu_info_var = tk.StringVar(value="GPU: checking...")
        StyledLabel(
            gpu_info_row, textvariable=self.gpu_info_var, style="mono",
            tooltip="Current GPU information and memory."
        ).pack(side="left")
        StyledButton(
            gpu_info_row, text="Refresh", variant="secondary",
            command=self._refresh_gpu_info,
            tooltip="Refresh GPU status and memory information."
        ).pack(side="left", padx=10)

        self.after(500, self._refresh_gpu_info)

        # ================================================================
        # SECTION 8: Archive Defaults
        # ================================================================
        defaults_frame = StyledLabelFrame(
            container, text="📦 Archive Defaults",
            tooltip="Default settings for new archive creation."
        )
        defaults_frame.pack(fill="x", pady=5, padx=5)

        ext_row = StyledFrame(defaults_frame)
        ext_row.pack(fill="x", pady=3)
        StyledLabel(ext_row, text="Default Extension:", style="body",
                    tooltip="File extension for new archives.").pack(side="left")
        self.ext_var = tk.StringVar()
        StyledEntry(
            ext_row, textvariable=self.ext_var, width=10,
            tooltip="Default archive file extension (usually .iarc)."
        ).pack(side="left", padx=5)

        gop_row = StyledFrame(defaults_frame)
        gop_row.pack(fill="x", pady=3)
        StyledLabel(gop_row, text="Default GOP Size:", style="body",
                    tooltip="Default Group of Pictures size for new archives.").pack(side="left")
        self.def_gop_var = tk.IntVar()
        StyledScale(
            gop_row, from_=4, to=48, orient="horizontal",
            variable=self.def_gop_var, resolution=1,
            tooltip="Default GOP size (4-48).\n"
                    "Smaller = more keyframes, better quality\n"
                    "Larger = fewer keyframes, more compression"
        ).pack(side="left", fill="x", expand=True, padx=5)

        qual_row = StyledFrame(defaults_frame)
        qual_row.pack(fill="x", pady=3)
        StyledLabel(qual_row, text="Default Quality:", style="body",
                    tooltip="Default WebP/JPEG compression quality.").pack(side="left")
        self.def_quality_var = tk.IntVar()
        StyledScale(
            qual_row, from_=50, to=100, orient="horizontal",
            variable=self.def_quality_var, resolution=1,
            tooltip="Default compression quality (50-100).\n"
                    "92 = recommended balance\n"
                    "Higher = better quality, larger file"
        ).pack(side="left", fill="x", expand=True, padx=5)

        # ================================================================
        # SECTION 9: Safety / Validation
        # ================================================================
        safety_frame = StyledLabelFrame(
            container, text="🛡 Safety / Validation",
            tooltip="Pre-build validation and safety checks."
        )
        safety_frame.pack(fill="x", pady=5, padx=5)

        self.verify_dims_var = tk.BooleanVar()
        StyledCheckbox(
            safety_frame, text="Verify frame dimensions match",
            variable=self.verify_dims_var,
            tooltip="Ensure all source images have the same resolution before building.\n"
                    "Mismatched frames will be rejected."
        ).pack(anchor="w", pady=1)

        self.detect_cuts_var = tk.BooleanVar()
        StyledCheckbox(
            safety_frame, text="Detect scene cuts",
            variable=self.detect_cuts_var,
            tooltip="Automatically detect hard scene cuts using histogram analysis.\n"
                    "Scene cuts force keyframe boundaries."
        ).pack(anchor="w", pady=1)

        self.reject_corrupt_var = tk.BooleanVar()
        StyledCheckbox(
            safety_frame, text="Reject corrupted source images",
            variable=self.reject_corrupt_var,
            tooltip="Skip images that cannot be loaded or decoded.\n"
                    "Prevents build failures from damaged files."
        ).pack(anchor="w", pady=1)

        self.save_manifest_var = tk.BooleanVar()
        StyledCheckbox(
            safety_frame, text="Save build manifest inside archive",
            variable=self.save_manifest_var,
            tooltip="Embed a JSON manifest with build settings and model info\n"
                    "inside the archive for future reference."
        ).pack(anchor="w", pady=1)

        # ================================================================
        # SECTION 10: Action buttons
        # ================================================================
        btn_frame = StyledFrame(container)
        btn_frame.pack(fill="x", pady=15, padx=5)

        StyledButton(
            btn_frame, text="💾 Save Settings", variant="success",
            command=self.save_settings,
            tooltip="Save all current settings to disk. Settings persist across sessions."
        ).pack(side="left", padx=5)

        StyledButton(
            btn_frame, text="🔄 Restore Defaults", variant="warning",
            command=self._restore_defaults,
            tooltip="Reset all settings to factory defaults. Requires save to persist."
        ).pack(side="left", padx=5)

        StyledButton(
            btn_frame, text="🧪 Test RIFE", variant="info",
            command=self._test_rife,
            tooltip="Load and test the RIFE interpolation model."
        ).pack(side="left", padx=5)

        StyledButton(
            btn_frame, text="🧪 Test RAFT", variant="info",
            command=self._test_raft,
            tooltip="Load and test the RAFT optical flow model."
        ).pack(side="left", padx=5)

        StyledButton(
            btn_frame, text="🧹 Unload All Models", variant="danger",
            command=self._unload_models,
            tooltip="Release all loaded AI models from GPU memory.\n"
                    "Frees VRAM. Models will be reloaded on next use."
        ).pack(side="left", padx=5)

    # ------------------------------------------------------------------
    # Load / Save settings ↔ UI
    # ------------------------------------------------------------------

    def _load_from_settings(self):
        """Populate UI controls from AppSettings."""
        s = self.app.settings

        self.rife_dir_var.set(s.rife_model_dir)
        self.raft_var.set(s.raft_model_name)
        if hasattr(self, 'raft_combo'):
            self.raft_combo.set(s.raft_model_name)

        self.transnet_var.set(s.transnet_weights)
        self.scene_cut_sens_var.set(s.scene_cut_sensitivity)

        self.face_parse_var.set(s.face_parse_weights)
        self.arcface_var.set(s.arcface_weights)
        self.face_thresh_var.set(s.face_similarity_threshold)
        self.identity_thresh_var.set(s.identity_mismatch_threshold)
        self.gfpgan_var.set(s.gfpgan_weights)
        self.gfpgan_mode_var.set(s.gfpgan_mode)

        self.yolo_seg_var.set(s.yolo_seg_weights)
        self.yolo_pose_var.set(s.yolo_pose_weights)
        self.deeplab_var.set(s.deeplab_weights)
        self.body_thresh_var.set(s.body_similarity_threshold)

        self.depth_enabled_var.set(s.use_depth_aware)
        self.depth_model_var.set(s.depth_model_name)

        self.downscale_enabled_var.set(s.use_archive_downscale)
        self.downscale_factor_var.set(s.downscale_factor)
        self.upscaler_var.set(s.upscaler)
        if hasattr(self, 'upscaler_combo'):
            self.upscaler_combo.set(s.upscaler)

        self.device_var.set(s.device)
        if hasattr(self, 'device_combo'):
            self.device_combo.set(s.device)
        self.precision_var.set(s.precision)
        if hasattr(self, 'precision_combo'):
            self.precision_combo.set(s.precision)
        self.batch_var.set(s.decode_batch_size)
        self.cache_dir_var.set(s.temp_cache_dir)

        self.ext_var.set(s.default_extension)
        self.def_gop_var.set(s.default_gop_size)
        self.def_quality_var.set(s.default_quality)

        self.verify_dims_var.set(s.verify_dimensions)
        self.detect_cuts_var.set(s.detect_scene_cuts)
        self.reject_corrupt_var.set(s.reject_corrupted)
        self.save_manifest_var.set(s.save_build_manifest)

    def save_settings(self):
        """Save UI controls to AppSettings and persist to disk."""
        s = self.app.settings

        s.rife_model_dir = self.rife_dir_var.get()
        s.raft_model_name = self.raft_var.get() or self.raft_combo.get()
        s.raft_weights = RAFT_WEIGHT_OPTIONS.get(s.raft_model_name, s.raft_weights)

        s.transnet_weights = self.transnet_var.get()
        s.scene_cut_sensitivity = self.scene_cut_sens_var.get()

        s.face_parse_weights = self.face_parse_var.get()
        s.arcface_weights = self.arcface_var.get()
        s.face_similarity_threshold = self.face_thresh_var.get()
        s.identity_mismatch_threshold = self.identity_thresh_var.get()
        s.gfpgan_weights = self.gfpgan_var.get()
        s.gfpgan_mode = self.gfpgan_mode_var.get()

        s.yolo_seg_weights = self.yolo_seg_var.get()
        s.yolo_pose_weights = self.yolo_pose_var.get()
        s.deeplab_weights = self.deeplab_var.get()
        s.body_similarity_threshold = self.body_thresh_var.get()

        s.use_depth_aware = self.depth_enabled_var.get()
        s.depth_model_name = self.depth_model_var.get()
        s.depth_weights = DEPTH_WEIGHT_OPTIONS.get(s.depth_model_name, s.depth_weights)

        s.use_archive_downscale = self.downscale_enabled_var.get()
        s.downscale_factor = self.downscale_factor_var.get()
        s.upscaler = self.upscaler_var.get() or self.upscaler_combo.get()
        up_path = UPSCALER_OPTIONS.get(s.upscaler)
        if up_path:
            s.realesrgan_weights = up_path

        s.device = self.device_var.get() or self.device_combo.get()
        s.precision = self.precision_var.get() or self.precision_combo.get()
        s.decode_batch_size = self.batch_var.get()
        s.temp_cache_dir = self.cache_dir_var.get()

        s.default_extension = self.ext_var.get()
        s.default_gop_size = self.def_gop_var.get()
        s.default_quality = self.def_quality_var.get()

        s.verify_dimensions = self.verify_dims_var.get()
        s.detect_scene_cuts = self.detect_cuts_var.get()
        s.reject_corrupted = self.reject_corrupt_var.get()
        s.save_build_manifest = self.save_manifest_var.get()

        s.save(SETTINGS_FILE)

        # Update model manager device
        self.app.model_manager.device = get_device(s)
        self.app.model_manager.use_fp16 = s.precision == "fp16"
        self.app.model_manager.settings = s
        self.app.archive_engine.settings = s

        self.app.set_status("Settings saved successfully")
        messagebox.showinfo("Settings", "Settings saved successfully.")

    def _restore_defaults(self):
        """Restore all settings to defaults."""
        if messagebox.askyesno("Restore Defaults",
                               "Reset all settings to factory defaults?\n"
                               "You must click 'Save Settings' to persist."):
            self.app.settings = AppSettings()
            self._load_from_settings()
            self.app.set_status("Settings restored to defaults (not saved yet)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _browse_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory(title="Select Directory")
        if path:
            var.set(path)

    def _browse_file(self, var: tk.StringVar, filetypes: list = None):
        if filetypes is None:
            filetypes = [("All Files", "*.*")]
        path = filedialog.askopenfilename(title="Select File", filetypes=filetypes)
        if path:
            var.set(path)

    def _test_rife(self):
        self.app.set_status("Testing RIFE model...")
        try:
            self.save_settings()
            self.app.model_manager.unload("rife")
            self.app.model_manager.get_rife()
            messagebox.showinfo("RIFE Test", "✅ RIFE model loaded successfully!")
            self.app.set_status("RIFE model OK")
        except Exception as e:
            messagebox.showerror("RIFE Test Failed", f"❌ {str(e)}")
            self.app.set_status("RIFE test failed")

    def _test_raft(self):
        self.app.set_status("Testing RAFT model...")
        try:
            self.save_settings()
            self.app.model_manager.unload("raft")
            self.app.model_manager.get_raft()
            messagebox.showinfo("RAFT Test", "✅ RAFT model loaded successfully!")
            self.app.set_status("RAFT model OK")
        except Exception as e:
            messagebox.showerror("RAFT Test Failed", f"❌ {str(e)}")
            self.app.set_status("RAFT test failed")

    def _unload_models(self):
        self.app.model_manager.unload_all()
        self._refresh_gpu_info()
        self.app.set_status("All models unloaded from GPU")
        messagebox.showinfo("Models", "All AI models unloaded from GPU memory.")

    def _refresh_gpu_info(self):
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                total = props.total_memory / (1024 ** 3)
                alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
                cached = torch.cuda.memory_reserved(0) / (1024 ** 3)
                free = total - alloc
                self.gpu_info_var.set(
                    f"🟢 {name} | Total: {total:.1f}GB | "
                    f"Allocated: {alloc:.1f}GB | "
                    f"Cached: {cached:.1f}GB | "
                    f"Free: {free:.1f}GB"
                )
            else:
                self.gpu_info_var.set("🔴 CUDA not available. CPU mode only.")
        except Exception as e:
            self.gpu_info_var.set(f"🔴 GPU info error: {e}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Launch ImgArchive Studio."""

    # Pre-flight checks
    print(f"Starting {APP_NAME} v{APP_VERSION}")
    print(f"Python {sys.version}")

    try:
        import torch
        print(f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"VRAM: {vram:.1f} GB")
        else:
            print("WARNING: CUDA not available. AI operations will be very slow.")
    except ImportError:
        print("WARNING: PyTorch not installed. AI features will not work.")

    print(f"OpenCV {cv2.__version__}")
    print(f"NumPy {np.__version__}")
    print()

    # Launch application
    app = ImgArchiveStudioApp()
    app.mainloop()


if __name__ == "__main__":
    main()


# ============================================================================
# .iarc FILE FORMAT SPECIFICATION (REFERENCE)
# ============================================================================
#
# ImgArchive Container Format v1
# Extension: .iarc
# Byte order: Little-endian
#
# ┌─────────────────────────────────────────────────────────────────┐
# │ PREAMBLE (24 bytes fixed)                                      │
# ├─────────────────────────────────────────────────────────────────┤
# │ Offset   Size   Field                                          │
# │ 0x00     4      Magic bytes: "IARC" (0x49 0x41 0x52 0x43)     │
# │ 0x04     4      Format version (uint32) — currently 1          │
# │ 0x08     8      Header JSON size in bytes (uint64)             │
# │ 0x10     8      Index JSON size in bytes (uint64)              │
# ├─────────────────────────────────────────────────────────────────┤
# │ HEADER BLOCK (variable)                                        │
# ├─────────────────────────────────────────────────────────────────┤
# │ Offset   Size              Field                               │
# │ 0x18     header_json_size  JSON-encoded ArchiveHeader          │
# │                                                                │
# │ Contains:                                                      │
# │   - total_frames           int                                 │
# │   - keyframe_count         int                                 │
# │   - interpolated_count     int                                 │
# │   - residual_count         int                                 │
# │   - forced_keyframe_count  int                                 │
# │   - deleted_count          int                                 │
# │   - gop_count              int                                 │
# │   - gop_size               int                                 │
# │   - original_width         int                                 │
# │   - original_height        int                                 │
# │   - compression_codec      str ("webp" | "jpeg")              │
# │   - compression_quality    int (1-100)                         │
# │   - archive_downscale      bool                                │
# │   - downscale_factor       float                               │
# │   - use_residuals          bool                                │
# │   - residual_strength      str ("low"|"medium"|"high")        │
# │   - face_safe              bool                                │
# │   - body_safe              bool                                │
# │   - identity_check         bool                                │
# │   - depth_aware            bool                                │
# │   - similarity_threshold   float                               │
# │   - face_threshold         float                               │
# │   - body_threshold         float                               │
# │   - rife_model             str (path)                          │
# │   - raft_model             str (path)                          │
# │   - upscaler_model         str (path)                          │
# │   - created_timestamp      str                                 │
# │   - build_time_seconds     float                               │
# │   - original_total_bytes   int                                 │
# │   - archive_total_bytes    int                                 │
# │   - reduction_percent      float                               │
# │   - index_offset           int                                 │
# │   - index_size             int                                 │
# │   - data_start_offset      int                                 │
# │   - checksum               str (SHA256)                        │
# ├─────────────────────────────────────────────────────────────────┤
# │ INDEX BLOCK (variable)                                         │
# ├─────────────────────────────────────────────────────────────────┤
# │ Offset   Size             Field                                │
# │ (after   index_json_size  JSON array of FrameEntry objects     │
# │  header)                                                       │
# │                                                                │
# │ Each FrameEntry contains:                                      │
# │   - index                  int (sequential frame number)       │
# │   - name                   str (original filename)             │
# │   - frame_type             int (0=K, 1=I, 2=R, 3=C, 4=D)     │
# │   - width                  int                                 │
# │   - height                 int                                 │
# │   - data_offset            int (relative to data_start)       │
# │   - data_size              int (bytes)                         │
# │   - residual_offset        int (relative to data_start)       │
# │   - residual_size          int (bytes)                         │
# │   - parent_keyframe_a      int (index of left keyframe)       │
# │   - parent_keyframe_b      int (index of right keyframe)      │
# │   - interpolation_timestep float (0.0-1.0)                    │
# │   - gop_id                 int                                 │
# │   - face_score             float (SSIM of face region)        │
# │   - body_score             float (SSIM of body region)        │
# │   - similarity_score       float (full-frame SSIM)            │
# │   - motion_score           float (optical flow magnitude)     │
# │   - scene_cut              bool                                │
# │   - identity_hash          str                                 │
# │   - is_deleted             bool                                │
# │   - checksum               str (SHA256 of stored data)        │
# │                                                                │
# │ Frame Types:                                                   │
# │   0 = KEYFRAME           (stored as compressed image)         │
# │   1 = INTERPOLATED       (reconstructed via RIFE, no storage) │
# │   2 = RESIDUAL           (RIFE + stored correction patch)     │
# │   3 = FORCED_KEYFRAME    (stored, forced by face/body/cut)   │
# │   4 = DELETED            (logically removed, data remains)    │
# ├─────────────────────────────────────────────────────────────────┤
# │ DATA BLOCK (variable)                                          │
# ├─────────────────────────────────────────────────────────────────┤
# │ Sequential binary blobs:                                       │
# │                                                                │
# │ For each frame in order:                                       │
# │   IF frame is KEYFRAME or FORCED_KEYFRAME:                    │
# │     → WebP/JPEG encoded image bytes (data_size bytes)         │
# │                                                                │
# │   IF frame is RESIDUAL:                                        │
# │     → WebP encoded residual patch (residual_size bytes)       │
# │                                                                │
# │   IF frame is INTERPOLATED:                                    │
# │     → nothing stored (reconstructed on-the-fly)               │
# │                                                                │
# │   IF frame is DELETED:                                         │
# │     → original data remains but is skipped on read            │
# │                                                                │
# │ Data offsets in FrameEntry are relative to DATA BLOCK start.  │
# │ Random access: seek to (data_start + data_offset), read       │
# │ data_size bytes.                                               │
# ├─────────────────────────────────────────────────────────────────┤
# │ FOOTER (64 bytes)                                              │
# ├─────────────────────────────────────────────────────────────────┤
# │ SHA256 checksum of entire file (header + index + data),       │
# │ encoded as ASCII hex, zero-padded to 64 bytes.                │
# └─────────────────────────────────────────────────────────────────┘
#
# RECONSTRUCTION ALGORITHM:
#
# To extract frame N:
#   1. Read FrameEntry[N] from index
#   2. If frame_type == KEYFRAME or FORCED_KEYFRAME:
#        a. Seek to data_start + data_offset
#        b. Read data_size bytes
#        c. Decode WebP/JPEG → BGR image
#        d. If archive_downscale: resize to original dimensions
#        e. If upscaler enabled: run RealESRGAN/SwinIR
#        f. Return image
#   3. If frame_type == INTERPOLATED:
#        a. Recursively extract parent_keyframe_a → img_A
#        b. Recursively extract parent_keyframe_b → img_B
#        c. Run RIFE(img_A, img_B, timestep) → interpolated
#        d. Return interpolated
#   4. If frame_type == RESIDUAL:
#        a. Steps 3a-3c above → interpolated
#        b. Seek to data_start + residual_offset
#        c. Read residual_size bytes → decode residual patch
#        d. output = interpolated + residual
#        e. Return output
#   5. If frame_type == DELETED:
#        → raise error or skip
#
# GOP STRUCTURE:
#
# Images are grouped into GOPs (Group of Pictures).
# Each GOP starts and ends with a keyframe.
# Interpolated frames only depend on keyframes within their GOP.
# Deleting a keyframe within a GOP invalidates dependent frames.
# Scene cuts always start a new GOP.
#
# EXAMPLE:
#
# GOP 1: [K] [I] [I] [I] [R] [I] [I] [I] [K]
# GOP 2: [C] [I] [I] [R] [I] [I] [K]
# GOP 3: [K] [I] [I] [I] [I] [I] [I] [I] [I] [I] [I] [K]
#
# Where:
#   K = Keyframe (stored)
#   I = Interpolated (not stored, reconstructed via RIFE)
#   R = Residual (RIFE + correction patch stored)
#   C = Forced keyframe (scene cut / face / body motion)
#
# ============================================================================

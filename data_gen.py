#!/usr/bin/env python3
"""
Build a raw-video fine-tuning dataset for a video-capable VLM from frame-level
industrial safety annotations.

What this script does
---------------------
1. Scans a data folder for video files and matching frame-level JSON annotations.
2. Loads the task prompt from prompt.txt.
3. Normalizes and validates frame-level labels.
4. Recomputes hazard_present / hazard_label from primitive states by default.
5. Converts frame-level labels into anchor-based clip samples:
      input  = contiguous raw video clip (default: 5s)
      target = state over the final anchor interval (default: last 1s)
6. Flags ambiguous / transition-heavy windows.
7. Creates a leakage-safe train/val/test split at the SOURCE-VIDEO level.
   Both val and test are held-out from distinct source videos to prevent
   any temporal or distribution leakage.
8. Extracts raw MP4 clips with ffmpeg.
9. Writes:
      - all_clips_manifest.jsonl    (full canonical manifest)
      - train_manifest.jsonl        (stable exported train samples)
      - val_manifest.jsonl          (stable exported val samples, for eval during training)
      - test_manifest.jsonl         (stable exported test samples, for final evaluation)
      - train_chat.jsonl            (raw-video chat format for fine-tuning)
      - val_chat.jsonl              (raw-video chat format for in-training evaluation)
      - test_chat.jsonl             (raw-video chat format for final evaluation)
      - split_report.json           (summary + chosen video-level split)

Expected input layout
---------------------
.
├── data/
│   ├── some_video.mp4
│   ├── some_video.json
│   ├── another_video.mp4
│   ├── another_video.json
│   └── ...
└── prompt.txt

Expected annotation schema (frame-level JSON)
---------------------------------------------
[
  {
    "frame_idx": 0,
    "Timestamp": "2026:01:01:07:39:41",
    "hazard_type": "forklift",
    "metrics": {
      "Object State": "no_forklift",
      "Danger Zone Relation": "no_forklift",
      "Object Direction": "none",
      "Hazard Label": "no_hazard",
      "Hazard Detected": "no"
    }
  },
  ...
]

Notes
-----
- By default the script includes a short templated "evidence" field in the final
  target JSON because your prompt asks for it. If you want only the 5 decision
  keys, run with --no-include-evidence.
- The split is done by source video, not by clip, to avoid leakage.
- The test split is chosen automatically by exhaustive search over video subsets,
  trying to preserve clip count and label distribution.

Dependencies
------------
- Python 3.9+
- OpenCV (cv2)
- ffmpeg available on PATH

Example
-------
python build_vlm_video_dataset.py \
  --data-dir data \
  --prompt-file prompt.txt \
  --out-dir vlm_dataset \
  --clip-sec 5.0 \
  --stride-sec 1.0 \
  --anchor-sec 1.0 \
  --test-ratio 0.2
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "OpenCV (cv2) is required. Install it with: pip install opencv-python"
    ) from exc

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

CANONICAL_VALUES = {
    "object_state": {"no_forklift", "stationary", "moving"},
    "zone_relation": {"no_forklift", "outside", "inside"},
    "object_direction": {"none", "towards", "away"},
    "hazard_present": {"yes", "no"},
    "hazard_label": {"unsafe_forklift_approach", "no_hazard"},
}

IMPORTANT_HARD_NEGATIVE_BUCKETS = {
    "inside_stationary",
    "inside_moving_away",
    "outside_moving_towards",
    "other_no_hazard",
    "no_forklift",
}


@dataclass
class FrameLabel:
    frame_idx: int
    time_sec: float
    object_state: str
    zone_relation: str
    object_direction: str
    hazard_present: str
    hazard_label: str
    timestamp: Optional[str] = None


@dataclass
class ClipTarget:
    hazard_label: str
    hazard_present: str
    zone_relation: str
    object_state: str
    object_direction: str
    evidence: Optional[str] = None

    def to_json_dict(self, include_evidence: bool) -> Dict[str, str]:
        out = {
            "hazard_label": self.hazard_label,
            "hazard_present": self.hazard_present,
            "zone_relation": self.zone_relation,
            "object_state": self.object_state,
            "object_direction": self.object_direction,
        }
        if include_evidence and self.evidence is not None:
            out["evidence"] = self.evidence
        return out


@dataclass
class ClipSample:
    sample_id: str
    source_video_id: str
    source_video_path: str
    source_annotation_path: str
    clip_start_sec: float
    clip_end_sec: float
    anchor_start_sec: float
    anchor_end_sec: float
    duration_sec: float
    target: ClipTarget
    anchor_mode_fraction: float
    anchor_num_frames: int
    clip_num_frames: int
    clip_positive_fraction: float
    is_transition: bool
    is_ambiguous: bool
    unique_states_in_clip: int
    hard_negative_bucket: Optional[str]
    label_signature: str
    split: Optional[str] = None
    clip_path: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a raw-video VLM fine-tuning dataset.")
    parser.add_argument("--data-dir", type=str, required=True, help="Folder containing video files and matching JSON annotation files.")
    parser.add_argument("--prompt-file", type=str, required=True, help="Path to prompt.txt.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output dataset directory.")
    parser.add_argument("--clip-sec", type=float, default=5.0, help="Contiguous clip length in seconds.")
    parser.add_argument("--stride-sec", type=float, default=1.0, help="Sliding window stride in seconds.")
    parser.add_argument("--anchor-sec", type=float, default=1.0, help="Anchor interval at end of each clip used for labeling.")
    parser.add_argument("--anchor-consensus-thr", type=float, default=0.70, help="Minimum anchor-mode fraction required for a stable label.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Desired proportion of stable clips assigned to validation.")
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Desired proportion of stable clips assigned to test.")
    parser.add_argument(
        "--split-strategy", type=str, default="clip", choices=["clip", "video"],
        help=(
            "'clip' (default): clips from every source video are distributed across all three splits, "
            "stratified by hazard_present, with temporal guard bands to prevent leakage. "
            "'video': entire source videos are assigned to a single split (guarantees zero leakage "
            "but requires >=3 source videos and means each split sees only some cameras)."
        ),
    )
    parser.add_argument(
        "--guard-clips", type=int, default=5,
        help=(
            "[clip strategy only] Number of clip positions to exclude adjacent to val/test selections "
            "to prevent temporal leakage from overlapping sliding windows. "
            "Default 5 = ceil(clip_sec=5 / stride_sec=1), guaranteeing zero frame overlap. "
            "Applied to negative (no-hazard) clips. See --positive-guard-clips for positives."
        ),
    )
    parser.add_argument(
        "--positive-test-ratio", type=float, default=0.35,
        help=(
            "[clip strategy only] Fraction of positive (hazard=yes) clips assigned to test. "
            "Higher than --test-ratio because positives are rare: you want enough in test "
            "to evaluate the model on the hard cases. Default 0.35."
        ),
    )
    parser.add_argument(
        "--positive-val-ratio", type=float, default=0.20,
        help=(
            "[clip strategy only] Fraction of positive clips assigned to val. Default 0.20."
        ),
    )
    parser.add_argument(
        "--positive-guard-clips", type=int, default=1,
        help=(
            "[clip strategy only] Guard band size (in clip positions) around val/test selections "
            "for POSITIVE clips only. Positive clips are sparse (rare hazard events) so a wide "
            "guard would consume most of them. Default 1 (just skip immediately adjacent clip)."
        ),
    )
    parser.add_argument("--min-val-videos", type=int, default=1, help="[video strategy only] Minimum source videos in val split.")
    parser.add_argument("--max-val-videos", type=int, default=None, help="[video strategy only] Maximum source videos in val split.")
    parser.add_argument("--min-test-videos", type=int, default=1, help="[video strategy only] Minimum source videos in test split.")
    parser.add_argument("--max-test-videos", type=int, default=None, help="[video strategy only] Maximum source videos in test split.")
    parser.add_argument("--keep-ambiguous", action="store_true", help="Keep ambiguous clips in exported train/test manifests instead of only the full manifest.")
    parser.add_argument("--include-evidence", dest="include_evidence", action="store_true", help="Include a short templated evidence field in the assistant JSON.")
    parser.add_argument("--no-include-evidence", dest="include_evidence", action="store_false", help="Exclude evidence from the assistant JSON.")
    parser.set_defaults(include_evidence=True)
    parser.add_argument("--recompute-hazard", dest="recompute_hazard", action="store_true", help="Recompute hazard_present/hazard_label from primitive states.")
    parser.add_argument("--no-recompute-hazard", dest="recompute_hazard", action="store_false", help="Keep the annotation's provided hazard fields after normalization.")
    parser.set_defaults(recompute_hazard=True)
    parser.add_argument("--easy-negative-keep-prob", type=float, default=0.20, help="Fraction of easy no_forklift negatives to retain in TRAIN after manifest generation.")
    parser.add_argument("--extract-clips", action="store_true", default=True, help="Extract physical MP4 clips with ffmpeg.")
    parser.add_argument("--no-extract-clips", dest="extract_clips", action="store_false", help="Do not extract clip files; keep original video path + timestamps only.")
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(x: object) -> str:
    s = str(x if x is not None else "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = s.replace("/", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def normalize_object_state(v: object) -> str:
    s = normalize_text(v)
    mapping = {
        "no_forklift": "no_forklift",
        "none": "no_forklift",
        "absent": "no_forklift",
        "missing": "no_forklift",
        "not_visible": "no_forklift",
        "stationary": "stationary",
        "stopped": "stationary",
        "idle": "stationary",
        "moving": "moving",
        "motion": "moving",
    }
    if s not in mapping:
        raise ValueError(f"Unknown object_state value: {v!r}")
    return mapping[s]


def normalize_zone_relation(v: object) -> str:
    s = normalize_text(v)
    mapping = {
        "no_forklift": "no_forklift",
        "none": "no_forklift",
        "absent": "no_forklift",
        "outside": "outside",
        "out": "outside",
        "inside": "inside",
        "in": "inside",
    }
    if s not in mapping:
        raise ValueError(f"Unknown zone_relation value: {v!r}")
    return mapping[s]


def normalize_direction(v: object) -> str:
    s = normalize_text(v)
    mapping = {
        "none": "none",
        "no_direction": "none",
        "stationary": "none",
        "towards": "towards",
        "toward": "towards",
        "towards_camera": "towards",
        "toward_camera": "towards",
        "approaching": "towards",
        "away": "away",
        "away_from_camera": "away",
        "departing": "away",
    }
    if s not in mapping:
        raise ValueError(f"Unknown object_direction value: {v!r}")
    return mapping[s]


def normalize_yes_no(v: object) -> str:
    s = normalize_text(v)
    mapping = {
        "yes": "yes",
        "true": "yes",
        "1": "yes",
        "hazard": "yes",
        "no": "no",
        "false": "no",
        "0": "no",
        "safe": "no",
    }
    if s not in mapping:
        raise ValueError(f"Unknown yes/no value: {v!r}")
    return mapping[s]


def normalize_hazard_label(v: object) -> str:
    s = normalize_text(v)
    mapping = {
        "no_hazard": "no_hazard",
        "safe": "no_hazard",
        "unsafe_forklift_approach": "unsafe_forklift_approach",
        "forklift_entry_hazard": "unsafe_forklift_approach",
        "hazard": "unsafe_forklift_approach",
    }
    if s not in mapping:
        raise ValueError(f"Unknown hazard_label value: {v!r}")
    return mapping[s]


def canonicalize_primitives(object_state: str, zone_relation: str, object_direction: str) -> Tuple[str, str, str]:
    # Force consistency for no-forklift cases.
    if object_state == "no_forklift" or zone_relation == "no_forklift":
        return "no_forklift", "no_forklift", "none"

    # Direction "none" is valid for stationary forklifts.
    if object_state == "stationary" and object_direction not in {"none", "towards", "away"}:
        object_direction = "none"

    return object_state, zone_relation, object_direction


def hazard_from_primitives(zone_relation: str, object_state: str, object_direction: str) -> Tuple[str, str]:
    present = "yes" if (
        zone_relation == "inside" and object_state == "moving" and object_direction == "towards"
    ) else "no"
    label = "unsafe_forklift_approach" if present == "yes" else "no_hazard"
    return present, label


def make_evidence(zone_relation: str, object_state: str, object_direction: str) -> str:
    if zone_relation == "no_forklift" or object_state == "no_forklift":
        return "No forklift is visible."

    workspace_phrase = "inside the workspace" if zone_relation == "inside" else "outside the workspace"

    if object_state == "stationary":
        return f"A forklift is visible {workspace_phrase} and appears stationary."

    if object_state == "moving" and object_direction == "towards":
        return f"A forklift is visible {workspace_phrase} and moving towards the camera."

    if object_state == "moving" and object_direction == "away":
        return f"A forklift is visible {workspace_phrase} and moving away from the camera."

    return f"A forklift is visible {workspace_phrase}."


def full_tuple_key(frame: FrameLabel) -> Tuple[str, str, str, str, str]:
    return (
        frame.zone_relation,
        frame.object_state,
        frame.object_direction,
        frame.hazard_present,
        frame.hazard_label,
    )


def label_signature_from_target(target: ClipTarget) -> str:
    return "|".join([
        target.hazard_label,
        target.hazard_present,
        target.zone_relation,
        target.object_state,
        target.object_direction,
    ])


def classify_hard_negative(target: ClipTarget) -> Optional[str]:
    if target.hazard_present == "yes":
        return None
    if target.zone_relation == "no_forklift":
        return "no_forklift"
    if target.zone_relation == "inside" and target.object_state == "stationary":
        return "inside_stationary"
    if target.zone_relation == "inside" and target.object_state == "moving" and target.object_direction == "away":
        return "inside_moving_away"
    if target.zone_relation == "outside" and target.object_state == "moving" and target.object_direction == "towards":
        return "outside_moving_towards"
    return "other_no_hazard"


def get_video_metadata(video_path: Path) -> Dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Could not read valid FPS/frame count from: {video_path}")

    duration_sec = frame_count / fps
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }


def find_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    videos = {}
    jsons = {}

    for p in data_dir.iterdir():
        if p.is_file():
            if p.suffix.lower() in VIDEO_EXTS:
                videos[p.stem] = p
            elif p.suffix.lower() == ".json":
                jsons[p.stem] = p

    pairs = []
    missing = []
    for stem, video_path in sorted(videos.items()):
        ann = jsons.get(stem)
        if ann is None:
            missing.append(str(video_path.name))
            continue
        pairs.append((video_path, ann))

    if missing:
        print("[WARN] Missing matching JSON annotation for videos:")
        for m in missing:
            print(f"  - {m}")

    if not pairs:
        raise SystemExit(f"No (video, json) pairs found in {data_dir}")

    return pairs


def load_frame_labels(annotation_path: Path, duration_sec: float, recompute_hazard: bool) -> Tuple[List[FrameLabel], Dict[str, int]]:
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Annotation file must contain a list: {annotation_path}")

    # Sort by frame_idx when present, otherwise keep file order.
    processed = []
    seen_frame_idxs = set()
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            continue
        frame_idx = row.get("frame_idx", i)
        if frame_idx in seen_frame_idxs:
            # Keep the last occurrence by overwriting later.
            pass
        seen_frame_idxs.add(frame_idx)
        processed.append(row)

    processed.sort(key=lambda x: int(x.get("frame_idx", 0)))

    ann_count = max(int(r.get("frame_idx", i)) for i, r in enumerate(processed)) + 1 if processed else 0
    if ann_count <= 1:
        raise ValueError(f"Annotation file has too few frames: {annotation_path}")

    ann_fps = ann_count / max(duration_sec, 1e-6)
    stats = Counter()
    frames: List[FrameLabel] = []

    for i, row in enumerate(processed):
        metrics = row.get("metrics", {}) or {}
        frame_idx = int(row.get("frame_idx", i))

        object_state = normalize_object_state(metrics.get("Object State"))
        zone_relation = normalize_zone_relation(metrics.get("Danger Zone Relation"))
        object_direction = normalize_direction(metrics.get("Object Direction"))
        object_state, zone_relation, object_direction = canonicalize_primitives(
            object_state, zone_relation, object_direction
        )

        ann_hazard_present = normalize_yes_no(metrics.get("Hazard Detected", "no"))
        ann_hazard_label = normalize_hazard_label(metrics.get("Hazard Label", "no_hazard"))
        recomputed_present, recomputed_label = hazard_from_primitives(zone_relation, object_state, object_direction)

        if (ann_hazard_present, ann_hazard_label) != (recomputed_present, recomputed_label):
            stats["recomputed_hazard_mismatch_frames"] += 1

        if recompute_hazard:
            hazard_present, hazard_label = recomputed_present, recomputed_label
        else:
            hazard_present, hazard_label = ann_hazard_present, ann_hazard_label

        time_sec = frame_idx / ann_fps
        frames.append(
            FrameLabel(
                frame_idx=frame_idx,
                time_sec=time_sec,
                object_state=object_state,
                zone_relation=zone_relation,
                object_direction=object_direction,
                hazard_present=hazard_present,
                hazard_label=hazard_label,
                timestamp=row.get("Timestamp"),
            )
        )

    # Ensure sort by time/frame.
    frames.sort(key=lambda f: (f.frame_idx, f.time_sec))

    stats["annotation_frames"] = len(frames)
    stats["annotation_count_derived_fps_x1000"] = int(round(ann_fps * 1000))
    stats["positive_frames"] = sum(1 for f in frames if f.hazard_present == "yes")
    return frames, dict(stats)


def select_frames_in_window(frames: Sequence[FrameLabel], start_sec: float, end_sec: float) -> List[FrameLabel]:
    return [f for f in frames if start_sec <= f.time_sec < end_sec]


def build_target_from_anchor(anchor_frames: Sequence[FrameLabel], include_evidence: bool) -> Tuple[ClipTarget, float]:
    if not anchor_frames:
        raise ValueError("Anchor contains no frames.")

    counter = Counter(full_tuple_key(f) for f in anchor_frames)
    mode_tuple, mode_count = counter.most_common(1)[0]
    anchor_mode_fraction = mode_count / len(anchor_frames)

    zone_relation, object_state, object_direction, hazard_present, hazard_label = mode_tuple
    evidence = make_evidence(zone_relation, object_state, object_direction) if include_evidence else None

    target = ClipTarget(
        hazard_label=hazard_label,
        hazard_present=hazard_present,
        zone_relation=zone_relation,
        object_state=object_state,
        object_direction=object_direction,
        evidence=evidence,
    )
    return target, anchor_mode_fraction


def make_sample_id(video_id: str, clip_start_sec: float, clip_end_sec: float) -> str:
    s_ms = int(round(clip_start_sec * 1000))
    e_ms = int(round(clip_end_sec * 1000))
    return f"{video_id}__{s_ms:09d}_{e_ms:09d}"


def generate_clip_samples(
    *,
    video_id: str,
    video_path: Path,
    annotation_path: Path,
    frames: Sequence[FrameLabel],
    duration_sec: float,
    clip_sec: float,
    stride_sec: float,
    anchor_sec: float,
    anchor_consensus_thr: float,
    include_evidence: bool,
) -> List[ClipSample]:
    if clip_sec <= 0 or stride_sec <= 0 or anchor_sec <= 0:
        raise ValueError("clip_sec, stride_sec, and anchor_sec must be > 0")
    if anchor_sec > clip_sec:
        raise ValueError("anchor_sec cannot be longer than clip_sec")
    if duration_sec < clip_sec:
        return []

    samples: List[ClipSample] = []
    num_steps = int(math.floor((duration_sec - clip_sec) / stride_sec)) + 1

    for step in range(num_steps):
        clip_start = round(step * stride_sec, 6)
        clip_end = round(min(clip_start + clip_sec, duration_sec), 6)
        anchor_start = round(clip_end - anchor_sec, 6)
        anchor_end = clip_end

        clip_frames = select_frames_in_window(frames, clip_start, clip_end)
        anchor_frames = select_frames_in_window(frames, anchor_start, anchor_end)
        if not clip_frames or not anchor_frames:
            continue

        target, anchor_mode_fraction = build_target_from_anchor(anchor_frames, include_evidence=include_evidence)
        label_signature = label_signature_from_target(target)
        positive_fraction = sum(1 for f in clip_frames if f.hazard_present == "yes") / len(clip_frames)
        unique_states_in_clip = len(set(full_tuple_key(f) for f in clip_frames))
        is_transition = unique_states_in_clip > 1
        is_ambiguous = anchor_mode_fraction < anchor_consensus_thr
        hard_negative_bucket = classify_hard_negative(target)

        sample = ClipSample(
            sample_id=make_sample_id(video_id, clip_start, clip_end),
            source_video_id=video_id,
            source_video_path=str(video_path),
            source_annotation_path=str(annotation_path),
            clip_start_sec=clip_start,
            clip_end_sec=clip_end,
            anchor_start_sec=anchor_start,
            anchor_end_sec=anchor_end,
            duration_sec=round(clip_end - clip_start, 6),
            target=target,
            anchor_mode_fraction=round(anchor_mode_fraction, 6),
            anchor_num_frames=len(anchor_frames),
            clip_num_frames=len(clip_frames),
            clip_positive_fraction=round(positive_fraction, 6),
            is_transition=is_transition,
            is_ambiguous=is_ambiguous,
            unique_states_in_clip=unique_states_in_clip,
            hard_negative_bucket=hard_negative_bucket,
            label_signature=label_signature,
        )
        samples.append(sample)

    return samples


def dumps_jsonl_line(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(dumps_jsonl_line(row) + "\n")


def clip_sample_to_manifest_row(sample: ClipSample, include_evidence: bool) -> Dict:
    row = {
        "sample_id": sample.sample_id,
        "source_video_id": sample.source_video_id,
        "source_video_path": sample.source_video_path,
        "source_annotation_path": sample.source_annotation_path,
        "clip_path": sample.clip_path,
        "clip_start_sec": sample.clip_start_sec,
        "clip_end_sec": sample.clip_end_sec,
        "anchor_start_sec": sample.anchor_start_sec,
        "anchor_end_sec": sample.anchor_end_sec,
        "duration_sec": sample.duration_sec,
        "target": sample.target.to_json_dict(include_evidence=include_evidence),
        "anchor_mode_fraction": sample.anchor_mode_fraction,
        "anchor_num_frames": sample.anchor_num_frames,
        "clip_num_frames": sample.clip_num_frames,
        "clip_positive_fraction": sample.clip_positive_fraction,
        "is_transition": sample.is_transition,
        "is_ambiguous": sample.is_ambiguous,
        "unique_states_in_clip": sample.unique_states_in_clip,
        "hard_negative_bucket": sample.hard_negative_bucket,
        "label_signature": sample.label_signature,
        "split": sample.split,
    }
    return row


def clip_sample_to_chat_row(sample: ClipSample, prompt_text: str, include_evidence: bool) -> Dict:
    assistant_json = sample.target.to_json_dict(include_evidence=include_evidence)

    if sample.clip_path:
        video_ref = sample.clip_path
    else:
        video_ref = sample.source_video_path

    row = {
        "sample_id": sample.sample_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_ref},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": json.dumps(assistant_json, ensure_ascii=False)}
                ],
            },
        ],
        "meta": {
            "source_video_id": sample.source_video_id,
            "clip_start_sec": sample.clip_start_sec,
            "clip_end_sec": sample.clip_end_sec,
            "anchor_start_sec": sample.anchor_start_sec,
            "anchor_end_sec": sample.anchor_end_sec,
            "label_signature": sample.label_signature,
            "hard_negative_bucket": sample.hard_negative_bucket,
            "is_transition": sample.is_transition,
            "is_ambiguous": sample.is_ambiguous,
        },
    }
    return row


def js_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    keys = set(p.keys()) | set(q.keys())
    m = {}
    for k in keys:
        pv = p.get(k, 0.0) + eps
        qv = q.get(k, 0.0) + eps
        m[k] = 0.5 * (pv + qv)

    def kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        total = 0.0
        for k in keys:
            av = a.get(k, 0.0) + eps
            bv = b.get(k, 0.0) + eps
            total += av * math.log(av / bv)
        return total

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def normalize_counter(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def summarize_samples(samples: Sequence[ClipSample]) -> Dict:
    total = len(samples)
    stable = [s for s in samples if not s.is_ambiguous]
    positives = [s for s in samples if s.target.hazard_present == "yes"]
    stable_positives = [s for s in stable if s.target.hazard_present == "yes"]

    hazard_counter = Counter(s.target.hazard_present for s in samples)
    sig_counter = Counter(s.label_signature for s in samples)
    hn_counter = Counter(s.hard_negative_bucket for s in samples if s.hard_negative_bucket)

    return {
        "num_samples": total,
        "num_stable": len(stable),
        "num_ambiguous": total - len(stable),
        "num_positive": len(positives),
        "num_positive_stable": len(stable_positives),
        "hazard_present_distribution": dict(hazard_counter),
        "label_signature_distribution": dict(sig_counter),
        "hard_negative_distribution": dict(hn_counter),
    }


def _assign_group_to_splits(
    clips: List[ClipSample],
    val_ratio: float,
    test_ratio: float,
    guard_clips: int,
    val_ids: set,
    test_ids: set,
) -> None:
    """
    Assign clips from a temporally-sorted group to val/test sets using
    systematic (evenly-spaced) sampling with guard-band exclusion.

    Guard bands: the `guard_clips` positions on each side of every selected
    val or test clip are excluded from selection for the opposing split.
    This prevents temporal leakage between splits for overlapping sliding-
    window clips (e.g. with clip_sec=5, stride_sec=1, clips that are 5
    positions apart have zero frame overlap, so guard_clips=5 is safe).
    """
    n = len(clips)
    if n == 0:
        return

    # ── Step 1: select val positions at regular intervals ────────────────────
    n_val = max(1, round(n * val_ratio)) if n >= 3 else 0
    if n_val > 0:
        step = max(1, n // n_val)
        val_positions: set = set(itertools.islice(range(0, n, step), n_val))
    else:
        val_positions = set()

    # ── Step 2: compute guard band around val positions ───────────────────────
    guarded: set = set()
    for p in val_positions:
        for g in range(max(0, p - guard_clips), min(n, p + guard_clips + 1)):
            if g not in val_positions:
                guarded.add(g)

    # ── Step 3: select test positions from non-val, non-guarded pool ─────────
    eligible: List[int] = [
        i for i in range(n) if i not in val_positions and i not in guarded
    ]
    n_test = max(1, round(n * test_ratio)) if eligible else 0
    if n_test > 0 and eligible:
        step_t = max(1, len(eligible) // n_test)
        test_positions: set = set(itertools.islice(eligible[::step_t], n_test))
    else:
        test_positions = set()

    # ── Step 4: record sample IDs ────────────────────────────────────────────
    for i, clip in enumerate(clips):
        if i in val_positions:
            val_ids.add(clip.sample_id)
        elif i in test_positions:
            test_ids.add(clip.sample_id)
        # else → train (default)


def choose_clip_level_stratified_split(
    samples: Sequence[ClipSample],
    val_ratio: float,
    test_ratio: float,
    guard_clips: int,
    positive_val_ratio: float,
    positive_test_ratio: float,
    positive_guard_clips: int,
) -> Tuple[set, set, Dict]:
    """
    Clip-level stratified split with temporal guard bands.

    Unlike the video-level split, this ensures **every source video is
    represented in all three splits**, so each split sees all camera angles
    and scene variations present in the data.

    Positive and negative clips are assigned using *separate* sampling
    parameters:

    - Negatives (no-hazard): use val_ratio / test_ratio / guard_clips.
      guard_clips=5 prevents leakage between the dense 1s-stride windows.

    - Positives (hazard=yes): use positive_val_ratio / positive_test_ratio /
      positive_guard_clips.  Positive clips are rare and sparse, so a large
      guard would silently consume most of them.  A smaller guard (default 1)
      and a higher test ratio (default 0.35) ensures enough positives reach
      val and test to make evaluation meaningful.

    Returns
    -------
    val_ids   : set of sample_id strings assigned to val
    test_ids  : set of sample_id strings assigned to test
    split_info: dict with statistics and per-video coverage breakdown
    """
    stable = [s for s in samples if not s.is_ambiguous]

    by_video: Dict[str, List[ClipSample]] = defaultdict(list)
    for s in stable:
        by_video[s.source_video_id].append(s)

    val_ids: set = set()
    test_ids: set = set()

    for video_id in sorted(by_video):
        clips_sorted = sorted(by_video[video_id], key=lambda s: s.clip_start_sec)

        # Stratify positive and negative clips independently  so that even
        # rare positive events are distributed across all three splits,
        # rather than all landing in train by chance.
        positives = [s for s in clips_sorted if s.target.hazard_present == "yes"]
        negatives = [s for s in clips_sorted if s.target.hazard_present == "no"]

        # Use separate ratios and guard sizes for positives vs negatives.
        # Negatives are dense (1 clip/sec) → large guard prevents leakage.
        # Positives are sparse (rare events) → small guard preserves coverage.
        _assign_group_to_splits(positives, positive_val_ratio, positive_test_ratio,
                                positive_guard_clips, val_ids, test_ids)
        _assign_group_to_splits(negatives, val_ratio, test_ratio, guard_clips, val_ids, test_ids)

    # ── Build split_info dict ─────────────────────────────────────────────────
    train_s = [s for s in stable if s.sample_id not in val_ids and s.sample_id not in test_ids]
    val_s   = [s for s in stable if s.sample_id in val_ids]
    test_s  = [s for s in stable if s.sample_id in test_ids]
    total   = len(stable)

    split_info = {
        "strategy": "clip_level_stratified",
        "guard_clips_negatives": guard_clips,
        "guard_clips_positives": positive_guard_clips,
        "positive_val_ratio_target":  positive_val_ratio,
        "positive_test_ratio_target": positive_test_ratio,
        "val_ratio_actual":  len(val_s)  / max(total, 1),
        "test_ratio_actual": len(test_s) / max(total, 1),
        "train_num_stable": len(train_s),
        "val_num_stable":   len(val_s),
        "test_num_stable":  len(test_s),
        "train_positive": sum(1 for s in train_s if s.target.hazard_present == "yes"),
        "val_positive":   sum(1 for s in val_s   if s.target.hazard_present == "yes"),
        "test_positive":  sum(1 for s in test_s  if s.target.hazard_present == "yes"),
        "train_videos": sorted({s.source_video_id for s in train_s}),
        "val_videos":   sorted({s.source_video_id for s in val_s}),
        "test_videos":  sorted({s.source_video_id for s in test_s}),
        "train_hazard_present_distribution": dict(Counter(s.target.hazard_present for s in train_s)),
        "val_hazard_present_distribution":   dict(Counter(s.target.hazard_present for s in val_s)),
        "test_hazard_present_distribution":  dict(Counter(s.target.hazard_present for s in test_s)),
        # Per-video breakdown so you can verify every camera is covered.
        "per_video_split_counts": {
            vid: {
                "train": sum(1 for s in by_video[vid]
                             if s.sample_id not in val_ids and s.sample_id not in test_ids
                             and not s.is_ambiguous),
                "val":   sum(1 for s in by_video[vid] if s.sample_id in val_ids),
                "test":  sum(1 for s in by_video[vid] if s.sample_id in test_ids),
            }
            for vid in sorted(by_video)
        },
    }
    return val_ids, test_ids, split_info


def _score_split_subset(
    subset_samples: List[ClipSample],
    remainder_samples: List[ClipSample],
    total_stable: int,
    target_ratio: float,
    global_hazard_dist: Dict[str, float],
    global_sig_dist: Dict[str, float],
    global_hn_dist: Dict[str, float],
) -> float:
    """Compute a split quality score (lower is better)."""
    ratio_actual = len(subset_samples) / max(total_stable, 1)
    sub_hazard_dist = normalize_counter(Counter(s.target.hazard_present for s in subset_samples))
    sub_sig_dist = normalize_counter(Counter(s.label_signature for s in subset_samples))
    sub_hn_dist = normalize_counter(Counter(s.hard_negative_bucket for s in subset_samples if s.hard_negative_bucket))

    score = 0.0
    score += 2.0 * abs(ratio_actual - target_ratio)
    score += 1.5 * js_divergence(global_hazard_dist, sub_hazard_dist)
    score += 1.0 * js_divergence(global_sig_dist, sub_sig_dist)
    if global_hn_dist:
        score += 0.5 * js_divergence(global_hn_dist, sub_hn_dist)

    sub_pos = sum(1 for s in subset_samples if s.target.hazard_present == "yes")
    rem_pos = sum(1 for s in remainder_samples if s.target.hazard_present == "yes")
    if sub_pos == 0:
        score += 5.0
    if rem_pos == 0:
        score += 20.0

    sub_hn_buckets = {s.hard_negative_bucket for s in subset_samples if s.hard_negative_bucket}
    score += 0.2 * len(IMPORTANT_HARD_NEGATIVE_BUCKETS - sub_hn_buckets)
    return score


def choose_video_level_three_way_split(
    samples: Sequence[ClipSample],
    val_ratio: float,
    test_ratio: float,
    min_val_videos: int,
    max_val_videos: Optional[int],
    min_test_videos: int,
    max_test_videos: Optional[int],
) -> Tuple[set, set, Dict]:
    """
    Choose a leakage-safe train/val/test split at the source-video level.

    Strategy:
      1. Pick the test videos first (exhaustive search over video subsets).
      2. From the remaining videos, pick val videos (exhaustive search).
      3. Everything else is train.

    Both val and test are optimised independently for label-distribution
    match and positive-sample presence.

    Returns:
        chosen_val_videos  – set of source_video_id strings assigned to val
        chosen_test_videos – set of source_video_id strings assigned to test
        split_info         – dict with split statistics
    """
    stable_samples = [s for s in samples if not s.is_ambiguous]
    by_video: Dict[str, List[ClipSample]] = defaultdict(list)
    for s in stable_samples:
        by_video[s.source_video_id].append(s)

    videos = sorted(by_video.keys())
    n = len(videos)
    if n < 3:
        raise ValueError(
            f"Need at least 3 source videos for a train/val/test split (found {n}). "
            "Lower --min-val-videos / --min-test-videos or add more source videos."
        )

    if max_test_videos is None:
        max_test_videos = max(min_test_videos, n - 2)
    max_test_videos = min(max_test_videos, n - 2)  # leave room for val + train

    if max_val_videos is None:
        max_val_videos = max(min_val_videos, n - 2)

    global_hazard_dist = normalize_counter(Counter(s.target.hazard_present for s in stable_samples))
    global_sig_dist = normalize_counter(Counter(s.label_signature for s in stable_samples))
    global_hn_dist = normalize_counter(Counter(s.hard_negative_bucket for s in stable_samples if s.hard_negative_bucket))
    total_stable = len(stable_samples)

    # ── Step 1: choose test videos ────────────────────────────────────────────
    best_test_subset: Optional[set] = None
    best_test_score = float("inf")

    for r in range(min_test_videos, max_test_videos + 1):
        for subset in itertools.combinations(videos, r):
            subset_set = set(subset)
            test_s = [s for s in stable_samples if s.source_video_id in subset_set]
            remainder_s = [s for s in stable_samples if s.source_video_id not in subset_set]
            if not test_s or len(remainder_s) < min_val_videos + 1:
                continue  # need at least min_val_videos left for val + 1 for train
            score = _score_split_subset(
                test_s, remainder_s, total_stable, test_ratio,
                global_hazard_dist, global_sig_dist, global_hn_dist,
            )
            if score < best_test_score:
                best_test_score = score
                best_test_subset = subset_set

    if best_test_subset is None:
        raise RuntimeError("Failed to find a valid video-level test split.")
    assert best_test_subset is not None  # satisfies type checker

    # ── Step 2: from the remaining videos, choose val videos ──────────────────
    remaining_videos = sorted(set(videos) - best_test_subset)
    remaining_stable = [s for s in stable_samples if s.source_video_id in set(remaining_videos)]

    # Recompute global dists over the non-test pool for fair val scoring.
    rem_hazard_dist = normalize_counter(Counter(s.target.hazard_present for s in remaining_stable))
    rem_sig_dist = normalize_counter(Counter(s.label_signature for s in remaining_stable))
    rem_hn_dist = normalize_counter(Counter(s.hard_negative_bucket for s in remaining_stable if s.hard_negative_bucket))
    total_remaining = len(remaining_stable)

    effective_val_ratio = val_ratio / max(1.0 - float(test_ratio), 1e-6)  # ratio within remaining

    best_val_subset: Optional[set] = None
    best_val_score = float("inf")
    effective_max_val = min(max_val_videos, len(remaining_videos) - 1)

    for r in range(min_val_videos, effective_max_val + 1):
        for subset in itertools.combinations(remaining_videos, r):
            subset_set = set(subset)
            val_s = [s for s in remaining_stable if s.source_video_id in subset_set]
            train_s = [s for s in remaining_stable if s.source_video_id not in subset_set]
            if not val_s or not train_s:
                continue
            score = _score_split_subset(
                val_s, train_s, total_remaining, effective_val_ratio,
                rem_hazard_dist, rem_sig_dist, rem_hn_dist,
            )
            if score < best_val_score:
                best_val_score = score
                best_val_subset = subset_set

    if best_val_subset is None:
        raise RuntimeError("Failed to find a valid video-level val split.")
    assert best_val_subset is not None  # satisfies type checker

    # ── Step 3: assemble info dict ────────────────────────────────────────────
    test_s = [s for s in stable_samples if s.source_video_id in best_test_subset]
    val_s = [s for s in stable_samples if s.source_video_id in best_val_subset]
    train_s = [s for s in stable_samples if s.source_video_id not in best_test_subset | best_val_subset]

    split_info = {
        "test_score": best_test_score,
        "val_score": best_val_score,
        "test_ratio_actual": len(test_s) / max(total_stable, 1),
        "val_ratio_actual": len(val_s) / max(total_stable, 1),
        "train_num_stable": len(train_s),
        "val_num_stable": len(val_s),
        "test_num_stable": len(test_s),
        "train_positive": sum(1 for s in train_s if s.target.hazard_present == "yes"),
        "val_positive": sum(1 for s in val_s if s.target.hazard_present == "yes"),
        "test_positive": sum(1 for s in test_s if s.target.hazard_present == "yes"),
        "train_videos": sorted(set(videos) - best_test_subset - best_val_subset),
        "val_videos": sorted(best_val_subset),
        "test_videos": sorted(best_test_subset),
        "train_hazard_present_distribution": dict(Counter(s.target.hazard_present for s in train_s)),
        "val_hazard_present_distribution": dict(Counter(s.target.hazard_present for s in val_s)),
        "test_hazard_present_distribution": dict(Counter(s.target.hazard_present for s in test_s)),
    }
    return best_val_subset, best_test_subset, split_info


def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_clip_ffmpeg(src_video: Path, dst_clip: Path, start_sec: float, end_sec: float) -> None:
    ensure_dir(dst_clip.parent)
    duration = max(end_sec - start_sec, 0.001)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(src_video),
        "-t",
        f"{duration:.6f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst_clip),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src_video.name} -> {dst_clip.name}\n"
            f"STDERR:\n{result.stderr}"
        )


def maybe_downsample_easy_negatives_train(train_samples: List[ClipSample], keep_prob: float) -> List[ClipSample]:
    if keep_prob >= 0.999:
        return train_samples
    keep_prob = max(0.0, min(1.0, keep_prob))

    retained: List[ClipSample] = []
    easy_no_forklift = [s for s in train_samples if s.hard_negative_bucket == "no_forklift" and s.target.hazard_present == "no"]
    others = [s for s in train_samples if not (s.hard_negative_bucket == "no_forklift" and s.target.hazard_present == "no")]

    # Deterministic sub-sampling based on sample_id hash.
    for s in easy_no_forklift:
        h = (abs(hash(s.sample_id)) % 10_000) / 10_000.0
        if h < keep_prob:
            retained.append(s)

    return sorted(others + retained, key=lambda x: (x.source_video_id, x.clip_start_sec))


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    prompt_file = Path(args.prompt_file)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {data_dir}")
    if not prompt_file.exists():
        raise SystemExit(f"Prompt file does not exist: {prompt_file}")
    if args.extract_clips and not ffmpeg_exists():
        raise SystemExit("ffmpeg is required but was not found on PATH.")

    prompt_text = read_text(prompt_file)
    pairs = find_pairs(data_dir)

    all_samples: List[ClipSample] = []
    per_video_report = {}

    print(f"Found {len(pairs)} (video, json) pairs.")

    for video_path, ann_path in pairs:
        video_id = video_path.stem
        meta = get_video_metadata(video_path)
        frames, ann_stats = load_frame_labels(
            annotation_path=ann_path,
            duration_sec=meta["duration_sec"],
            recompute_hazard=args.recompute_hazard,
        )

        samples = generate_clip_samples(
            video_id=video_id,
            video_path=video_path,
            annotation_path=ann_path,
            frames=frames,
            duration_sec=meta["duration_sec"],
            clip_sec=args.clip_sec,
            stride_sec=args.stride_sec,
            anchor_sec=args.anchor_sec,
            anchor_consensus_thr=args.anchor_consensus_thr,
            include_evidence=args.include_evidence,
        )
        all_samples.extend(samples)

        per_video_report[video_id] = {
            "video_path": str(video_path),
            "annotation_path": str(ann_path),
            "video_meta": meta,
            "annotation_stats": ann_stats,
            "clip_summary": summarize_samples(samples),
        }

        print(
            f"[OK] {video_id}: duration={meta['duration_sec']:.2f}s, "
            f"ann_frames={ann_stats['annotation_frames']}, clips={len(samples)}, "
            f"stable={sum(not s.is_ambiguous for s in samples)}, "
            f"positive={sum(s.target.hazard_present == 'yes' for s in samples)}"
        )

    if not all_samples:
        raise SystemExit("No clip samples were generated.")

    # ── Choose train / val / test split ──────────────────────────────────────
    if args.split_strategy == "clip":
        # Default: distribute clips from every video across all three splits,
        # stratified by hazard_present, with temporal guard bands.
        print(
            f"\nSplit strategy: clip-level stratified "
            f"(guard_clips={args.guard_clips}, val={args.val_ratio:.0%}, test={args.test_ratio:.0%})"
        )
        val_ids, test_ids, split_info = choose_clip_level_stratified_split(
            all_samples,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            guard_clips=args.guard_clips,
            positive_val_ratio=args.positive_val_ratio,
            positive_test_ratio=args.positive_test_ratio,
            positive_guard_clips=args.positive_guard_clips,
        )
        for s in all_samples:
            if s.sample_id in test_ids:
                s.split = "test"
            elif s.sample_id in val_ids:
                s.split = "val"
            else:
                s.split = "train"
    else:
        # Fallback: whole source videos assigned to a single split.
        # Requires >=3 source videos; guarantees zero temporal leakage.
        print(
            f"\nSplit strategy: video-level "
            f"(val={args.val_ratio:.0%}, test={args.test_ratio:.0%})"
        )
        chosen_val_videos, chosen_test_videos, split_info = choose_video_level_three_way_split(
            all_samples,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            min_val_videos=args.min_val_videos,
            max_val_videos=args.max_val_videos,
            min_test_videos=args.min_test_videos,
            max_test_videos=args.max_test_videos,
        )
        for s in all_samples:
            if s.source_video_id in chosen_test_videos:
                s.split = "test"
            elif s.source_video_id in chosen_val_videos:
                s.split = "val"
            else:
                s.split = "train"

    exportable = all_samples if args.keep_ambiguous else [s for s in all_samples if not s.is_ambiguous]
    train_samples = [s for s in exportable if s.split == "train"]
    val_samples   = [s for s in exportable if s.split == "val"]
    test_samples  = [s for s in exportable if s.split == "test"]

    # Easy-negative downsampling only on train — never on val/test so
    # evaluation distributions stay representative.
    train_samples = maybe_downsample_easy_negatives_train(train_samples, keep_prob=args.easy_negative_keep_prob)

    if args.extract_clips:
        clips_root = out_dir / "clips"
        for s in train_samples + val_samples + test_samples:
            split_dir = clips_root / s.split / s.source_video_id
            dst_clip = split_dir / f"{s.sample_id}.mp4"
            extract_clip_ffmpeg(
                src_video=Path(s.source_video_path),
                dst_clip=dst_clip,
                start_sec=s.clip_start_sec,
                end_sec=s.clip_end_sec,
            )
            s.clip_path = str(dst_clip)

    all_manifest_path   = out_dir / "all_clips_manifest.jsonl"
    train_manifest_path = out_dir / "train_manifest.jsonl"
    val_manifest_path   = out_dir / "val_manifest.jsonl"
    test_manifest_path  = out_dir / "test_manifest.jsonl"
    train_chat_path     = out_dir / "train_chat.jsonl"
    val_chat_path       = out_dir / "val_chat.jsonl"
    test_chat_path      = out_dir / "test_chat.jsonl"
    split_report_path   = out_dir / "split_report.json"

    write_jsonl(
        all_manifest_path,
        [clip_sample_to_manifest_row(s, include_evidence=args.include_evidence) for s in all_samples],
    )
    write_jsonl(
        train_manifest_path,
        [clip_sample_to_manifest_row(s, include_evidence=args.include_evidence) for s in train_samples],
    )
    write_jsonl(
        val_manifest_path,
        [clip_sample_to_manifest_row(s, include_evidence=args.include_evidence) for s in val_samples],
    )
    write_jsonl(
        test_manifest_path,
        [clip_sample_to_manifest_row(s, include_evidence=args.include_evidence) for s in test_samples],
    )
    write_jsonl(
        train_chat_path,
        [clip_sample_to_chat_row(s, prompt_text=prompt_text, include_evidence=args.include_evidence) for s in train_samples],
    )
    write_jsonl(
        val_chat_path,
        [clip_sample_to_chat_row(s, prompt_text=prompt_text, include_evidence=args.include_evidence) for s in val_samples],
    )
    write_jsonl(
        test_chat_path,
        [clip_sample_to_chat_row(s, prompt_text=prompt_text, include_evidence=args.include_evidence) for s in test_samples],
    )

    report = {
        "config": {
            "data_dir": str(data_dir),
            "prompt_file": str(prompt_file),
            "out_dir": str(out_dir),
            "clip_sec": args.clip_sec,
            "stride_sec": args.stride_sec,
            "anchor_sec": args.anchor_sec,
            "anchor_consensus_thr": args.anchor_consensus_thr,
            "val_ratio_target": args.val_ratio,
            "test_ratio_target": args.test_ratio,
            "include_evidence": args.include_evidence,
            "recompute_hazard": args.recompute_hazard,
            "keep_ambiguous_in_export": args.keep_ambiguous,
            "easy_negative_keep_prob_train": args.easy_negative_keep_prob,
            "extract_clips": args.extract_clips,
        },
        "overall_summary_all": summarize_samples(all_samples),
        "overall_summary_export_train": summarize_samples(train_samples),
        "overall_summary_export_val": summarize_samples(val_samples),
        "overall_summary_export_test": summarize_samples(test_samples),
        "split_selection": split_info,
        "per_video": per_video_report,
        "output_files": {
            "all_clips_manifest": str(all_manifest_path),
            "train_manifest": str(train_manifest_path),
            "val_manifest": str(val_manifest_path),
            "test_manifest": str(test_manifest_path),
            "train_chat": str(train_chat_path),
            "val_chat": str(val_chat_path),
            "test_chat": str(test_chat_path),
        },
    }
    split_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDone.")
    print(f"  All manifest  : {all_manifest_path}")
    print(f"  Train manifest: {train_manifest_path}  ({len(train_samples)} samples)")
    print(f"  Val manifest  : {val_manifest_path}   ({len(val_samples)} samples)")
    print(f"  Test manifest : {test_manifest_path}  ({len(test_samples)} samples)")
    print(f"  Train chat    : {train_chat_path}")
    print(f"  Val chat      : {val_chat_path}")
    print(f"  Test chat     : {test_chat_path}")
    print(f"  Split report  : {split_report_path}")
    print()
    print("Split summary:")
    print(f"  Train : {split_info['train_num_stable']} stable clips from {len(split_info['train_videos'])} video(s)")
    print(f"          positives={split_info['train_positive']}, distribution={split_info['train_hazard_present_distribution']}")
    print(f"  Val   : {split_info['val_num_stable']} stable clips from {len(split_info['val_videos'])} video(s)")
    print(f"          positives={split_info['val_positive']}, distribution={split_info['val_hazard_present_distribution']}")
    print(f"  Test  : {split_info['test_num_stable']} stable clips from {len(split_info['test_videos'])} video(s)")
    print(f"          positives={split_info['test_positive']}, distribution={split_info['test_hazard_present_distribution']}")


if __name__ == "__main__":
    main()

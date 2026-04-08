#!/usr/bin/env python3
# Builds raw-video fine-tuning data from frame-level industrial safety annotations.
# Supported task modes:
#   - forklift: process only files whose stem starts with `fork_`
#   - robot: process only files whose stem starts with `robot_`
#   - both: process both task types together into merged outputs
# Required prompt args:
#   - `--fork-prompt-file` for forklift mode
#   - `--robot-prompt-file` for robot mode
#   - both prompt args for `--task-mode both`
# Example commands:
#   python data_gen.py --data-dir data --task-mode forklift --fork-prompt-file prompts/fork_prompt_v2.txt --out-dir vlm_dataset_forklift --no-extract-clips
#   python data_gen.py --data-dir data --task-mode robot --robot-prompt-file prompts/robot_propmt_v1.txt --out-dir vlm_dataset_robot --no-extract-clips
#   python data_gen.py --data-dir data --task-mode both --fork-prompt-file prompts/fork_prompt_v2.txt --robot-prompt-file prompts/robot_propmt_v1.txt --out-dir vlm_dataset_both --no-extract-clips
"""
Build a raw-video fine-tuning dataset for a video-capable VLM from frame-level
industrial safety annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
TASK_PREFIXES = {
    "fork_": "forklift",
    "robot_": "robot",
}

TASK_CONFIGS: Dict[str, Dict[str, object]] = {
    "forklift": {
        "filename_prefix": "fork_",
        "has_direction": True,
        "positive_hazard_label": "unsafe_forklift_approach",
        "easy_negative_bucket": "no_forklift",
        "important_hard_negative_buckets": {
            "inside_stationary",
            "inside_moving_away",
            "outside_moving_towards",
            "other_no_hazard",
            "no_forklift",
        },
        "allowed_values": {
            "hazard_label": {"unsafe_forklift_approach", "no_hazard"},
            "hazard_present": {"yes", "no"},
            "zone_relation": {"outside", "inside", "no_forklift"},
            "object_state": {"stationary", "moving", "no_forklift"},
            "object_direction": {"towards", "away", "none"},
        },
        "object_state_map": {
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
        },
        "zone_relation_map": {
            "no_forklift": "no_forklift",
            "none": "no_forklift",
            "absent": "no_forklift",
            "outside": "outside",
            "out": "outside",
            "inside": "inside",
            "in": "inside",
        },
        "hazard_label_map": {
            "no_hazard": "no_hazard",
            "safe": "no_hazard",
            "unsafe_forklift_approach": "unsafe_forklift_approach",
            "forklift_entry_hazard": "unsafe_forklift_approach",
            "hazard": "unsafe_forklift_approach",
        },
        "direction_map": {
            "": "none",
            "none": "none",
            "no_direction": "none",
            "stationary": "none",
            "missing": "none",
            "towards": "towards",
            "toward": "towards",
            "towards_camera": "towards",
            "toward_camera": "towards",
            "approaching": "towards",
            "away": "away",
            "away_from_camera": "away",
            "departing": "away",
        },
    },
    "robot": {
        "filename_prefix": "robot_",
        "has_direction": False,
        "positive_hazard_label": "unsafe_machine_proximity",
        "easy_negative_bucket": "no_worker",
        "important_hard_negative_buckets": {
            "inside_stationary",
            "outside_moving",
            "outside_stationary",
            "other_no_hazard",
            "no_worker",
        },
        "allowed_values": {
            "hazard_label": {"unsafe_machine_proximity", "no_hazard"},
            "hazard_present": {"yes", "no"},
            "zone_relation": {"outside", "inside", "no_worker"},
            "object_state": {"stationary", "moving"},
        },
        "object_state_map": {
            "stationary": "stationary",
            "stopped": "stationary",
            "idle": "stationary",
            "still": "stationary",
            "moving": "moving",
            "motion": "moving",
            "active": "moving",
        },
        "zone_relation_map": {
            "no_worker": "no_worker",
            "none": "no_worker",
            "absent": "no_worker",
            "no_person": "no_worker",
            "no_human": "no_worker",
            "outside": "outside",
            "out": "outside",
            "inside": "inside",
            "in": "inside",
        },
        "hazard_label_map": {
            "no_hazard": "no_hazard",
            "safe": "no_hazard",
            "unsafe_machine_proximity": "unsafe_machine_proximity",
            "unsafe_machinery_proximity": "unsafe_machine_proximity",
            "machine_proximity_hazard": "unsafe_machine_proximity",
            "hazard": "unsafe_machine_proximity",
        },
        "direction_map": {},
    },
}


@dataclass
class FrameLabel:
    task: str
    frame_idx: int
    time_sec: float
    object_state: str
    zone_relation: str
    object_direction: Optional[str]
    hazard_present: str
    hazard_label: str
    timestamp: Optional[str] = None


@dataclass
class ClipTarget:
    task: str
    hazard_label: str
    hazard_present: str
    zone_relation: str
    object_state: str
    object_direction: Optional[str] = None
    evidence: Optional[str] = None

    def to_json_dict(self, include_evidence: bool) -> Dict[str, str]:
        out = {
            "hazard_label": self.hazard_label,
            "hazard_present": self.hazard_present,
            "zone_relation": self.zone_relation,
            "object_state": self.object_state,
        }
        if self.object_direction is not None:
            out["object_direction"] = self.object_direction
        if include_evidence and self.evidence is not None:
            out["evidence"] = self.evidence
        return out


@dataclass
class ClipSample:
    sample_id: str
    task: str
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
    parser.add_argument("--out-dir", type=str, required=True, help="Output dataset directory.")
    parser.add_argument("--task-mode", type=str, default="forklift", choices=["forklift", "robot", "both"], help="Which filename-prefix task subset to process.")
    parser.add_argument("--fork-prompt-file", type=str, help="Prompt file used for forklift samples.")
    parser.add_argument("--robot-prompt-file", type=str, help="Prompt file used for robot samples.")
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
            "'video': entire source videos are assigned to a single split."
        ),
    )
    parser.add_argument("--guard-clips", type=int, default=5, help="[clip strategy only] Number of clip positions to exclude adjacent to val/test selections for negative clips.")
    parser.add_argument("--positive-test-ratio", type=float, default=0.35, help="[clip strategy only] Fraction of positive clips assigned to test.")
    parser.add_argument("--positive-val-ratio", type=float, default=0.20, help="[clip strategy only] Fraction of positive clips assigned to val.")
    parser.add_argument("--positive-guard-clips", type=int, default=1, help="[clip strategy only] Guard band size around positive val/test selections.")
    parser.add_argument("--min-val-videos", type=int, default=1, help="[video strategy only] Minimum source videos in val split.")
    parser.add_argument("--max-val-videos", type=int, default=None, help="[video strategy only] Maximum source videos in val split.")
    parser.add_argument("--min-test-videos", type=int, default=1, help="[video strategy only] Minimum source videos in test split.")
    parser.add_argument("--max-test-videos", type=int, default=None, help="[video strategy only] Maximum source videos in test split.")
    parser.add_argument("--keep-ambiguous", action="store_true", help="Keep ambiguous clips in exported train/val/test manifests instead of only the full manifest.")
    parser.add_argument("--include-evidence", dest="include_evidence", action="store_true", help="Include a short templated evidence field in manifest target JSON. Chat JSON stays strict task schema.")
    parser.add_argument("--no-include-evidence", dest="include_evidence", action="store_false", help="Exclude evidence from manifest target JSON.")
    parser.set_defaults(include_evidence=False)
    parser.add_argument("--recompute-hazard", dest="recompute_hazard", action="store_true", help="Recompute hazard_present/hazard_label from primitive states.")
    parser.add_argument("--no-recompute-hazard", dest="recompute_hazard", action="store_false", help="Keep the annotation's provided hazard fields after normalization.")
    parser.set_defaults(recompute_hazard=True)
    parser.add_argument("--easy-negative-keep-prob", type=float, default=0.20, help="Fraction of easy task-specific negatives to retain in TRAIN after manifest generation.")
    parser.add_argument("--extract-clips", action="store_true", default=True, help="Extract physical MP4 clips with ffmpeg.")
    parser.add_argument("--no-extract-clips", dest="extract_clips", action="store_false", help="Do not extract clip files; keep original video path + timestamps only.")
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_console_text(text: object) -> str:
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def normalize_text(x: object) -> str:
    s = str(x if x is not None else "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = s.replace("/", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def selected_tasks(task_mode: str) -> Set[str]:
    return set(TASK_CONFIGS) if task_mode == "both" else {task_mode}


def detect_task_from_stem(stem: str) -> Optional[str]:
    for prefix, task in TASK_PREFIXES.items():
        if stem.startswith(prefix):
            return task
    return None


def load_prompt_texts(args: argparse.Namespace) -> Dict[str, str]:
    required_tasks = selected_tasks(args.task_mode)
    prompt_paths = {
        "forklift": Path(args.fork_prompt_file) if args.fork_prompt_file else None,
        "robot": Path(args.robot_prompt_file) if args.robot_prompt_file else None,
    }
    arg_names = {
        "forklift": "--fork-prompt-file",
        "robot": "--robot-prompt-file",
    }

    prompt_texts: Dict[str, str] = {}
    for task in required_tasks:
        prompt_path = prompt_paths[task]
        if prompt_path is None:
            raise SystemExit(f"{task} mode requires {arg_names[task]}.")
        if not prompt_path.exists():
            raise SystemExit(f"Prompt file does not exist: {prompt_path}")
        prompt_texts[task] = read_text(prompt_path)
    return prompt_texts


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


def normalize_object_state(task: str, v: object) -> str:
    mapping = TASK_CONFIGS[task]["object_state_map"]
    assert isinstance(mapping, dict)
    s = normalize_text(v)
    if s not in mapping:
        raise ValueError(f"Unknown object_state value for {task}: {v!r}")
    return str(mapping[s])


def normalize_zone_relation(task: str, v: object) -> str:
    mapping = TASK_CONFIGS[task]["zone_relation_map"]
    assert isinstance(mapping, dict)
    s = normalize_text(v)
    if s not in mapping:
        raise ValueError(f"Unknown zone_relation value for {task}: {v!r}")
    return str(mapping[s])


def normalize_direction(task: str, v: object) -> Optional[str]:
    if not bool(TASK_CONFIGS[task]["has_direction"]):
        return None
    mapping = TASK_CONFIGS[task]["direction_map"]
    assert isinstance(mapping, dict)
    s = normalize_text(v)
    if s not in mapping:
        raise ValueError(f"Unknown object_direction value for {task}: {v!r}")
    return str(mapping[s])


def normalize_hazard_label(task: str, v: object) -> str:
    mapping = TASK_CONFIGS[task]["hazard_label_map"]
    assert isinstance(mapping, dict)
    s = normalize_text(v)
    if s not in mapping:
        raise ValueError(f"Unknown hazard_label value for {task}: {v!r}")
    return str(mapping[s])


def canonicalize_primitives(task: str, object_state: str, zone_relation: str, object_direction: Optional[str]) -> Tuple[str, str, Optional[str]]:
    if task == "forklift":
        if object_state == "no_forklift" or zone_relation == "no_forklift":
            return "no_forklift", "no_forklift", "none"
        if object_state == "stationary":
            return object_state, zone_relation, "none"
        return object_state, zone_relation, object_direction or "none"
    return object_state, zone_relation, None


def validate_primitives(task: str, object_state: str, zone_relation: str, object_direction: Optional[str]) -> None:
    allowed = TASK_CONFIGS[task]["allowed_values"]
    assert isinstance(allowed, dict)
    if object_state not in allowed["object_state"]:
        raise ValueError(f"Invalid canonical object_state for {task}: {object_state}")
    if zone_relation not in allowed["zone_relation"]:
        raise ValueError(f"Invalid canonical zone_relation for {task}: {zone_relation}")
    if bool(TASK_CONFIGS[task]["has_direction"]) and object_direction not in allowed["object_direction"]:
        raise ValueError(f"Invalid canonical object_direction for {task}: {object_direction}")


def hazard_from_primitives(task: str, zone_relation: str, object_state: str, object_direction: Optional[str]) -> Tuple[str, str]:
    positive_label = str(TASK_CONFIGS[task]["positive_hazard_label"])
    if task == "forklift":
        present = "yes" if (zone_relation == "inside" and object_state == "moving" and object_direction == "towards") else "no"
    else:
        present = "yes" if (zone_relation == "inside" and object_state == "moving") else "no"
    return present, positive_label if present == "yes" else "no_hazard"


def validate_hazard_fields(task: str, hazard_present: str, hazard_label: str) -> None:
    allowed = TASK_CONFIGS[task]["allowed_values"]
    assert isinstance(allowed, dict)
    if hazard_present not in allowed["hazard_present"]:
        raise ValueError(f"Invalid hazard_present for {task}: {hazard_present}")
    if hazard_label not in allowed["hazard_label"]:
        raise ValueError(f"Invalid hazard_label for {task}: {hazard_label}")


def make_evidence(task: str, zone_relation: str, object_state: str, object_direction: Optional[str]) -> str:
    if task == "forklift":
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

    if zone_relation == "no_worker":
        return "No worker is visible in the machine workspace."
    workspace_phrase = "inside the machine workspace" if zone_relation == "inside" else "outside the machine workspace"
    if object_state == "moving":
        return f"A worker is visible {workspace_phrase} while the robot or machine is moving."
    return f"A worker is visible {workspace_phrase} while the robot or machine is stationary."


def full_tuple_key(frame: FrameLabel) -> Tuple[str, str, str, str, str, str]:
    return (
        frame.task,
        frame.zone_relation,
        frame.object_state,
        frame.object_direction or "",
        frame.hazard_present,
        frame.hazard_label,
    )


def label_signature_from_target(target: ClipTarget) -> str:
    parts = [target.task, target.hazard_label, target.hazard_present, target.zone_relation, target.object_state]
    if target.object_direction is not None:
        parts.append(target.object_direction)
    return "|".join(parts)


def classify_hard_negative(target: ClipTarget) -> Optional[str]:
    if target.hazard_present == "yes":
        return None
    if target.task == "forklift":
        if target.zone_relation == "no_forklift":
            return "no_forklift"
        if target.zone_relation == "inside" and target.object_state == "stationary":
            return "inside_stationary"
        if target.zone_relation == "inside" and target.object_state == "moving" and target.object_direction == "away":
            return "inside_moving_away"
        if target.zone_relation == "outside" and target.object_state == "moving" and target.object_direction == "towards":
            return "outside_moving_towards"
        return "other_no_hazard"
    if target.zone_relation == "no_worker":
        return "no_worker"
    if target.zone_relation == "inside" and target.object_state == "stationary":
        return "inside_stationary"
    if target.zone_relation == "outside" and target.object_state == "moving":
        return "outside_moving"
    if target.zone_relation == "outside" and target.object_state == "stationary":
        return "outside_stationary"
    return "other_no_hazard"


def important_hard_negative_buckets_for_samples(samples: Sequence[ClipSample]) -> Set[str]:
    buckets: Set[str] = set()
    for sample in samples:
        task_buckets = TASK_CONFIGS[sample.task]["important_hard_negative_buckets"]
        assert isinstance(task_buckets, set)
        buckets.update(task_buckets)
    return buckets

def get_video_metadata(video_path: Path) -> Dict[str, float]:
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            if fps > 0 and frame_count > 0:
                return {
                    "fps": fps,
                    "frame_count": frame_count,
                    "duration_sec": frame_count / fps,
                }
        else:
            cap.release()

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not read video metadata from {video_path} with cv2 or ffprobe.\n"
            f"STDERR:\n{result.stderr}"
        )

    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video streams reported for {video_path}")
    stream = streams[0]

    def parse_rate(value: object) -> float:
        if not value:
            return 0.0
        text = str(value)
        if "/" in text:
            num, den = text.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(text)

    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))
    duration_sec = float(stream.get("duration") or 0.0)
    frame_count_raw = stream.get("nb_frames")
    frame_count = int(frame_count_raw) if frame_count_raw not in {None, "N/A"} else 0
    if frame_count <= 0 and fps > 0 and duration_sec > 0:
        frame_count = int(round(fps * duration_sec))

    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Could not read valid FPS/frame count from: {video_path}")

    if duration_sec <= 0:
        duration_sec = frame_count / fps

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }


def find_pairs(data_dir: Path, task_mode: str) -> List[Tuple[str, Path, Path]]:
    videos: Dict[str, Tuple[str, Path]] = {}
    jsons: Dict[str, Tuple[str, Path]] = {}
    skipped_unrelated: List[str] = []
    wanted_tasks = selected_tasks(task_mode)

    for path in data_dir.iterdir():
        if not path.is_file():
            continue
        task = detect_task_from_stem(path.stem)
        if task is None:
            skipped_unrelated.append(path.name)
            continue
        if task not in wanted_tasks:
            continue
        if path.suffix.lower() in VIDEO_EXTS:
            videos[path.stem] = (task, path)
        elif path.suffix.lower() == ".json":
            jsons[path.stem] = (task, path)

    pairs: List[Tuple[str, Path, Path]] = []
    missing: List[str] = []
    for stem, (task, video_path) in sorted(videos.items()):
        ann_info = jsons.get(stem)
        if ann_info is None:
            missing.append(video_path.name)
            continue
        _, ann_path = ann_info
        pairs.append((task, video_path, ann_path))

    if skipped_unrelated:
        print(f"[INFO] Skipped {len(skipped_unrelated)} unrelated file(s) without a supported task prefix.")
    if missing:
        print("[WARN] Missing matching JSON annotation for videos:")
        for name in missing:
            print(f"  - {name}")
    if not pairs:
        raise SystemExit(f"No matching task-filtered (video, json) pairs found in {data_dir}")

    return pairs


def load_frame_labels(annotation_path: Path, duration_sec: float, recompute_hazard: bool, task: str) -> Tuple[List[FrameLabel], Dict[str, int]]:
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Annotation file must contain a list: {annotation_path}")

    processed = []
    seen_frame_idxs = set()
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            continue
        frame_idx = row.get("frame_idx", i)
        if frame_idx in seen_frame_idxs:
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

        object_state = normalize_object_state(task, metrics.get("Object State"))
        zone_relation = normalize_zone_relation(task, metrics.get("Danger Zone Relation"))
        object_direction = normalize_direction(task, metrics.get("Object Direction"))
        object_state, zone_relation, object_direction = canonicalize_primitives(task, object_state, zone_relation, object_direction)
        validate_primitives(task, object_state, zone_relation, object_direction)

        ann_hazard_present = normalize_yes_no(metrics.get("Hazard Detected", "no"))
        ann_hazard_label = normalize_hazard_label(task, metrics.get("Hazard Label", "no_hazard"))
        recomputed_present, recomputed_label = hazard_from_primitives(task, zone_relation, object_state, object_direction)

        if (ann_hazard_present, ann_hazard_label) != (recomputed_present, recomputed_label):
            stats["recomputed_hazard_mismatch_frames"] += 1

        if recompute_hazard:
            hazard_present, hazard_label = recomputed_present, recomputed_label
        else:
            hazard_present, hazard_label = ann_hazard_present, ann_hazard_label
        validate_hazard_fields(task, hazard_present, hazard_label)

        frames.append(
            FrameLabel(
                task=task,
                frame_idx=frame_idx,
                time_sec=frame_idx / ann_fps,
                object_state=object_state,
                zone_relation=zone_relation,
                object_direction=object_direction,
                hazard_present=hazard_present,
                hazard_label=hazard_label,
                timestamp=row.get("Timestamp"),
            )
        )

    frames.sort(key=lambda f: (f.frame_idx, f.time_sec))
    stats["annotation_frames"] = len(frames)
    stats["annotation_count_derived_fps_x1000"] = int(round(ann_fps * 1000))
    stats["positive_frames"] = sum(1 for f in frames if f.hazard_present == "yes")
    return frames, dict(stats)


def select_frames_in_window(frames: Sequence[FrameLabel], start_sec: float, end_sec: float) -> List[FrameLabel]:
    return [frame for frame in frames if start_sec <= frame.time_sec < end_sec]


def build_target_from_anchor(anchor_frames: Sequence[FrameLabel], include_evidence: bool) -> Tuple[ClipTarget, float]:
    if not anchor_frames:
        raise ValueError("Anchor contains no frames.")

    counter = Counter(full_tuple_key(frame) for frame in anchor_frames)
    mode_tuple, mode_count = counter.most_common(1)[0]
    anchor_mode_fraction = mode_count / len(anchor_frames)

    task, zone_relation, object_state, object_direction, hazard_present, hazard_label = mode_tuple
    direction_value = object_direction or None
    evidence = make_evidence(task, zone_relation, object_state, direction_value) if include_evidence else None

    return ClipTarget(
        task=task,
        hazard_label=hazard_label,
        hazard_present=hazard_present,
        zone_relation=zone_relation,
        object_state=object_state,
        object_direction=direction_value,
        evidence=evidence,
    ), anchor_mode_fraction


def make_sample_id(video_id: str, clip_start_sec: float, clip_end_sec: float) -> str:
    s_ms = int(round(clip_start_sec * 1000))
    e_ms = int(round(clip_end_sec * 1000))
    return f"{video_id}__{s_ms:09d}_{e_ms:09d}"


def generate_clip_samples(*, task: str, video_id: str, video_path: Path, annotation_path: Path, frames: Sequence[FrameLabel], duration_sec: float, clip_sec: float, stride_sec: float, anchor_sec: float, anchor_consensus_thr: float, include_evidence: bool) -> List[ClipSample]:
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
        positive_fraction = sum(1 for frame in clip_frames if frame.hazard_present == "yes") / len(clip_frames)
        unique_states_in_clip = len(set(full_tuple_key(frame) for frame in clip_frames))
        is_transition = unique_states_in_clip > 1
        is_ambiguous = anchor_mode_fraction < anchor_consensus_thr
        hard_negative_bucket = classify_hard_negative(target)

        samples.append(
            ClipSample(
                sample_id=make_sample_id(video_id, clip_start, clip_end),
                task=task,
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
        )

    return samples


def dumps_jsonl_line(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(dumps_jsonl_line(row) + "\n")


def clip_sample_to_manifest_row(sample: ClipSample, include_evidence: bool) -> Dict:
    return {
        "sample_id": sample.sample_id,
        "task": sample.task,
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


def clip_sample_to_chat_row(sample: ClipSample, prompt_texts: Dict[str, str]) -> Dict:
    assistant_json = sample.target.to_json_dict(include_evidence=False)
    video_ref = sample.clip_path if sample.clip_path else sample.source_video_path

    return {
        "sample_id": sample.sample_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_ref},
                    {"type": "text", "text": prompt_texts[sample.task]},
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
            "task": sample.task,
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

def js_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    keys = set(p.keys()) | set(q.keys())
    m = {}
    for key in keys:
        pv = p.get(key, 0.0) + eps
        qv = q.get(key, 0.0) + eps
        m[key] = 0.5 * (pv + qv)

    def kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        total = 0.0
        for key in keys:
            av = a.get(key, 0.0) + eps
            bv = b.get(key, 0.0) + eps
            total += av * math.log(av / bv)
        return total

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def normalize_counter(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def summarize_samples(samples: Sequence[ClipSample]) -> Dict:
    total = len(samples)
    stable = [sample for sample in samples if not sample.is_ambiguous]
    positives = [sample for sample in samples if sample.target.hazard_present == "yes"]
    stable_positives = [sample for sample in stable if sample.target.hazard_present == "yes"]

    return {
        "num_samples": total,
        "num_stable": len(stable),
        "num_ambiguous": total - len(stable),
        "num_positive": len(positives),
        "num_positive_stable": len(stable_positives),
        "task_distribution": dict(Counter(sample.task for sample in samples)),
        "hazard_present_distribution": dict(Counter(sample.target.hazard_present for sample in samples)),
        "label_signature_distribution": dict(Counter(sample.label_signature for sample in samples)),
        "hard_negative_distribution": dict(Counter(sample.hard_negative_bucket for sample in samples if sample.hard_negative_bucket)),
    }


def summarize_samples_by_task(samples: Sequence[ClipSample]) -> Dict[str, Dict]:
    by_task: Dict[str, List[ClipSample]] = defaultdict(list)
    for sample in samples:
        by_task[sample.task].append(sample)
    return {task: summarize_samples(task_samples) for task, task_samples in sorted(by_task.items())}


def _assign_group_to_splits(clips: List[ClipSample], val_ratio: float, test_ratio: float, guard_clips: int, val_ids: set, test_ids: set) -> None:
    n = len(clips)
    if n == 0:
        return

    n_val = max(1, round(n * val_ratio)) if n >= 3 else 0
    val_positions = set(itertools.islice(range(0, n, max(1, n // n_val)), n_val)) if n_val > 0 else set()

    guarded: set = set()
    for pos in val_positions:
        for guarded_idx in range(max(0, pos - guard_clips), min(n, pos + guard_clips + 1)):
            if guarded_idx not in val_positions:
                guarded.add(guarded_idx)

    eligible = [idx for idx in range(n) if idx not in val_positions and idx not in guarded]
    n_test = max(1, round(n * test_ratio)) if eligible else 0
    test_positions = set(itertools.islice(eligible[::max(1, len(eligible) // n_test)], n_test)) if n_test > 0 else set()

    for idx, clip in enumerate(clips):
        if idx in val_positions:
            val_ids.add(clip.sample_id)
        elif idx in test_positions:
            test_ids.add(clip.sample_id)


def choose_clip_level_stratified_split(samples: Sequence[ClipSample], val_ratio: float, test_ratio: float, guard_clips: int, positive_val_ratio: float, positive_test_ratio: float, positive_guard_clips: int) -> Tuple[set, set, Dict]:
    stable = [sample for sample in samples if not sample.is_ambiguous]
    by_video: Dict[str, List[ClipSample]] = defaultdict(list)
    for sample in stable:
        by_video[sample.source_video_id].append(sample)

    val_ids: set = set()
    test_ids: set = set()

    for video_id in sorted(by_video):
        clips_sorted = sorted(by_video[video_id], key=lambda sample: sample.clip_start_sec)
        positives = [sample for sample in clips_sorted if sample.target.hazard_present == "yes"]
        negatives = [sample for sample in clips_sorted if sample.target.hazard_present == "no"]

        _assign_group_to_splits(positives, positive_val_ratio, positive_test_ratio, positive_guard_clips, val_ids, test_ids)
        _assign_group_to_splits(negatives, val_ratio, test_ratio, guard_clips, val_ids, test_ids)

    train_s = [sample for sample in stable if sample.sample_id not in val_ids and sample.sample_id not in test_ids]
    val_s = [sample for sample in stable if sample.sample_id in val_ids]
    test_s = [sample for sample in stable if sample.sample_id in test_ids]
    total = len(stable)

    return val_ids, test_ids, {
        "strategy": "clip_level_stratified",
        "guard_clips_negatives": guard_clips,
        "guard_clips_positives": positive_guard_clips,
        "positive_val_ratio_target": positive_val_ratio,
        "positive_test_ratio_target": positive_test_ratio,
        "val_ratio_actual": len(val_s) / max(total, 1),
        "test_ratio_actual": len(test_s) / max(total, 1),
        "train_num_stable": len(train_s),
        "val_num_stable": len(val_s),
        "test_num_stable": len(test_s),
        "train_positive": sum(1 for sample in train_s if sample.target.hazard_present == "yes"),
        "val_positive": sum(1 for sample in val_s if sample.target.hazard_present == "yes"),
        "test_positive": sum(1 for sample in test_s if sample.target.hazard_present == "yes"),
        "train_videos": sorted({sample.source_video_id for sample in train_s}),
        "val_videos": sorted({sample.source_video_id for sample in val_s}),
        "test_videos": sorted({sample.source_video_id for sample in test_s}),
        "train_hazard_present_distribution": dict(Counter(sample.target.hazard_present for sample in train_s)),
        "val_hazard_present_distribution": dict(Counter(sample.target.hazard_present for sample in val_s)),
        "test_hazard_present_distribution": dict(Counter(sample.target.hazard_present for sample in test_s)),
        "per_video_split_counts": {
            video_id: {
                "task": by_video[video_id][0].task,
                "train": sum(1 for sample in by_video[video_id] if sample.sample_id not in val_ids and sample.sample_id not in test_ids and not sample.is_ambiguous),
                "val": sum(1 for sample in by_video[video_id] if sample.sample_id in val_ids),
                "test": sum(1 for sample in by_video[video_id] if sample.sample_id in test_ids),
            }
            for video_id in sorted(by_video)
        },
        "per_task_split_counts": {
            task: {
                "train": sum(1 for sample in train_s if sample.task == task),
                "val": sum(1 for sample in val_s if sample.task == task),
                "test": sum(1 for sample in test_s if sample.task == task),
            }
            for task in sorted({sample.task for sample in stable})
        },
    }


def _score_split_subset(subset_samples: List[ClipSample], remainder_samples: List[ClipSample], total_stable: int, target_ratio: float, global_hazard_dist: Dict[str, float], global_sig_dist: Dict[str, float], global_hn_dist: Dict[str, float], expected_hn_buckets: Set[str]) -> float:
    ratio_actual = len(subset_samples) / max(total_stable, 1)
    sub_hazard_dist = normalize_counter(Counter(sample.target.hazard_present for sample in subset_samples))
    sub_sig_dist = normalize_counter(Counter(sample.label_signature for sample in subset_samples))
    sub_hn_dist = normalize_counter(Counter(sample.hard_negative_bucket for sample in subset_samples if sample.hard_negative_bucket))

    score = 0.0
    score += 2.0 * abs(ratio_actual - target_ratio)
    score += 1.5 * js_divergence(global_hazard_dist, sub_hazard_dist)
    score += 1.0 * js_divergence(global_sig_dist, sub_sig_dist)
    if global_hn_dist:
        score += 0.5 * js_divergence(global_hn_dist, sub_hn_dist)

    sub_pos = sum(1 for sample in subset_samples if sample.target.hazard_present == "yes")
    rem_pos = sum(1 for sample in remainder_samples if sample.target.hazard_present == "yes")
    if sub_pos == 0:
        score += 5.0
    if rem_pos == 0:
        score += 20.0

    sub_hn_buckets = {sample.hard_negative_bucket for sample in subset_samples if sample.hard_negative_bucket}
    if expected_hn_buckets:
        score += 0.2 * len(expected_hn_buckets - sub_hn_buckets)
    return score


def choose_video_level_three_way_split(samples: Sequence[ClipSample], val_ratio: float, test_ratio: float, min_val_videos: int, max_val_videos: Optional[int], min_test_videos: int, max_test_videos: Optional[int]) -> Tuple[set, set, Dict]:
    stable_samples = [sample for sample in samples if not sample.is_ambiguous]
    by_video: Dict[str, List[ClipSample]] = defaultdict(list)
    for sample in stable_samples:
        by_video[sample.source_video_id].append(sample)

    videos = sorted(by_video)
    if len(videos) < 3:
        raise ValueError(
            f"Need at least 3 source videos for a train/val/test split (found {len(videos)}). "
            "Lower --min-val-videos / --min-test-videos or add more source videos."
        )

    if max_test_videos is None:
        max_test_videos = max(min_test_videos, len(videos) - 2)
    max_test_videos = min(max_test_videos, len(videos) - 2)

    if max_val_videos is None:
        max_val_videos = max(min_val_videos, len(videos) - 2)

    global_hazard_dist = normalize_counter(Counter(sample.target.hazard_present for sample in stable_samples))
    global_sig_dist = normalize_counter(Counter(sample.label_signature for sample in stable_samples))
    global_hn_dist = normalize_counter(Counter(sample.hard_negative_bucket for sample in stable_samples if sample.hard_negative_bucket))
    expected_hn_buckets = important_hard_negative_buckets_for_samples(stable_samples)
    total_stable = len(stable_samples)

    best_test_subset: Optional[set] = None
    best_test_score = float("inf")
    for r in range(min_test_videos, max_test_videos + 1):
        for subset in itertools.combinations(videos, r):
            subset_set = set(subset)
            test_s = [sample for sample in stable_samples if sample.source_video_id in subset_set]
            remainder_s = [sample for sample in stable_samples if sample.source_video_id not in subset_set]
            if not test_s or len(remainder_s) < min_val_videos + 1:
                continue
            score = _score_split_subset(test_s, remainder_s, total_stable, test_ratio, global_hazard_dist, global_sig_dist, global_hn_dist, expected_hn_buckets)
            if score < best_test_score:
                best_test_score = score
                best_test_subset = subset_set

    if best_test_subset is None:
        raise RuntimeError("Failed to find a valid video-level test split.")

    remaining_videos = sorted(set(videos) - best_test_subset)
    remaining_stable = [sample for sample in stable_samples if sample.source_video_id in set(remaining_videos)]
    rem_hazard_dist = normalize_counter(Counter(sample.target.hazard_present for sample in remaining_stable))
    rem_sig_dist = normalize_counter(Counter(sample.label_signature for sample in remaining_stable))
    rem_hn_dist = normalize_counter(Counter(sample.hard_negative_bucket for sample in remaining_stable if sample.hard_negative_bucket))
    rem_expected_hn_buckets = important_hard_negative_buckets_for_samples(remaining_stable)
    total_remaining = len(remaining_stable)
    effective_val_ratio = val_ratio / max(1.0 - float(test_ratio), 1e-6)

    best_val_subset: Optional[set] = None
    best_val_score = float("inf")
    effective_max_val = min(max_val_videos, len(remaining_videos) - 1)
    for r in range(min_val_videos, effective_max_val + 1):
        for subset in itertools.combinations(remaining_videos, r):
            subset_set = set(subset)
            val_s = [sample for sample in remaining_stable if sample.source_video_id in subset_set]
            train_s = [sample for sample in remaining_stable if sample.source_video_id not in subset_set]
            if not val_s or not train_s:
                continue
            score = _score_split_subset(val_s, train_s, total_remaining, effective_val_ratio, rem_hazard_dist, rem_sig_dist, rem_hn_dist, rem_expected_hn_buckets)
            if score < best_val_score:
                best_val_score = score
                best_val_subset = subset_set

    if best_val_subset is None:
        raise RuntimeError("Failed to find a valid video-level val split.")

    test_s = [sample for sample in stable_samples if sample.source_video_id in best_test_subset]
    val_s = [sample for sample in stable_samples if sample.source_video_id in best_val_subset]
    train_s = [sample for sample in stable_samples if sample.source_video_id not in best_test_subset | best_val_subset]

    return best_val_subset, best_test_subset, {
        "strategy": "video_level",
        "test_score": best_test_score,
        "val_score": best_val_score,
        "test_ratio_actual": len(test_s) / max(total_stable, 1),
        "val_ratio_actual": len(val_s) / max(total_stable, 1),
        "train_num_stable": len(train_s),
        "val_num_stable": len(val_s),
        "test_num_stable": len(test_s),
        "train_positive": sum(1 for sample in train_s if sample.target.hazard_present == "yes"),
        "val_positive": sum(1 for sample in val_s if sample.target.hazard_present == "yes"),
        "test_positive": sum(1 for sample in test_s if sample.target.hazard_present == "yes"),
        "train_videos": sorted(set(videos) - best_test_subset - best_val_subset),
        "val_videos": sorted(best_val_subset),
        "test_videos": sorted(best_test_subset),
        "train_hazard_present_distribution": dict(Counter(sample.target.hazard_present for sample in train_s)),
        "val_hazard_present_distribution": dict(Counter(sample.target.hazard_present for sample in val_s)),
        "test_hazard_present_distribution": dict(Counter(sample.target.hazard_present for sample in test_s)),
        "per_task_split_counts": {
            task: {
                "train": sum(1 for sample in train_s if sample.task == task),
                "val": sum(1 for sample in val_s if sample.task == task),
                "test": sum(1 for sample in test_s if sample.task == task),
            }
            for task in sorted({sample.task for sample in stable_samples})
        },
    }


def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_clip_ffmpeg(src_video: Path, dst_clip: Path, start_sec: float, end_sec: float) -> None:
    ensure_dir(dst_clip.parent)
    duration = max(end_sec - start_sec, 0.001)
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start_sec:.6f}", "-i", str(src_video),
        "-t", f"{duration:.6f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst_clip),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src_video.name} -> {dst_clip.name}\nSTDERR:\n{result.stderr}"
        )


def stable_keep_score(sample_id: str) -> float:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def maybe_downsample_easy_negatives_train(train_samples: List[ClipSample], keep_prob: float) -> List[ClipSample]:
    if keep_prob >= 0.999:
        return train_samples
    keep_prob = max(0.0, min(1.0, keep_prob))

    easy_buckets = {task: str(config["easy_negative_bucket"]) for task, config in TASK_CONFIGS.items()}
    easy_samples = [sample for sample in train_samples if sample.hard_negative_bucket == easy_buckets[sample.task] and sample.target.hazard_present == "no"]
    others = [sample for sample in train_samples if not (sample.hard_negative_bucket == easy_buckets[sample.task] and sample.target.hazard_present == "no")]

    retained = [sample for sample in easy_samples if stable_keep_score(sample.sample_id) < keep_prob]
    return sorted(others + retained, key=lambda sample: (sample.task, sample.source_video_id, sample.clip_start_sec))

def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {data_dir}")
    if args.extract_clips and not ffmpeg_exists():
        raise SystemExit("ffmpeg is required but was not found on PATH.")

    prompt_texts = load_prompt_texts(args)
    pairs = find_pairs(data_dir, args.task_mode)

    all_samples: List[ClipSample] = []
    per_video_report = {}

    print(f"Found {len(pairs)} task-filtered (video, json) pair(s).")

    for task, video_path, ann_path in pairs:
        video_id = video_path.stem
        meta = get_video_metadata(video_path)
        frames, ann_stats = load_frame_labels(
            annotation_path=ann_path,
            duration_sec=meta["duration_sec"],
            recompute_hazard=args.recompute_hazard,
            task=task,
        )

        # Keep the clip/anchor workflow unchanged; only labels and prompts branch by task.
        samples = generate_clip_samples(
            task=task,
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
            "task": task,
            "video_path": str(video_path),
            "annotation_path": str(ann_path),
            "video_meta": meta,
            "annotation_stats": ann_stats,
            "clip_summary": summarize_samples(samples),
        }

        print(
            f"[OK] {safe_console_text(video_id)} ({task}): duration={meta['duration_sec']:.2f}s, "
            f"ann_frames={ann_stats['annotation_frames']}, clips={len(samples)}, "
            f"stable={sum(not sample.is_ambiguous for sample in samples)}, "
            f"positive={sum(sample.target.hazard_present == 'yes' for sample in samples)}"
        )

    if not all_samples:
        raise SystemExit("No clip samples were generated.")

    if args.split_strategy == "clip":
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
        for sample in all_samples:
            if sample.sample_id in test_ids:
                sample.split = "test"
            elif sample.sample_id in val_ids:
                sample.split = "val"
            else:
                sample.split = "train"
    else:
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
        for sample in all_samples:
            if sample.source_video_id in chosen_test_videos:
                sample.split = "test"
            elif sample.source_video_id in chosen_val_videos:
                sample.split = "val"
            else:
                sample.split = "train"

    exportable = all_samples if args.keep_ambiguous else [sample for sample in all_samples if not sample.is_ambiguous]
    train_samples = [sample for sample in exportable if sample.split == "train"]
    val_samples = [sample for sample in exportable if sample.split == "val"]
    test_samples = [sample for sample in exportable if sample.split == "test"]

    train_samples = maybe_downsample_easy_negatives_train(train_samples, keep_prob=args.easy_negative_keep_prob)

    if args.extract_clips:
        clips_root = out_dir / "clips"
        for sample in train_samples + val_samples + test_samples:
            split_dir = clips_root / sample.split / sample.source_video_id
            dst_clip = split_dir / f"{sample.sample_id}.mp4"
            extract_clip_ffmpeg(
                src_video=Path(sample.source_video_path),
                dst_clip=dst_clip,
                start_sec=sample.clip_start_sec,
                end_sec=sample.clip_end_sec,
            )
            sample.clip_path = str(dst_clip)

    all_manifest_path = out_dir / "all_clips_manifest.jsonl"
    train_manifest_path = out_dir / "train_manifest.jsonl"
    val_manifest_path = out_dir / "val_manifest.jsonl"
    test_manifest_path = out_dir / "test_manifest.jsonl"
    train_chat_path = out_dir / "train_chat.jsonl"
    val_chat_path = out_dir / "val_chat.jsonl"
    test_chat_path = out_dir / "test_chat.jsonl"
    split_report_path = out_dir / "split_report.json"

    write_jsonl(all_manifest_path, [clip_sample_to_manifest_row(sample, include_evidence=args.include_evidence) for sample in all_samples])
    write_jsonl(train_manifest_path, [clip_sample_to_manifest_row(sample, include_evidence=args.include_evidence) for sample in train_samples])
    write_jsonl(val_manifest_path, [clip_sample_to_manifest_row(sample, include_evidence=args.include_evidence) for sample in val_samples])
    write_jsonl(test_manifest_path, [clip_sample_to_manifest_row(sample, include_evidence=args.include_evidence) for sample in test_samples])
    write_jsonl(train_chat_path, [clip_sample_to_chat_row(sample, prompt_texts=prompt_texts) for sample in train_samples])
    write_jsonl(val_chat_path, [clip_sample_to_chat_row(sample, prompt_texts=prompt_texts) for sample in val_samples])
    write_jsonl(test_chat_path, [clip_sample_to_chat_row(sample, prompt_texts=prompt_texts) for sample in test_samples])

    report = {
        "config": {
            "data_dir": str(data_dir),
            "task_mode": args.task_mode,
            "fork_prompt_file": args.fork_prompt_file,
            "robot_prompt_file": args.robot_prompt_file,
            "out_dir": str(out_dir),
            "clip_sec": args.clip_sec,
            "stride_sec": args.stride_sec,
            "anchor_sec": args.anchor_sec,
            "anchor_consensus_thr": args.anchor_consensus_thr,
            "val_ratio_target": args.val_ratio,
            "test_ratio_target": args.test_ratio,
            "include_evidence_manifest": args.include_evidence,
            "recompute_hazard": args.recompute_hazard,
            "keep_ambiguous_in_export": args.keep_ambiguous,
            "easy_negative_keep_prob_train": args.easy_negative_keep_prob,
            "extract_clips": args.extract_clips,
        },
        "overall_summary_all": summarize_samples(all_samples),
        "overall_summary_export_train": summarize_samples(train_samples),
        "overall_summary_export_val": summarize_samples(val_samples),
        "overall_summary_export_test": summarize_samples(test_samples),
        "per_task_summary_all": summarize_samples_by_task(all_samples),
        "per_task_summary_export_train": summarize_samples_by_task(train_samples),
        "per_task_summary_export_val": summarize_samples_by_task(val_samples),
        "per_task_summary_export_test": summarize_samples_by_task(test_samples),
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

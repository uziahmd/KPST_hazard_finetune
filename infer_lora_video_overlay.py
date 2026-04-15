#!/usr/bin/env python3
"""
End-to-end video inference pipeline for a LoRA fine-tuned multimodal hazard model.

Pipeline:
1. Discover task-specific videos from a directory.
2. Split each video into consecutive fixed-length chunks with ffmpeg.
3. Load the base model and LoRA adapter once.
4. Run multimodal generation on each chunk with the correct task prompt.
5. Save structured inference results to JSON.
6. Render per-chunk overlays with ffmpeg.
7. Concatenate annotated chunks back into final videos.

run:
python infer_lora_video_overlay.py \
  --base_model Qwen/Qwen3.5-9B \
  --adapter_dir runs/qwen35_9b_both_aug/checkpoint-4796 \
  --video_dir test \
  --output_dir runs/infer_overlay_run \
  --task_mode both \
  --robot_prompt_file prompts/robot_propmt_v1.txt \
  --fork_prompt_file prompts/fork_prompt_v2.txt \
  --chunk_sec 5 \
  --num_frames 12 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --no-do_sample \
  --device cuda:0 \
  --save_chunks \
  --save_overlay
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv

IMPORT_ERROR: Optional[Exception] = None
np = None
torch = None
PeftModel = None
AutoProcessor = None
AutoModelForImageTextToText = None
AutoModelForMultimodalLM = None

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
TASK_PREFIX_MAP = {
    "fork_": "forklift",
    "robot_": "robot",
}
TASK_REQUIRED_FIELDS = {
    "robot": [
        "hazard_label",
        "hazard_present",
        "zone_relation",
        "object_state",
    ],
    "forklift": [
        "hazard_label",
        "hazard_present",
        "zone_relation",
        "object_state",
        "object_direction",
    ],
}


def ensure_ml_dependencies() -> None:
    global IMPORT_ERROR, np, torch, PeftModel, AutoProcessor, AutoModelForImageTextToText, AutoModelForMultimodalLM

    if torch is not None and np is not None and PeftModel is not None and AutoProcessor is not None:
        return

    try:
        import numpy as _np
        import torch as _torch
        from peft import PeftModel as _PeftModel
        from transformers import AutoModelForImageTextToText as _AutoModelForImageTextToText
        from transformers import AutoProcessor as _AutoProcessor

        try:
            from transformers import AutoModelForMultimodalLM as _AutoModelForMultimodalLM
        except ImportError:
            _AutoModelForMultimodalLM = None
    except Exception as exc:
        IMPORT_ERROR = exc
        raise ImportError(
            "Missing runtime dependencies. Install torch, transformers, peft, and numpy before running inference."
        ) from exc

    np = _np
    torch = _torch
    PeftModel = _PeftModel
    AutoProcessor = _AutoProcessor
    AutoModelForImageTextToText = _AutoModelForImageTextToText
    AutoModelForMultimodalLM = _AutoModelForMultimodalLM


def initialize_environment(logger: Optional[logging.Logger] = None) -> None:
    load_dotenv(override=True)
    if logger is None:
        return

    if os.getenv("HF_TOKEN"):
        logger.info("HF_TOKEN successfully loaded from environment/.env")
    else:
        logger.warning("HF_TOKEN not found in environment/.env")


def default_overlay_workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoRA video inference, JSON export, and overlay rendering on a folder of hazard videos."
    )
    parser.add_argument("--base_model", type=str, required=True, help="Base multimodal model name or local path.")
    parser.add_argument("--adapter_dir", type=str, required=True, help="LoRA adapter directory.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing test videos.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for chunks, results, logs, and overlays.")
    parser.add_argument(
        "--task_mode",
        type=str,
        default="both",
        choices=["robot", "forklift", "both"],
        help="Which task videos to process.",
    )
    parser.add_argument(
        "--robot_prompt_file",
        type=str,
        default="prompts/robot_propmt_v1.txt",
        help="Prompt file for the robot task.",
    )
    parser.add_argument(
        "--fork_prompt_file",
        type=str,
        default="prompts/fork_prompt_v2.txt",
        help="Prompt file for the forklift task.",
    )
    parser.add_argument("--chunk_sec", type=float, default=5.0, help="Chunk length in seconds.")
    parser.add_argument("--num_frames", type=int, default=12, help="Frames sampled per chunk by the processor.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum generated tokens per chunk.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable stochastic generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--save_chunks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep intermediate chunk mp4 files after the run.",
    )
    parser.add_argument(
        "--save_overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep intermediate annotated chunk mp4 files after concatenation.",
    )
    parser.add_argument(
        "--font_file",
        type=str,
        default="",
        help="Optional font file for ffmpeg drawtext. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--overlay_workers",
        type=int,
        default=default_overlay_workers(),
        help="Number of parallel worker processes for overlay rendering.",
    )
    parser.add_argument(
        "--skip_inference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip chunking and model inference and reuse an existing results JSON.",
    )
    parser.add_argument(
        "--skip_overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip overlay rendering and final annotated video concatenation.",
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default="",
        help="Optional path for the results JSON. Defaults to <output_dir>/results/inference_results.json.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code through to Transformers model and processor loading.",
    )
    return parser.parse_args()


def ensure_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"Required executable not found on PATH: {name}")


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("infer_lora_video_overlay")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def run_cmd(cmd: Sequence[str], logger: Optional[logging.Logger] = None) -> subprocess.CompletedProcess[str]:
    if logger is not None:
        logger.info("Running command: %s", " ".join(cmd))
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def is_probably_url(path_str: str) -> bool:
    parsed = urlparse(path_str)
    return parsed.scheme in {"http", "https", "s3", "gs"}


def resolve_media_path(path_str: str, project_root: str, test_file_dir: str) -> str:
    if not path_str:
        return path_str

    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str

    if os.path.exists(path_str):
        return str(Path(path_str).resolve())

    candidates = [
        os.path.join(project_root, path_str),
        os.path.join(test_file_dir, path_str),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return str(Path(candidate).resolve())
    return path_str


def normalize_content_items(
    content: Any,
    example: Dict[str, Any],
    project_root: str,
    test_file_dir: str,
) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]

    normalized: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            normalized.append({"type": "text", "text": item})
            continue

        if not isinstance(item, dict):
            normalized.append({"type": "text", "text": str(item)})
            continue

        item_type = item.get("type")
        if item_type == "text":
            normalized.append({"type": "text", "text": item.get("text", "")})
        elif item_type == "video":
            video_path = resolve_media_path(item.get("video", ""), project_root, test_file_dir)
            normalized.append({"type": "video", "video": video_path})
        elif item_type == "image":
            image_path = resolve_media_path(item.get("image", ""), project_root, test_file_dir)
            normalized.append({"type": "image", "image": image_path})
        else:
            normalized.append({"type": "text", "text": json.dumps(item, ensure_ascii=False)})

    return normalized


def normalize_message(
    msg: Dict[str, Any],
    example: Dict[str, Any],
    project_root: str,
    test_file_dir: str,
) -> Dict[str, Any]:
    if "role" in msg:
        return {
            "role": msg["role"],
            "content": normalize_content_items(
                msg.get("content", []),
                example=example,
                project_root=project_root,
                test_file_dir=test_file_dir,
            ),
        }

    if "from" in msg:
        role_map = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "system": "system",
        }
        role = role_map.get(msg["from"], msg["from"])
        value = msg.get("value", "")

        content: List[Dict[str, Any]] = []
        if isinstance(value, str) and "<video>" in value and example.get("video"):
            video_path = resolve_media_path(example["video"], project_root, test_file_dir)
            content.append({"type": "video", "video": video_path})
            text_only = value.replace("<video>", "").strip()
            if text_only:
                content.append({"type": "text", "text": text_only})
        else:
            content = normalize_content_items(
                value,
                example=example,
                project_root=project_root,
                test_file_dir=test_file_dir,
            )

        return {"role": role, "content": content}

    raise ValueError(f"Unsupported message format: {msg}")


def detect_task_from_name(path: Path) -> Optional[str]:
    lower_name = path.name.lower()
    for prefix, task in TASK_PREFIX_MAP.items():
        if lower_name.startswith(prefix):
            return task
    return None


def should_include_task(task: str, task_mode: str) -> bool:
    return task_mode == "both" or task == task_mode


def discover_videos(video_dir: Path, task_mode: str, logger: logging.Logger) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for path in sorted(video_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        task = detect_task_from_name(path)
        if task is None:
            reason = "unknown_filename_prefix"
            logger.warning("Skipping %s: %s", path.name, reason)
            skipped.append({"path": str(path.resolve()), "reason": reason})
            continue

        if not should_include_task(task, task_mode):
            reason = f"filtered_by_task_mode:{task_mode}"
            logger.info("Skipping %s: %s", path.name, reason)
            skipped.append({"path": str(path.resolve()), "task": task, "reason": reason})
            continue

        selected.append({
            "path": path.resolve(),
            "name": path.name,
            "stem": path.stem,
            "task": task,
        })

    logger.info("Discovered %d matching videos and %d skipped files.", len(selected), len(skipped))
    return selected, skipped


@lru_cache(maxsize=None)
def ffprobe_json(video_path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,width,height,r_frame_rate,avg_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def ffprobe_duration(video_path: Path) -> float:
    data = ffprobe_json(video_path)
    fmt = data.get("format", {}) or {}
    duration = fmt.get("duration")
    if duration is None:
        raise RuntimeError(f"Could not read duration for {video_path}")
    value = float(duration)
    if value <= 0:
        raise RuntimeError(f"Video duration must be positive for {video_path}")
    return value


def ffprobe_size(video_path: Path) -> Tuple[int, int]:
    data = ffprobe_json(video_path)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return int(stream["width"]), int(stream["height"])
    raise RuntimeError(f"No video stream found in {video_path}")


def format_time_token(seconds: float) -> str:
    return f"{int(round(seconds)):06d}"


def format_time_label(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    mins, ms_rem = divmod(total_ms, 60_000)
    secs, ms = divmod(ms_rem, 1000)
    if ms == 0:
        return f"{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}.{ms:03d}"


def split_video_into_chunks(
    video_path: Path,
    chunk_root: Path,
    chunk_sec: float,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], float]:
    duration = ffprobe_duration(video_path)
    video_chunk_dir = chunk_root / video_path.stem
    video_chunk_dir.mkdir(parents=True, exist_ok=True)

    chunks: List[Dict[str, Any]] = []
    total_chunks = int(math.ceil(duration / chunk_sec))

    for chunk_index in range(total_chunks):
        start_sec = chunk_index * chunk_sec
        end_sec = min(duration, start_sec + chunk_sec)
        part_duration = max(0.0, end_sec - start_sec)
        if part_duration <= 0:
            continue

        chunk_name = f"{video_path.stem}__{format_time_token(start_sec)}_{format_time_token(end_sec)}.mp4"
        chunk_path = video_chunk_dir / chunk_name

        cmd = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{part_duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(chunk_path),
        ]
        result = run_cmd(cmd, logger=logger)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create chunk {chunk_path.name} from {video_path.name}: {result.stderr.strip()}"
            )

        chunks.append({
            "original_video_path": str(video_path.resolve()),
            "video_name": video_path.name,
            "video_stem": video_path.stem,
            "chunk_index": chunk_index,
            "start_time": round(start_sec, 3),
            "end_time": round(end_sec, 3),
            "duration": round(part_duration, 3),
            "chunk_path": str(chunk_path.resolve()),
        })

    return chunks, duration


def select_dtype(device_str: str) -> torch.dtype:
    if device_str == "cpu":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def normalize_device_arg(device_arg: str) -> str:
    device_arg = device_arg.strip().lower()
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def infer_model_device(model: torch.nn.Module, requested_device: str) -> torch.device:
    if requested_device == "auto":
        requested_device = normalize_device_arg(requested_device)

    if hasattr(model, "device") and isinstance(model.device, torch.device):
        return model.device

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(requested_device)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def load_model_and_processor(
    base_model: str,
    adapter_dir: str,
    device_arg: str,
    trust_remote_code: bool,
    logger: logging.Logger,
) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    dtype = select_dtype(normalize_device_arg(device_arg))
    logger.info("Loading processor from %s", base_model)
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=trust_remote_code)

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
    }

    normalized_device = normalize_device_arg(device_arg)
    if device_arg == "auto" and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model_candidates = []
    if AutoModelForMultimodalLM is not None:
        model_candidates.append(("AutoModelForMultimodalLM", AutoModelForMultimodalLM))
    model_candidates.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))

    base_model_obj = None
    load_errors: List[str] = []
    for name, model_cls in model_candidates:
        try:
            logger.info("Trying model loader: %s", name)
            base_model_obj = model_cls.from_pretrained(base_model, **model_kwargs)
            logger.info("Loaded base model with %s", name)
            break
        except Exception as exc:
            load_errors.append(f"{name}: {exc}")

    if base_model_obj is None:
        raise RuntimeError("Unable to load base model. Errors:\n" + "\n".join(load_errors))

    if device_arg != "auto":
        base_model_obj = base_model_obj.to(torch.device(normalized_device))

    logger.info("Applying adapter from %s", adapter_dir)
    model = PeftModel.from_pretrained(base_model_obj, adapter_dir, is_trainable=False)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    actual_device = infer_model_device(model, device_arg)
    model_info = {
        "requested_device": device_arg,
        "resolved_device": str(actual_device),
        "dtype": str(dtype),
        "model_class": model.__class__.__name__,
        "processor_class": processor.__class__.__name__,
        "base_model": base_model,
        "adapter_dir": str(Path(adapter_dir).resolve()),
        "hf_device_map": getattr(model, "hf_device_map", None),
    }
    return model, processor, model_info


def _sample_indices(
    total_frames: int,
    source_fps: Optional[float],
    num_frames: Optional[int],
    fps: Optional[float],
) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("Decoded video contains no frames.")

    if fps is not None and source_fps is not None and source_fps > 0:
        step = max(source_fps / fps, 1.0)
        sampled = np.arange(0, total_frames, step, dtype=float)
        indices = np.clip(np.round(sampled).astype(int), 0, total_frames - 1)
        if indices.size == 0:
            indices = np.array([0], dtype=int)
        elif indices[-1] != total_frames - 1:
            indices = np.concatenate([indices, np.array([total_frames - 1], dtype=int)])
        indices = np.unique(indices)
    else:
        indices = np.arange(total_frames, dtype=int)

    if num_frames is not None and len(indices) > num_frames:
        positions = np.linspace(0, len(indices) - 1, num_frames, dtype=int)
        indices = indices[positions]

    return indices


def _decode_video_torchvision(video_path: Path) -> Tuple[np.ndarray, Optional[float]]:
    from torchvision.io import read_video

    try:
        video, _, info = read_video(str(video_path), pts_unit="sec", output_format="THWC")
    except TypeError:
        video, _, info = read_video(str(video_path), pts_unit="sec")

    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    else:
        video = np.asarray(video)

    if video.ndim != 4:
        raise ValueError(f"Expected decoded video to have 4 dims, got shape {video.shape} for {video_path}")

    if video.shape[-1] not in (1, 3) and video.shape[1] in (1, 3):
        video = np.transpose(video, (0, 2, 3, 1))

    source_fps = info.get("video_fps") if isinstance(info, dict) else None
    if source_fps is not None:
        source_fps = float(source_fps)
    return video, source_fps


def _decode_video_pyav(video_path: Path) -> Tuple[np.ndarray, Optional[float]]:
    import av

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate) if stream.average_rate is not None else None
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    finally:
        container.close()

    if not frames:
        raise ValueError(f"No video frames decoded from {video_path}")

    return np.stack(frames), source_fps


def decode_local_video(
    video_path: Path,
    *,
    num_frames: Optional[int],
    fps: Optional[float],
    backend: str,
) -> np.ndarray:
    if backend == "torchvision":
        video, source_fps = _decode_video_torchvision(video_path)
    elif backend == "pyav":
        video, source_fps = _decode_video_pyav(video_path)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")

    indices = _sample_indices(video.shape[0], source_fps, num_frames, fps)
    return np.ascontiguousarray(video[indices])


def choose_video_backend() -> str:
    try:
        import torchvision  # noqa: F401

        return "torchvision"
    except Exception:
        pass

    try:
        import av  # noqa: F401

        return "pyav"
    except Exception:
        pass

    raise RuntimeError("No local video backend available. Install torchvision or av/PyAV.")


def materialize_local_videos(
    messages: List[Dict[str, Any]],
    *,
    num_frames: Optional[int],
    backend: str,
) -> List[Dict[str, Any]]:
    messages = copy.deepcopy(messages)

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict) or item.get("type") != "video":
                continue

            raw_video = item.get("video")
            if isinstance(raw_video, str) and raw_video and not is_probably_url(raw_video):
                video_path = Path(raw_video)
                if not video_path.exists():
                    raise FileNotFoundError(f"Video file not found: {video_path}")
                item["video"] = decode_local_video(
                    video_path,
                    num_frames=num_frames,
                    fps=None,
                    backend=backend,
                )

    return messages


def apply_chat_template_with_fallback(
    processor: Any,
    messages: List[Dict[str, Any]],
    num_frames: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    processor_kwargs: Dict[str, Any] = {
        "padding": False,
        "truncation": False,
        "num_frames": num_frames,
        "fps": None,
    }
    base_kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }

    attempts.append({"mode": "path", "kwargs": {**base_kwargs, "enable_thinking": False, "processor_kwargs": processor_kwargs}})
    attempts.append({"mode": "path", "kwargs": {**base_kwargs, "processor_kwargs": processor_kwargs}})
    attempts.append({"mode": "path", "kwargs": {**base_kwargs, "enable_thinking": False}})
    attempts.append({"mode": "path", "kwargs": base_kwargs})

    path_errors: List[str] = []
    for attempt in attempts:
        try:
            return processor.apply_chat_template(messages, **attempt["kwargs"]), {
                "template_mode": attempt["mode"],
                "video_backend": None,
            }
        except Exception as exc:
            path_errors.append(str(exc))

    backend = choose_video_backend()
    materialized = materialize_local_videos(messages, num_frames=num_frames, backend=backend)

    decode_attempts: List[Dict[str, Any]] = []
    decode_attempts.append({**base_kwargs, "enable_thinking": False, "processor_kwargs": {"num_frames": num_frames}})
    decode_attempts.append({**base_kwargs, "processor_kwargs": {"num_frames": num_frames}})
    decode_attempts.append({**base_kwargs, "enable_thinking": False})
    decode_attempts.append(base_kwargs)

    decode_errors: List[str] = []
    for kwargs in decode_attempts:
        try:
            return processor.apply_chat_template(materialized, **kwargs), {
                "template_mode": "materialized_video",
                "video_backend": backend,
            }
        except Exception as exc:
            decode_errors.append(str(exc))

    error_text = "\n".join(["Path mode errors:"] + path_errors + ["Materialized video errors:"] + decode_errors)
    raise RuntimeError(f"Unable to apply chat template for multimodal video inference.\n{error_text}")


def build_user_messages(chunk_path: Path, prompt_text: str, project_root: Path) -> List[Dict[str, Any]]:
    raw_message = {
        "role": "user",
        "content": [
            {"type": "video", "video": str(chunk_path.resolve())},
            {"type": "text", "text": prompt_text},
        ],
    }
    normalized = normalize_message(
        raw_message,
        example={"video": str(chunk_path.resolve())},
        project_root=str(project_root.resolve()),
        test_file_dir=str(chunk_path.parent.resolve()),
    )
    return [normalized]


def run_inference_single(
    model: torch.nn.Module,
    processor: Any,
    messages: List[Dict[str, Any]],
    num_frames: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    device_arg: str,
) -> Tuple[str, Dict[str, Any]]:
    inputs, template_meta = apply_chat_template_with_fallback(processor, messages, num_frames=num_frames)
    device = infer_model_device(model, device_arg)
    inputs = move_batch_to_device(inputs, device)

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
    }
    if do_sample and temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    input_ids = inputs["input_ids"]
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
    pred_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return pred_text.strip(), template_meta


def extract_json_candidate(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start: idx + 1]
    return None


def try_parse_response_json(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(raw_text, str):
        return None, "raw_response_is_not_string"

    stripped = raw_text.strip()
    if not stripped:
        return None, "empty_response"

    candidates = [stripped]
    extracted = extract_json_candidate(stripped)
    if extracted and extracted != stripped:
        candidates.append(extracted)

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj, None

    return None, "no_valid_json_object_found"


def parse_chunk_response(task: str, raw_text: str) -> Dict[str, Any]:
    parsed_obj, parse_error = try_parse_response_json(raw_text)
    required_fields = TASK_REQUIRED_FIELDS[task]
    missing_required_fields: List[str] = []
    extra_fields: List[str] = []

    if parsed_obj is not None:
        missing_required_fields = [field for field in required_fields if field not in parsed_obj]
        extra_fields = sorted([field for field in parsed_obj.keys() if field not in required_fields])

    return {
        "parsed_response": parsed_obj,
        "parse_success": parsed_obj is not None,
        "parse_error": parse_error,
        "required_fields": required_fields,
        "missing_required_fields": missing_required_fields,
        "extra_fields": extra_fields,
        "required_fields_present": parsed_obj is not None and not missing_required_fields,
    }


def normalize_overlay_value(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_overlay_lines(chunk_result: Dict[str, Any]) -> Tuple[List[str], bool]:
    task = chunk_result["task"]
    fields = TASK_REQUIRED_FIELDS[task]
    parsed = chunk_result.get("parsed_response")
    lines = [
        f"time: {format_time_label(chunk_result['start_time'])} - {format_time_label(chunk_result['end_time'])}",
    ]

    if isinstance(parsed, dict):
        for field in fields:
            lines.append(f"{field}: {normalize_overlay_value(parsed.get(field))}")
    else:
        lines.append("parse_status: failed")
        for field in fields:
            lines.append(f"{field}: missing")

    hazard_present_yes = False
    if isinstance(parsed, dict):
        hazard_present_yes = str(parsed.get("hazard_present", "")).strip().lower() == "yes"

    return lines, hazard_present_yes


def detect_font_file(user_font: str = "") -> Path:
    if user_font:
        font_path = Path(user_font)
        if not font_path.exists():
            raise FileNotFoundError(f"Font file not found: {font_path}")
        return font_path.resolve()

    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError("Could not auto-detect a usable font file for ffmpeg drawtext.")


def compute_overlay_layout(width: int, height: int, num_lines: int) -> Dict[str, int]:
    margin = max(18, int(round(min(width, height) * 0.02)))
    padding = max(12, int(round(min(width, height) * 0.012)))
    panel_w = min(int(round(width * 0.42)), width - 2 * margin)
    fontsize = max(18, min(30, int(round(height * 0.026))))
    line_spacing = max(4, int(round(fontsize * 0.28)))
    line_h = fontsize + line_spacing
    panel_h = padding * 2 + num_lines * line_h
    panel_x = width - margin - panel_w
    panel_y = height - margin - panel_h
    text_x = panel_x + padding
    text_y = panel_y + padding + fontsize

    return {
        "margin": margin,
        "padding": padding,
        "panel_w": panel_w,
        "panel_h": panel_h,
        "fontsize": fontsize,
        "line_spacing": line_spacing,
        "line_h": line_h,
        "panel_x": panel_x,
        "panel_y": panel_y,
        "text_x": text_x,
        "text_y": text_y,
    }


def escape_ffmpeg_path(path: Path) -> str:
    value = str(path)
    value = value.replace("\\", "\\\\")
    value = value.replace(":", "\\:")
    value = value.replace("'", r"\'")
    value = value.replace(",", r"\,")
    value = value.replace("[", r"\[")
    value = value.replace("]", r"\]")
    return value


def render_annotated_chunk(
    input_chunk: Path,
    output_chunk: Path,
    chunk_result: Dict[str, Any],
    font_file: Path,
    logger: Optional[logging.Logger] = None,
    frame_size: Optional[Tuple[int, int]] = None,
) -> None:
    if frame_size is None:
        width, height = ffprobe_size(input_chunk)
    else:
        width, height = frame_size

    lines, hazard_present_yes = build_overlay_lines(chunk_result)
    layout = compute_overlay_layout(width, height, len(lines))

    with tempfile.TemporaryDirectory(prefix="overlay_lines_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        lines_file = temp_dir / "overlay_lines.txt"
        lines_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        font_color = "red" if hazard_present_yes else "white"

        filters = [
            (
                f"drawbox=x={layout['panel_x']}:y={layout['panel_y']}:"
                f"w={layout['panel_w']}:h={layout['panel_h']}:color=black@0.55:t=fill"
            ),
            (
                f"drawbox=x={layout['panel_x']}:y={layout['panel_y']}:"
                f"w={layout['panel_w']}:h={layout['panel_h']}:color=white@0.30:t=2"
            ),
            (
                "drawtext="
                f"fontfile='{escape_ffmpeg_path(font_file)}':"
                f"textfile='{escape_ffmpeg_path(lines_file)}':"
                "reload=0:"
                f"x={layout['text_x']}:y={layout['text_y']}:"
                f"fontsize={layout['fontsize']}:"
                f"fontcolor={font_color}:"
                f"line_spacing={layout['line_spacing']}:"
                "box=0:"
                "fix_bounds=true"
            ),
        ]

        filter_complex = ",".join(filters)
        output_chunk.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(input_chunk),
            "-vf",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output_chunk),
        ]
        result = run_cmd(cmd, logger=logger)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg overlay failed for {input_chunk.name}: {result.stderr.strip()}")


def concatenate_chunks(
    chunk_paths: List[Path],
    output_path: Path,
    logger: logging.Logger,
    expected_size: Optional[Tuple[int, int]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="concat_list_") as temp_dir_name:
        concat_file = Path(temp_dir_name) / "concat.txt"
        lines = []
        for chunk_path in chunk_paths:
            normalized = str(chunk_path.resolve()).replace("'", "'\\''")
            lines.append(f"file '{normalized}'")
        concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        copy_cmd = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_path),
        ]
        copy_result = run_cmd(copy_cmd, logger=logger)
        if copy_result.returncode == 0:
            return

        logger.warning("Concat copy failed for %s. Falling back to re-encode.", output_path.name)
        reencode_cmd = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
        ]
        if expected_size is not None:
            expected_width, expected_height = expected_size
            reencode_cmd.extend([
                "-vf",
                (
                    f"scale={expected_width}:{expected_height}:force_original_aspect_ratio=decrease,"
                    f"pad={expected_width}:{expected_height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
                ),
            ])

        reencode_cmd.extend([
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output_path),
        ])
        reencode_result = run_cmd(reencode_cmd, logger=logger)
        if reencode_result.returncode != 0:
            raise RuntimeError(
                f"Failed to concatenate annotated chunks for {output_path.name}: {reencode_result.stderr.strip()}"
            )


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def cleanup_dir_if_allowed(path: Path, allowed_root: Path, logger: logging.Logger) -> None:
    try:
        resolved_path = path.resolve()
        resolved_root = allowed_root.resolve()
        resolved_path.relative_to(resolved_root)
    except Exception:
        logger.warning("Skipping cleanup outside allowed root: %s", path)
        return

    if resolved_path.exists():
        shutil.rmtree(resolved_path)
        logger.info("Removed temporary directory: %s", resolved_path)


def create_output_layout(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": output_dir,
        "results_dir": output_dir / "results",
        "chunks_dir": output_dir / "chunks",
        "overlays_dir": output_dir / "overlays",
        "overlay_chunks_dir": output_dir / "overlays" / "_chunk_overlays",
        "logs_dir": output_dir / "logs",
        "temp_dir": output_dir / "_tmp",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def build_run_metadata(
    args: argparse.Namespace,
    prompts: Dict[str, str],
    model_info: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter_dir": str(Path(args.adapter_dir).resolve()),
        "video_dir": str(Path(args.video_dir).resolve()),
        "task_mode": args.task_mode,
        "prompt_files": {
            "robot": str(Path(args.robot_prompt_file).resolve()),
            "forklift": str(Path(args.fork_prompt_file).resolve()),
        },
        "prompt_lengths": {task: len(text) for task, text in prompts.items()},
        "chunk_sec": args.chunk_sec,
        "generation": {
            "num_frames": args.num_frames,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": args.do_sample,
        },
        "model_info": model_info,
    }


def build_empty_results(
    *,
    args: argparse.Namespace,
    video_dir: Path,
    robot_prompt_path: Path,
    fork_prompt_path: Path,
    skipped_files: List[Dict[str, Any]],
    model_info: Optional[Dict[str, Any]] = None,
    prompts: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if model_info is None:
        model_info = {}
    if prompts is None:
        prompts = {"robot": "", "forklift": ""}

    return {
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_model": args.base_model,
            "adapter_dir": str(Path(args.adapter_dir).resolve()),
            "video_dir": str(video_dir.resolve()),
            "task_mode": args.task_mode,
            "prompt_files": {
                "robot": str(robot_prompt_path.resolve()),
                "forklift": str(fork_prompt_path.resolve()),
            },
            "prompt_lengths": {task: len(text) for task, text in prompts.items()},
            "chunk_sec": args.chunk_sec,
            "generation": {
                "num_frames": args.num_frames,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "do_sample": args.do_sample,
            },
            "model_info": model_info,
        },
        "videos": [],
        "skipped_files": skipped_files,
        "failed_chunks": [],
    }


def build_video_record(video_path: Path, task: str) -> Dict[str, Any]:
    return {
        "source_path": str(video_path.resolve()),
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "task": task,
        "total_duration": None,
        "total_chunks": 0,
        "annotated_video_path": None,
        "chunks": [],
    }


def build_chunk_result(
    *,
    video_path: Path,
    task: str,
    prompt_path: Path,
    chunk_meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "source_video": str(video_path.resolve()),
        "video_name": video_path.name,
        "task": task,
        "chunk_index": chunk_meta["chunk_index"],
        "chunk_path": str(Path(chunk_meta["chunk_path"]).resolve()),
        "start_time": chunk_meta["start_time"],
        "end_time": chunk_meta["end_time"],
        "duration": chunk_meta["duration"],
        "prompt_file_used": str(prompt_path.resolve()),
        "raw_response_text": None,
        "parsed_response": None,
        "parse_success": False,
        "parse_error": None,
        "required_fields": TASK_REQUIRED_FIELDS[task],
        "missing_required_fields": list(TASK_REQUIRED_FIELDS[task]),
        "extra_fields": [],
        "required_fields_present": False,
        "template_mode": None,
        "video_backend": None,
        "error_message": None,
        "annotated_chunk_path": None,
    }


def append_failed_chunk(
    failed_chunks: List[Dict[str, Any]],
    *,
    chunk_result: Dict[str, Any],
    error_message: str,
) -> None:
    failed_chunks.append({
        "source_video": chunk_result["source_video"],
        "task": chunk_result["task"],
        "chunk_index": chunk_result["chunk_index"],
        "chunk_path": chunk_result["chunk_path"],
        "start_time": chunk_result["start_time"],
        "end_time": chunk_result["end_time"],
        "duration": chunk_result["duration"],
        "error_message": error_message,
    })


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_progress(results: Dict[str, Any], results_json_path: Path, failed_chunks_path: Path) -> None:
    save_json(results_json_path, results)
    save_json(failed_chunks_path, results.get("failed_chunks", []))


def render_annotated_chunk_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        render_annotated_chunk(
            input_chunk=Path(job["input_chunk"]),
            output_chunk=Path(job["output_chunk"]),
            chunk_result=job["chunk_result"],
            font_file=Path(job["font_file"]),
            frame_size=(job["width"], job["height"]),
        )
        return {
            "success": True,
            "annotated_chunk_path": str(Path(job["output_chunk"]).resolve()),
        }
    except Exception as exc:
        return {
            "success": False,
            "error_message": str(exc),
        }


def run_inference_phase(
    *,
    args: argparse.Namespace,
    selected_videos: List[Dict[str, Any]],
    prompts: Dict[str, str],
    prompt_files: Dict[str, Path],
    project_root: Path,
    chunk_root: Path,
    results: Dict[str, Any],
    results_json_path: Path,
    failed_chunks_path: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    ensure_ml_dependencies()
    model, processor, model_info = load_model_and_processor(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        device_arg=args.device,
        trust_remote_code=args.trust_remote_code,
        logger=logger,
    )
    results["run"] = build_run_metadata(args, prompts, model_info)
    results["videos"] = []
    results.setdefault("failed_chunks", [])

    for video_info in selected_videos:
        video_path = Path(video_info["path"])
        logger.info("Phase A | Processing video: %s", video_path.name)

        video_record = build_video_record(video_path=video_path, task=video_info["task"])

        try:
            chunk_records, total_duration = split_video_into_chunks(
                video_path=video_path,
                chunk_root=chunk_root,
                chunk_sec=args.chunk_sec,
                logger=logger,
            )
            video_record["total_duration"] = round(total_duration, 3)
            video_record["total_chunks"] = len(chunk_records)
        except Exception as exc:
            logger.exception("Chunking failed for %s", video_path.name)
            failure = {
                "source_video": str(video_path.resolve()),
                "task": video_info["task"],
                "chunk_index": None,
                "chunk_path": None,
                "start_time": None,
                "end_time": None,
                "duration": None,
                "error_message": f"chunking_failed: {exc}",
            }
            results["failed_chunks"].append(failure)
            video_record["error"] = failure["error_message"]
            results["videos"].append(video_record)
            save_progress(results, results_json_path, failed_chunks_path)
            continue

        for chunk_meta in chunk_records:
            chunk_path = Path(chunk_meta["chunk_path"])
            task = video_info["task"]
            prompt_text = prompts[task]
            prompt_path = prompt_files[task]
            chunk_result = build_chunk_result(
                video_path=video_path,
                task=task,
                prompt_path=prompt_path,
                chunk_meta=chunk_meta,
            )

            try:
                messages = build_user_messages(chunk_path=chunk_path, prompt_text=prompt_text, project_root=project_root)
                raw_text, template_meta = run_inference_single(
                    model=model,
                    processor=processor,
                    messages=messages,
                    num_frames=args.num_frames,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    device_arg=args.device,
                )
                chunk_result["raw_response_text"] = raw_text
                chunk_result["template_mode"] = template_meta.get("template_mode")
                chunk_result["video_backend"] = template_meta.get("video_backend")
                chunk_result.update(parse_chunk_response(task=task, raw_text=raw_text))
            except Exception as exc:
                logger.exception("Inference failed for chunk %s", chunk_path.name)
                chunk_result["error_message"] = str(exc)
                append_failed_chunk(results["failed_chunks"], chunk_result=chunk_result, error_message=str(exc))

            video_record["chunks"].append(chunk_result)

        results["videos"].append(video_record)
        save_progress(results, results_json_path, failed_chunks_path)

    return results


def render_video_overlays(
    *,
    args: argparse.Namespace,
    results: Dict[str, Any],
    overlay_chunk_root: Path,
    font_file: Path,
    results_json_path: Path,
    failed_chunks_path: Path,
    logger: logging.Logger,
) -> None:
    results.setdefault("failed_chunks", [])

    for video_record in results.get("videos", []):
        source_path = Path(video_record["source_path"])
        logger.info("Phase B | Rendering overlays for %s", source_path.name)
        video_record["annotated_video_path"] = None

        try:
            frame_size = ffprobe_size(source_path)
        except Exception as exc:
            logger.exception("Failed to probe video size for %s", source_path.name)
            video_record["error"] = f"overlay_probe_failed: {exc}"
            save_progress(results, results_json_path, failed_chunks_path)
            continue

        jobs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for chunk_result in sorted(video_record.get("chunks", []), key=lambda item: item.get("chunk_index", 0)):
            chunk_result["annotated_chunk_path"] = None
            input_chunk = Path(chunk_result["chunk_path"])
            if not input_chunk.exists():
                error_message = f"overlay_input_missing: {input_chunk}"
                logger.error("Missing chunk for overlay: %s", input_chunk)
                chunk_result["error_message"] = (
                    f"{chunk_result['error_message']} | {error_message}"
                    if chunk_result.get("error_message")
                    else error_message
                )
                append_failed_chunk(results["failed_chunks"], chunk_result=chunk_result, error_message=error_message)
                continue

            output_chunk = overlay_chunk_root / video_record["video_stem"] / input_chunk.name
            jobs.append((
                {
                    "input_chunk": str(input_chunk.resolve()),
                    "output_chunk": str(output_chunk.resolve()),
                    "chunk_result": {
                        "task": chunk_result["task"],
                        "start_time": chunk_result["start_time"],
                        "end_time": chunk_result["end_time"],
                        "parsed_response": chunk_result.get("parsed_response"),
                    },
                    "font_file": str(font_file.resolve()),
                    "width": frame_size[0],
                    "height": frame_size[1],
                },
                chunk_result,
            ))

        if args.overlay_workers == 1 or len(jobs) <= 1:
            worker_results = [(job, chunk_result, render_annotated_chunk_worker(job)) for job, chunk_result in jobs]
        else:
            worker_results = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.overlay_workers) as executor:
                future_map = {
                    executor.submit(render_annotated_chunk_worker, job): (job, chunk_result)
                    for job, chunk_result in jobs
                }
                for future in concurrent.futures.as_completed(future_map):
                    job, chunk_result = future_map[future]
                    try:
                        worker_result = future.result()
                    except Exception as exc:
                        worker_result = {
                            "success": False,
                            "error_message": f"overlay_worker_crashed: {exc}",
                        }
                    worker_results.append((job, chunk_result, worker_result))

        for _, chunk_result, worker_result in worker_results:
            if worker_result["success"]:
                chunk_result["annotated_chunk_path"] = worker_result["annotated_chunk_path"]
                continue

            overlay_error = f"overlay_failed: {worker_result['error_message']}"
            logger.error("Overlay rendering failed for chunk %s: %s", chunk_result["chunk_path"], worker_result["error_message"])
            chunk_result["error_message"] = (
                f"{chunk_result['error_message']} | {overlay_error}"
                if chunk_result.get("error_message")
                else overlay_error
            )
            append_failed_chunk(results["failed_chunks"], chunk_result=chunk_result, error_message=overlay_error)

        annotated_chunk_paths = [
            Path(chunk_result["annotated_chunk_path"]).resolve()
            for chunk_result in sorted(video_record.get("chunks", []), key=lambda item: item.get("chunk_index", 0))
            if chunk_result.get("annotated_chunk_path")
        ]

        if annotated_chunk_paths:
            final_video_path = overlay_chunk_root.parent / f"{video_record['video_stem']}_annotated.mp4"
            try:
                concatenate_chunks(
                    annotated_chunk_paths,
                    final_video_path,
                    logger,
                    expected_size=frame_size,
                )
                video_record["annotated_video_path"] = str(final_video_path.resolve())
                if video_record.get("error") == "no_annotated_chunks_created":
                    video_record.pop("error", None)
            except Exception as exc:
                logger.exception("Final concatenation failed for %s", source_path.name)
                video_record["error"] = f"concat_failed: {exc}"
        else:
            video_record["error"] = "no_annotated_chunks_created"

        save_progress(results, results_json_path, failed_chunks_path)


def run_pipeline(args: argparse.Namespace) -> None:
    if args.overlay_workers < 1:
        raise ValueError("--overlay_workers must be at least 1.")

    output_paths = create_output_layout(Path(args.output_dir).resolve())
    logger = setup_logging(output_paths["logs_dir"] / "run.log")
    initialize_environment(logger)

    project_root = Path.cwd().resolve()
    video_dir = Path(args.video_dir).resolve()
    robot_prompt_path = Path(args.robot_prompt_file).resolve()
    fork_prompt_path = Path(args.fork_prompt_file).resolve()
    results_json_path = Path(args.results_json).resolve() if args.results_json else (output_paths["results_dir"] / "inference_results.json")
    failed_chunks_path = output_paths["logs_dir"] / "failed_chunks.json"
    chunk_root = output_paths["chunks_dir"] if args.save_chunks else (output_paths["temp_dir"] / "chunks")
    overlay_chunk_root = output_paths["overlay_chunks_dir"] if args.save_overlay else (output_paths["temp_dir"] / "overlay_chunks")
    chunk_root.mkdir(parents=True, exist_ok=True)
    overlay_chunk_root.mkdir(parents=True, exist_ok=True)

    should_chunk_or_overlay = not (args.skip_inference and args.skip_overlay)
    if should_chunk_or_overlay:
        ensure_binary("ffmpeg")
        ensure_binary("ffprobe")

    if not args.skip_inference:
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        if not robot_prompt_path.exists():
            raise FileNotFoundError(f"Robot prompt file not found: {robot_prompt_path}")
        if not fork_prompt_path.exists():
            raise FileNotFoundError(f"Forklift prompt file not found: {fork_prompt_path}")

        prompts = {
            "robot": load_text(robot_prompt_path),
            "forklift": load_text(fork_prompt_path),
        }
        prompt_files = {
            "robot": robot_prompt_path,
            "forklift": fork_prompt_path,
        }

        selected_videos, skipped_files = discover_videos(video_dir, args.task_mode, logger)
        save_json(output_paths["logs_dir"] / "skipped_files.json", skipped_files)

        results = build_empty_results(
            args=args,
            video_dir=video_dir,
            robot_prompt_path=robot_prompt_path,
            fork_prompt_path=fork_prompt_path,
            skipped_files=skipped_files,
            prompts=prompts,
        )

        if not selected_videos:
            logger.warning("No matching videos found. Exiting after writing skipped file log.")
            save_progress(results, results_json_path, failed_chunks_path)
            return

        results = run_inference_phase(
            args=args,
            selected_videos=selected_videos,
            prompts=prompts,
            prompt_files=prompt_files,
            project_root=project_root,
            chunk_root=chunk_root,
            results=results,
            results_json_path=results_json_path,
            failed_chunks_path=failed_chunks_path,
            logger=logger,
        )
        logger.info("Phase A complete. Inference results written to %s", results_json_path)
    else:
        if not results_json_path.exists():
            raise FileNotFoundError(
                f"--skip_inference was set, but results JSON was not found: {results_json_path}"
            )
        results = load_json(results_json_path)
        save_json(output_paths["logs_dir"] / "skipped_files.json", results.get("skipped_files", []))
        logger.info("Loaded existing results from %s", results_json_path)

    if not args.skip_overlay:
        font_file = detect_font_file(args.font_file)
        logger.info("Using font file: %s", font_file)
        render_video_overlays(
            args=args,
            results=results,
            overlay_chunk_root=overlay_chunk_root,
            font_file=font_file,
            results_json_path=results_json_path,
            failed_chunks_path=failed_chunks_path,
            logger=logger,
        )
        logger.info("Phase B complete. Overlay outputs written under %s", output_paths["overlays_dir"])
    else:
        save_progress(results, results_json_path, failed_chunks_path)

    if not args.save_chunks:
        cleanup_dir_if_allowed(chunk_root, output_paths["root"], logger)
    if not args.save_overlay:
        cleanup_dir_if_allowed(overlay_chunk_root, output_paths["root"], logger)

    logger.info("Run complete. Results written to %s", results_json_path)


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()

# overlay_eval_results_ffmpeg.py
#
# Uses ffmpeg drawbox + drawtext to overlay:
#   - GT   at bottom-left
#   - PRED at bottom-right
#
# Text becomes red when hazard_present == "yes".
# Output video keeps the same resolution as input.
#
# Example:
#   python overlay_eval_results_ffmpeg.py
#   python overlay_eval_results_ffmpeg.py --limit 3
#   python overlay_eval_results_ffmpeg.py --only_mismatches

import os
import json
import math
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_RESULTS_JSON = "runs/qwen35_9b_lora_newprompt/eval_results_og/checkpoint-486.json"
DEFAULT_CLIPS_ROOT = "vlm_dataset/clips/test"
DEFAULT_OUTPUT_ROOT = "runs/qwen35_9b_lora_newprompt/eval_results_og/overlay_checkpoint-486_ffmpeg"

# Change this if needed on your machine
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

ORDERED_KEYS = [
    "hazard_label",
    "hazard_present",
    "zone_relation",
    "object_state",
    "object_direction",
]


def run_cmd(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result


def ffprobe_video_size(video_path: Path) -> Tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(video_path),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}\n{result.stderr}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    w = int(streams[0]["width"])
    h = int(streams[0]["height"])
    return w, h


def normalize_str(x: Any) -> str:
    if x is None:
        return "null"
    return str(x)


def try_parse_json_text(text: Any) -> Optional[Dict[str, Any]]:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def hazard_yes(parsed: Optional[Dict[str, Any]], raw_text: Any) -> bool:
    if parsed is not None:
        return normalize_str(parsed.get("hazard_present", "")).strip().lower() == "yes"
    if isinstance(raw_text, str):
        s = raw_text.lower()
        return '"hazard_present"' in s and '"yes"' in s
    return False


def build_block_text(title: str, parsed: Optional[Dict[str, Any]], raw_text: Any) -> str:
    if parsed is not None:
        lines = [title]
        for key in ORDERED_KEYS:
            lines.append(f"{key}: {normalize_str(parsed.get(key, 'missing'))}")
        return "\n".join(lines)

    return f"{title}\n{str(raw_text)}"


def resolve_video_path(clips_root: Path, item: Dict[str, Any]) -> Optional[Path]:
    sample_id = item.get("sample_id")
    meta = item.get("meta", {}) or {}
    source_video_id = meta.get("source_video_id")

    if not sample_id:
        return None

    if source_video_id:
        p = clips_root / source_video_id / f"{sample_id}.mp4"
        if p.exists():
            return p

    matches = list(clips_root.rglob(f"{sample_id}.mp4"))
    if matches:
        return matches[0]

    return None


def compute_layout(w: int, h: int) -> Dict[str, int]:
    # keep text moderate and stable; no resizing of the video itself
    margin = max(18, int(round(min(w, h) * 0.02)))
    padding = max(12, int(round(min(w, h) * 0.012)))
    panel_w = int(round(w * 0.42))

    # moderate font size; clamped so it doesn't become huge
    fontsize = max(18, min(30, int(round(h * 0.026))))
    line_spacing = max(4, int(round(fontsize * 0.28)))

    # 6 lines total: title + 5 fields
    num_lines = 6
    line_h = fontsize + line_spacing
    panel_h = padding * 2 + num_lines * line_h

    y_top = h - margin - panel_h
    left_x = margin
    right_x = w - margin - panel_w

    text_left_x = left_x + padding
    text_right_x = right_x + padding
    text_y = y_top + padding + fontsize

    return {
        "margin": margin,
        "padding": padding,
        "panel_w": panel_w,
        "panel_h": panel_h,
        "fontsize": fontsize,
        "line_spacing": line_spacing,
        "left_x": left_x,
        "right_x": right_x,
        "y_top": y_top,
        "text_left_x": text_left_x,
        "text_right_x": text_right_x,
        "text_y": text_y,
    }


def escape_ffmpeg_path(path: Path) -> str:
    # for textfile/fontfile paths in ffmpeg filter expressions
    s = str(path)
    s = s.replace("\\", "\\\\")
    s = s.replace(":", "\\:")
    s = s.replace("'", r"\'")
    s = s.replace(",", r"\,")
    s = s.replace("[", r"\[")
    s = s.replace("]", r"\]")
    return s


def make_filter_complex(
    gt_text_path: Path,
    pred_text_path: Path,
    font_path: Path,
    layout: Dict[str, int],
    gt_red: bool,
    pred_red: bool,
) -> str:
    gt_color = "red" if gt_red else "white"
    pred_color = "red" if pred_red else "white"

    gt_textfile = escape_ffmpeg_path(gt_text_path)
    pred_textfile = escape_ffmpeg_path(pred_text_path)
    fontfile = escape_ffmpeg_path(font_path)

    lx = layout["left_x"]
    rx = layout["right_x"]
    yt = layout["y_top"]
    pw = layout["panel_w"]
    ph = layout["panel_h"]
    txl = layout["text_left_x"]
    txr = layout["text_right_x"]
    ty = layout["text_y"]
    fs = layout["fontsize"]
    ls = layout["line_spacing"]

    filters = []

    # bottom-left GT background
    filters.append(
        f"drawbox=x={lx}:y={yt}:w={pw}:h={ph}:color=black@0.55:t=fill"
    )
    filters.append(
        f"drawbox=x={lx}:y={yt}:w={pw}:h={ph}:color=white@0.35:t=2"
    )

    # bottom-right prediction background
    filters.append(
        f"drawbox=x={rx}:y={yt}:w={pw}:h={ph}:color=black@0.55:t=fill"
    )
    filters.append(
        f"drawbox=x={rx}:y={yt}:w={pw}:h={ph}:color=white@0.35:t=2"
    )

    # GT text
    filters.append(
        "drawtext="
        f"fontfile='{fontfile}':"
        f"textfile='{gt_textfile}':"
        f"reload=0:"
        f"x={txl}:y={ty}:"
        f"fontsize={fs}:"
        f"fontcolor={gt_color}:"
        f"line_spacing={ls}:"
        "box=0:"
        "fix_bounds=true"
    )

    # prediction text
    filters.append(
        "drawtext="
        f"fontfile='{fontfile}':"
        f"textfile='{pred_textfile}':"
        f"reload=0:"
        f"x={txr}:y={ty}:"
        f"fontsize={fs}:"
        f"fontcolor={pred_color}:"
        f"line_spacing={ls}:"
        "box=0:"
        "fix_bounds=true"
    )

    return ",".join(filters)


def render_one(
    video_path: Path,
    output_path: Path,
    gt_text: str,
    pred_text: str,
    gt_red: bool,
    pred_red: bool,
    font_path: Path,
    crf: int,
    preset: str,
) -> Tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    w, h = ffprobe_video_size(video_path)
    layout = compute_layout(w, h)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        gt_file = tmpdir / "gt.txt"
        pred_file = tmpdir / "pred.txt"

        gt_file.write_text(gt_text, encoding="utf-8")
        pred_file.write_text(pred_text, encoding="utf-8")

        filter_complex = make_filter_complex(
            gt_text_path=gt_file,
            pred_text_path=pred_file,
            font_path=font_path,
            layout=layout,
            gt_red=gt_red,
            pred_red=pred_red,
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vf", filter_complex,
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:a", "copy",
            str(output_path),
        ]

        result = run_cmd(cmd)
        if result.returncode != 0:
            return False, result.stderr

    return True, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--clips_root", default=DEFAULT_CLIPS_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--font_path", default=DEFAULT_FONT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only_mismatches", action="store_true")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    args = parser.parse_args()

    results_json = Path(args.results_json)
    clips_root = Path(args.clips_root)
    output_root = Path(args.output_root)
    font_path = Path(args.font_path)

    if not results_json.exists():
        raise FileNotFoundError(f"Results file not found: {results_json}")
    if not clips_root.exists():
        raise FileNotFoundError(f"Clips root not found: {clips_root}")
    if not font_path.exists():
        raise FileNotFoundError(
            f"Font file not found: {font_path}\n"
            "Set --font_path to a valid .ttf file on your machine."
        )

    with results_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("per_sample", [])
    if args.only_mismatches:
        items = [x for x in items if not x.get("exact_match", False)]
    if args.limit is not None:
        items = items[:args.limit]

    print(f"Selected samples: {len(items)}")

    ok_count = 0
    miss_count = 0
    fail_count = 0

    for i, item in enumerate(items, start=1):
        sample_id = item.get("sample_id", f"sample_{i}")
        source_video_id = (item.get("meta", {}) or {}).get("source_video_id", "unknown_source")

        video_path = resolve_video_path(clips_root, item)
        if video_path is None or not video_path.exists():
            print(f"[MISS {i:03d}] {sample_id}")
            miss_count += 1
            continue

        gt_raw = item.get("ground_truth_text", "")
        pred_raw = item.get("prediction_raw", "")

        gt_parsed = try_parse_json_text(gt_raw) or item.get("ground_truth")
        pred_parsed = try_parse_json_text(pred_raw) or item.get("prediction_parsed")

        gt_text = build_block_text("GT", gt_parsed, gt_raw)
        pred_text = build_block_text("PRED", pred_parsed, pred_raw)

        gt_red = hazard_yes(gt_parsed, gt_raw)
        pred_red = hazard_yes(pred_parsed, pred_raw)

        out_dir = output_root / source_video_id
        out_path = out_dir / f"{sample_id}.mp4"

        success, err = render_one(
            video_path=video_path,
            output_path=out_path,
            gt_text=gt_text,
            pred_text=pred_text,
            gt_red=gt_red,
            pred_red=pred_red,
            font_path=font_path,
            crf=args.crf,
            preset=args.preset,
        )

        if success:
            ok_count += 1
            print(f"[OK   {i:03d}] {out_path}")
        else:
            fail_count += 1
            print(f"[FAIL {i:03d}] {sample_id}")
            print(err)

    print("\nDone")
    print(f"Rendered      : {ok_count}")
    print(f"Missing video : {miss_count}")
    print(f"Failed        : {fail_count}")
    print(f"Output root   : {output_root}")


if __name__ == "__main__":
    main()
import json
import os
import cv2
import random
import shutil
import re
import copy
import hashlib
import subprocess
from pathlib import Path

import albumentations as A
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================
ORIGINAL_JSONL_PATHS = [
    "vlm_dataset_both/train_chat.jsonl",
    "vlm_dataset_both/val_chat.jsonl",
    "vlm_dataset_both/test_chat.jsonl",
]

OUTPUT_DIR = "vlm_dataset_both_aug"
OUTPUT_VIDEOS_DIR = os.path.join(OUTPUT_DIR, "clips")
TEMP_DIR = os.path.join(OUTPUT_DIR, "_temp")
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

PROJECT_ROOT = "."
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

# If True, rewrap unchanged copies with ffmpeg stream copy instead of shutil.copy2
USE_FFMPEG_FOR_CLEAN_COPY = True

# Create a night-version augmentation only for records whose parsed
# assistant JSON has zone_relation in {"no_worker", "no_forklift"}
NIGHT_AUG_ZONE_RELATIONS = {"no_worker", "no_forklift"}

# ==========================================
# 2. AUGMENTATION SETUP
# ==========================================
transform_tilt = A.ReplayCompose([
    A.Perspective(scale=(0.02, 0.05), p=1.0)
])

transform_bc = A.ReplayCompose([
    A.RandomBrightnessContrast(
        brightness_limit=0.10,
        contrast_limit=0.10,
        p=1.0
    )
])


# ==========================================
# 3. FFMPEG / FFPROBE HELPERS
# ==========================================
def run_cmd(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode == 0, proc.stdout, proc.stderr


def ffprobe_video_info(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    ok, out, err = run_cmd(cmd)
    if not ok:
        raise RuntimeError(f"ffprobe failed for {video_path}\n{err}")

    data = json.loads(out)

    video_stream = None
    audio_stream = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and video_stream is None:
            video_stream = s
        elif s.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = s

    if video_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")

    return {
        "video_codec": video_stream.get("codec_name"),
        "pix_fmt": video_stream.get("pix_fmt"),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "r_frame_rate": video_stream.get("r_frame_rate"),
        "avg_frame_rate": video_stream.get("avg_frame_rate"),
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "has_audio": audio_stream is not None,
        "format_name": data.get("format", {}).get("format_name"),
    }


def parse_fraction(frac_str, default=30.0):
    try:
        if not frac_str or frac_str == "0/0":
            return default
        num, den = frac_str.split("/")
        num = float(num)
        den = float(den)
        if den == 0:
            return default
        val = num / den
        return val if val > 0 else default
    except Exception:
        return default


def choose_output_codec(src_codec):
    """
    Try to stay close to original codec when possible.
    """
    if src_codec == "h264":
        return "libx264"
    if src_codec == "hevc":
        return "libx265"
    if src_codec == "mpeg4":
        return "mpeg4"
    return "libx264"


def choose_pixel_format(src_pix_fmt):
    supported_common = {
        "yuv420p", "yuvj420p",
        "yuv422p", "yuv444p",
        "nv12"
    }
    if src_pix_fmt in supported_common:
        if src_pix_fmt == "yuvj420p":
            return "yuv420p"
        return src_pix_fmt
    return "yuv420p"


def ffmpeg_stream_copy(src_path, dst_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-map", "0",
        "-c", "copy",
        "-map_metadata", "0",
        dst_path,
    ]
    ok, _, err = run_cmd(cmd)
    if not ok:
        raise RuntimeError(f"ffmpeg stream copy failed:\n{err}")


def ffmpeg_normalize_augmented(temp_aug_path, src_reference_path, final_out_path):
    info = ffprobe_video_info(src_reference_path)

    src_codec = info["video_codec"]
    dst_codec = choose_output_codec(src_codec)
    pix_fmt = choose_pixel_format(info["pix_fmt"])
    fps = parse_fraction(info["avg_frame_rate"] or info["r_frame_rate"], default=30.0)

    cmd = [
        "ffmpeg", "-y",
        "-i", temp_aug_path,
        "-i", src_reference_path,
        "-map", "0:v:0",
        "-map_metadata", "1",
        "-r", f"{fps:.6f}",
        "-c:v", dst_codec,
        "-pix_fmt", pix_fmt,
    ]

    if info["has_audio"]:
        cmd += ["-map", "1:a?", "-c:a", "copy"]
    else:
        cmd += ["-an"]

    if dst_codec == "libx264":
        cmd += ["-crf", "18", "-preset", "medium", "-movflags", "+faststart"]
    elif dst_codec == "libx265":
        cmd += ["-crf", "20", "-preset", "medium", "-tag:v", "hvc1"]
    elif dst_codec == "mpeg4":
        cmd += ["-q:v", "2"]
    else:
        cmd += ["-crf", "18", "-preset", "medium"]

    cmd.append(final_out_path)

    ok, _, err = run_cmd(cmd)
    if not ok:
        raise RuntimeError(f"ffmpeg normalize failed:\n{err}")


# ==========================================
# 4. VIDEO AUGMENTATION
# ==========================================
def apply_night_transform(frame_bgr, rng):
    """
    Mild night-style transform:
    - grayscale
    - very slight darkening
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Very mild darkening only
    brightness_scale = rng.uniform(0.82, 0.92)
    gray *= brightness_scale

    gray = np.clip(gray, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return out


def augment_video_with_ffmpeg_match(input_path, output_path, aug_type):
    """
    1) Decode and augment frames with OpenCV + Albumentations / custom transform
    2) Save temporary video
    3) Re-encode with ffmpeg to match source properties more closely
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps != fps:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        print(f"Error: Invalid video shape for {input_path}")
        cap.release()
        return False

    temp_path = os.path.join(
        TEMP_DIR,
        f"tmp_{hashlib.md5((str(input_path)+str(output_path)+aug_type).encode()).hexdigest()[:12]}.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not create temporary output {temp_path}")
        cap.release()
        return False

    frame_count = 0
    replay_params = None
    night_rng = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elif aug_type == "tilt":
                if frame_count == 0:
                    data = transform_tilt(image=frame)
                    aug_frame = data["image"]
                    replay_params = data["replay"]
                else:
                    aug_frame = A.ReplayCompose.replay(replay_params, image=frame)["image"]

            elif aug_type == "bc":
                if frame_count == 0:
                    data = transform_bc(image=frame)
                    aug_frame = data["image"]
                    replay_params = data["replay"]
                else:
                    aug_frame = A.ReplayCompose.replay(replay_params, image=frame)["image"]

            elif aug_type == "night":
                if frame_count == 0:
                    seed_int = int(hashlib.md5(str(input_path).encode("utf-8")).hexdigest()[:8], 16)
                    night_rng = np.random.default_rng(seed_int)
                aug_frame = apply_night_transform(frame, night_rng)

            else:
                print(f"Error: Unknown augmentation type: {aug_type}")
                cap.release()
                out.release()
                return False

            out.write(aug_frame)
            frame_count += 1

    except Exception as e:
        print(f"Error during augmentation for {input_path}: {e}")
        cap.release()
        out.release()
        return False
    finally:
        cap.release()
        out.release()

    if frame_count == 0:
        print(f"Error: No frames written for {input_path}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

    try:
        ffmpeg_normalize_augmented(temp_path, input_path, output_path)
    except Exception as e:
        print(f"Error: ffmpeg normalization failed for {input_path}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return True


# ==========================================
# 5. DATASET HELPERS
# ==========================================
def parse_possible_json(text):
    if not isinstance(text, str):
        return None

    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def get_assistant_text(record):
    messages = record.get("messages", [])
    if isinstance(messages, list):
        for msg in reversed(messages):
            role = msg.get("role", "") or msg.get("from", "")
            if str(role).lower() in {"assistant", "gpt"}:
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text" and "text" in item:
                                parts.append(str(item["text"]))
                            elif "value" in item:
                                parts.append(str(item["value"]))
                    if parts:
                        return "\n".join(parts)

    conversations = record.get("conversations", [])
    if isinstance(conversations, list):
        for msg in reversed(conversations):
            if str(msg.get("from", "")).lower() in {"assistant", "gpt"}:
                value = msg.get("value", "")
                if isinstance(value, str):
                    return value

    return json.dumps(record, ensure_ascii=False)


def get_parsed_assistant_json(record):
    assistant_text = get_assistant_text(record)
    parsed = parse_possible_json(assistant_text)
    return parsed if isinstance(parsed, dict) else None


def is_hazard_present(record):
    parsed = get_parsed_assistant_json(record)
    if isinstance(parsed, dict):
        val = str(parsed.get("hazard_present", "")).strip().lower()
        if val == "yes":
            return True
        if val == "no":
            return False

    record_str = json.dumps(record, ensure_ascii=False).lower()
    clean_str = re.sub(r'[\s"\'\\:]', '', record_str)
    return "hazard_presentyes" in clean_str


def get_zone_relation(record):
    parsed = get_parsed_assistant_json(record)
    if isinstance(parsed, dict):
        return str(parsed.get("zone_relation", "")).strip().lower()
    return ""


def should_create_night_version(record):
    return get_zone_relation(record) in NIGHT_AUG_ZONE_RELATIONS


def get_video_path(record):
    top_video = record.get("video")
    if isinstance(top_video, str) and top_video.strip():
        return top_video.strip()

    messages = record.get("messages", [])
    if isinstance(messages, list):
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = str(item.get("type", "")).lower()
                    if item_type == "video":
                        for key in ["video", "video_url", "path", "file"]:
                            value = item.get(key)
                            if isinstance(value, str) and value.strip():
                                return value.strip()
                    for key in ["video", "video_url", "path", "file"]:
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            return value.strip()
    return ""


def set_video_path(record, new_path):
    new_record = copy.deepcopy(record)

    if "video" in new_record:
        new_record["video"] = new_path
        return new_record

    messages = new_record.get("messages", [])
    if isinstance(messages, list):
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if str(item.get("type", "")).lower() == "video":
                        for key in ["video", "video_url", "path", "file"]:
                            if key in item:
                                item[key] = new_path
                                return new_record

    new_record["video"] = new_path
    return new_record


def resolve_input_video_path(raw_path, source_jsonl_path, project_root="."):
    if not raw_path:
        return None

    candidates = [
        Path(raw_path),
        Path.cwd() / raw_path,
        Path(project_root) / raw_path,
        Path(source_jsonl_path).resolve().parent / raw_path,
        Path(source_jsonl_path).resolve().parent.parent / raw_path,
    ]

    seen = set()
    for c in candidates:
        try:
            r = c.resolve()
        except Exception:
            continue
        if str(r) in seen:
            continue
        seen.add(str(r))
        if r.exists():
            return str(r)
    return None


def make_stable_output_name(src_path):
    src_path = str(src_path)
    p = Path(src_path)
    stem = p.stem
    parent_bits = [x for x in p.parent.parts if x not in ("", ".", "/")]
    parent_str = "__".join(parent_bits[-3:]) if parent_bits else "root"
    digest = hashlib.md5(src_path.encode("utf-8")).hexdigest()[:10]
    filename = f"{parent_str}__{stem}__{digest}.mp4"
    filename = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", filename)
    return filename


def write_jsonl(records, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ==========================================
# 6. MAIN PIPELINE
# ==========================================
def main():
    random.seed(RANDOM_SEED)

    hazard_families = []
    clean_records = []

    print("1. Parsing Original Data...")
    for jsonl_file in ORIGINAL_JSONL_PATHS:
        if not os.path.exists(jsonl_file):
            print(f"Warning: Could not find {jsonl_file}")
            continue

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except Exception as e:
                    print(f"Warning: Bad JSON at {jsonl_file}:{line_num}: {e}")
                    continue

                record["_source_jsonl"] = jsonl_file

                if is_hazard_present(record):
                    hazard_families.append([record])
                else:
                    clean_records.append(record)

    print(f"Found {len(hazard_families)} hazard instances and {len(clean_records)} clean instances.")

    print("\n2. Processing Hazard Data (Copying & Augmenting)...")
    processed_hazard_families = []

    for idx, family in enumerate(hazard_families, start=1):
        original_record = family[0]
        raw_video_path = get_video_path(original_record)
        source_jsonl_path = original_record.get("_source_jsonl", "")

        resolved_video_path = resolve_input_video_path(
            raw_video_path,
            source_jsonl_path,
            PROJECT_ROOT,
        )

        if not resolved_video_path:
            print(f"Missing video file, skipping: {raw_video_path}")
            continue

        base_output_name = make_stable_output_name(resolved_video_path)
        base_stem = base_output_name[:-4] if base_output_name.endswith(".mp4") else base_output_name

        new_orig_path = os.path.join(OUTPUT_VIDEOS_DIR, f"{base_stem}.mp4")

        if not os.path.exists(new_orig_path):
            if USE_FFMPEG_FOR_CLEAN_COPY:
                ffmpeg_stream_copy(resolved_video_path, new_orig_path)
            else:
                shutil.copy2(resolved_video_path, new_orig_path)

        new_family = [set_video_path(original_record, new_orig_path)]

        for aug in ["tilt", "bc"]:
            aug_vid_path = os.path.join(OUTPUT_VIDEOS_DIR, f"{base_stem}_aug_{aug}.mp4")
            if not os.path.exists(aug_vid_path):
                ok = augment_video_with_ffmpeg_match(resolved_video_path, aug_vid_path, aug)
                if not ok:
                    print(f"Warning: augmentation failed for {resolved_video_path} [{aug}]")
                    continue
            new_family.append(set_video_path(original_record, aug_vid_path))

        cleaned_family = []
        for rec in new_family:
            rec2 = copy.deepcopy(rec)
            rec2.pop("_source_jsonl", None)
            cleaned_family.append(rec2)

        processed_hazard_families.append(cleaned_family)

        if idx % 10 == 0 or idx == len(hazard_families):
            print(f"Processed {idx}/{len(hazard_families)} hazard families...")

    hazard_families = processed_hazard_families
    print(f"Kept {len(hazard_families)} hazard families after file validation.")

    print("\n3. Processing Clean Data as Families (copy + selective night aug)...")
    processed_clean_families = []
    night_aug_count = 0
    
    for record in clean_records:
        raw_video_path = get_video_path(record)
        source_jsonl_path = record.get("_source_jsonl", "")
    
        resolved_video_path = resolve_input_video_path(
            raw_video_path,
            source_jsonl_path,
            PROJECT_ROOT,
        )
    
        if not resolved_video_path:
            continue
    
        base_output_name = make_stable_output_name(resolved_video_path)
        base_stem = base_output_name[:-4] if base_output_name.endswith(".mp4") else base_output_name
    
        new_vid_path = os.path.join(OUTPUT_VIDEOS_DIR, f"{base_stem}.mp4")
    
        if not os.path.exists(new_vid_path):
            if USE_FFMPEG_FOR_CLEAN_COPY:
                ffmpeg_stream_copy(resolved_video_path, new_vid_path)
            else:
                shutil.copy2(resolved_video_path, new_vid_path)
    
        family = []
    
        new_record = set_video_path(record, new_vid_path)
        new_record.pop("_source_jsonl", None)
        family.append(new_record)
    
        if should_create_night_version(record):
            night_vid_path = os.path.join(OUTPUT_VIDEOS_DIR, f"{base_stem}_aug_night.mp4")
    
            if not os.path.exists(night_vid_path):
                ok = augment_video_with_ffmpeg_match(resolved_video_path, night_vid_path, "night")
                if not ok:
                    print(f"Warning: night augmentation failed for {resolved_video_path}")
                else:
                    night_record = set_video_path(record, night_vid_path)
                    night_record.pop("_source_jsonl", None)
                    family.append(night_record)
                    night_aug_count += 1
            else:
                night_record = set_video_path(record, night_vid_path)
                night_record.pop("_source_jsonl", None)
                family.append(night_record)
                night_aug_count += 1
    
        processed_clean_families.append(family)
    
    clean_families = processed_clean_families
    print(f"Kept {len(clean_families)} clean families after file validation.")
    print(f"Created {night_aug_count} selective night-augmented clean records.")

    print("\n4. Shuffling and Splitting Data Safely...")
    random.shuffle(hazard_families)
    random.shuffle(clean_families)
    
    total_hazard_families = len(hazard_families)
    train_h_fam_end = int(total_hazard_families * SPLIT_RATIOS["train"])
    val_h_fam_end = train_h_fam_end + int(total_hazard_families * SPLIT_RATIOS["val"])
    
    train_hazard_families = hazard_families[:train_h_fam_end]
    val_hazard_families = hazard_families[train_h_fam_end:val_h_fam_end]
    test_hazard_families = hazard_families[val_h_fam_end:]
    
    total_clean_families = len(clean_families)
    train_c_fam_end = int(total_clean_families * SPLIT_RATIOS["train"])
    val_c_fam_end = train_c_fam_end + int(total_clean_families * SPLIT_RATIOS["val"])
    
    train_clean_families = clean_families[:train_c_fam_end]
    val_clean_families = clean_families[train_c_fam_end:val_c_fam_end]
    test_clean_families = clean_families[val_c_fam_end:]

    train_yes = [r for fam in train_hazard_families for r in fam]
    val_yes = [r for fam in val_hazard_families for r in fam]
    test_yes = [r for fam in test_hazard_families for r in fam]
    
    train_no = [r for fam in train_clean_families for r in fam]
    val_no = [r for fam in val_clean_families for r in fam]
    test_no = [r for fam in test_clean_families for r in fam]
    
    final_train = train_yes + train_no
    final_val = val_yes + val_no
    final_test = test_yes + test_no

    random.shuffle(final_train)
    random.shuffle(final_val)
    random.shuffle(final_test)

    print(f"Train: {len(final_train)} records ({len(train_yes)} hazard-derived + {len(train_no)} clean)")
    print(f"Val:   {len(final_val)} records ({len(val_yes)} hazard-derived + {len(val_no)} clean)")
    print(f"Test:  {len(final_test)} records ({len(test_yes)} hazard-derived + {len(test_no)} clean)")

    print("\n5. Writing Standalone JSONL Files...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, records in {
        "train_chat.jsonl": final_train,
        "val_chat.jsonl": final_val,
        "test_chat.jsonl": final_test,
    }.items():
        out_path = os.path.join(OUTPUT_DIR, name)
        write_jsonl(records, out_path)
        print(f"Wrote {len(records)} records to {name}")

    print("\nDone! Dataset rebuilt with FFmpeg-normalized outputs.")


if __name__ == "__main__":
    main()
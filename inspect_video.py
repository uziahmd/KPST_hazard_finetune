import os, sys

VIDEO = r"D:\Test\KPST_hazard_finetune\vlm_dataset\clips\train\금진Camera05_S20251213073941_E20251213074740\금진Camera05_S20251213082156_E20251213083037__000506000_000511000.mp4"

print("File exists:", os.path.exists(VIDEO))
size_mb = os.path.getsize(VIDEO) / 1024**2
print(f"File size: {size_mb:.2f} MB")

try:
    import cv2
    cap = cv2.VideoCapture(VIDEO)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / fps if fps else 0
    print(f"Resolution : {w}x{h}")
    print(f"FPS        : {fps}")
    print(f"Frames     : {n}")
    print(f"Duration   : {dur:.2f}s")
    cap.release()
except ImportError:
    print("cv2 not available, trying decord...")
    try:
        from decord import VideoReader
        vr = VideoReader(VIDEO)
        print(f"Frames     : {len(vr)}")
        print(f"FPS        : {vr.get_avg_fps()}")
        h, w, _ = vr[0].shape
        print(f"Resolution : {w}x{h}")
        print(f"Duration   : {len(vr)/vr.get_avg_fps():.2f}s")
    except Exception as e:
        print("decord also failed:", e)
except Exception as e:
    print("Error reading video:", e)

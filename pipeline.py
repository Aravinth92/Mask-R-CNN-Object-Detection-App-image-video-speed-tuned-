"""Image / video processing pipeline. Takes any predict_fn(frame_bgr) -> det dict."""
import os
import shutil
import subprocess
import tempfile
import time

import cv2

from visuals import annotate, resize_max_side

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _to_h264(src, dst):
    """Re-encode to browser-playable H.264. Returns True on success."""
    exe = _ffmpeg_exe()
    if not exe:
        return False
    cmd = [exe, "-y", "-loglevel", "error", "-i", src, "-vcodec", "libx264",
           "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p", dst]
    try:
        subprocess.run(cmd, check=True, timeout=600)
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        return False


def process_image(frame_bgr, predict_fn, categories, colors, selected_ids, alpha, max_side):
    frame = resize_max_side(frame_bgr, max_side)
    t0 = time.perf_counter()
    det = predict_fn(frame)
    t1 = time.perf_counter()
    out, counts = annotate(frame, det, categories, colors, selected_ids, alpha)
    return out, counts, t1 - t0


def process_video(path, predict_fn, categories, colors, selected_ids, alpha,
                  max_side, stride, max_frames, progress=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    total = max(1, min(total, max_frames))

    raw_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = None
    peak, last_det, idx, n_infer = {}, None, 0, 0
    t0 = time.perf_counter()

    try:
        while idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_max_side(frame, max_side)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if last_det is None or idx % stride == 0:
                last_det = predict_fn(frame)
                n_infer += 1
            out, counts = annotate(frame, last_det, categories, colors, selected_ids, alpha)
            for k, v in counts.items():
                peak[k] = max(peak.get(k, 0), v)
            writer.write(out)
            idx += 1
            if progress:
                progress(idx / total, f"Frame {idx}/{total}")
    finally:
        cap.release()
        if writer:
            writer.release()

    if idx == 0:
        raise ValueError("The video has no readable frames.")

    final_path = raw_path.replace(".mp4", "_h264.mp4")
    result_path = final_path if _to_h264(raw_path, final_path) else raw_path
    elapsed = time.perf_counter() - t0
    return result_path, peak, {"frames": idx, "inferences": n_infer, "seconds": elapsed,
                               "hit_limit": idx >= max_frames}

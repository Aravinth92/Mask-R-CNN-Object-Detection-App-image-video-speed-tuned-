"""Drawing helpers (numpy + OpenCV only, no torch)."""
import cv2
import numpy as np


def make_colors(n, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(70, 255, size=(n, 3), dtype=np.uint8)


def resize_max_side(frame, max_side):
    """Downscale so the longest side <= max_side (never upscales). Returns even dims."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    nw, nh = nw - nw % 2, nh - nh % 2  # even sizes are needed for H.264
    nw, nh = max(nw, 2), max(nh, 2)
    if (nw, nh) == (w, h):
        return frame
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def annotate(frame, det, categories, colors, selected_ids=None, alpha=0.45):
    """Draw masks, contours, boxes and labels. Returns (image, {class_name: count}).

    det: dict of numpy arrays -> boxes [N,4], labels [N], scores [N], masks [N,H,W] bool
    """
    n = len(det["scores"])
    keep = np.ones(n, dtype=bool)
    if selected_ids:
        keep &= np.isin(det["labels"], list(selected_ids))
    idxs = np.flatnonzero(keep)

    out = frame.copy()
    counts = {}
    if len(idxs) == 0:
        return out, counts

    h, w = frame.shape[:2]

    # 1) all masks in ONE blend (much faster than blending per object)
    overlay = frame.copy()
    union = np.zeros((h, w), dtype=bool)
    for i in idxs:
        m = det["masks"][i]
        overlay[m] = colors[int(det["labels"][i]) % len(colors)]
        union |= m
    if alpha > 0:
        blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        out[union] = blended[union]

    # 2) contours, boxes, labels
    fs = max(0.45, w / 1500)
    th = max(1, int(round(w / 700)))
    for i in idxs:
        label = int(det["labels"][i])
        name = categories[label]
        counts[name] = counts.get(name, 0) + 1
        color = tuple(int(c) for c in colors[label % len(colors)])

        x1, y1, x2, y2 = (int(v) for v in det["boxes"][i])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w - 1), min(y2, h - 1)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = det["masks"][i][y1:y2 + 1, x1:x2 + 1].astype(np.uint8)
        cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, th, offset=(x1, y1))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, th)

        text = f"{name} {float(det['scores'][i]):.2f}"
        (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        top = max(y1 - tht - base - 4, 0)
        cv2.rectangle(out, (x1, top), (x1 + tw + 6, top + tht + base + 4), color, -1)
        cv2.putText(out, text, (x1 + 3, top + tht + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (0, 0, 0), 1, cv2.LINE_AA)

    return out, counts

"""
Mask R-CNN Object Detection App  (image + video, speed-tuned)  -  Professional UI

Run:   python app.py        -> opens http://127.0.0.1:7860
"""
import os
import time

import cv2
import gradio as gr

from detector import DEVICE, PROFILES, Detector
from pipeline import IMAGE_EXTS, VIDEO_EXTS, process_image, process_video

detector = Detector()
print(f"Device: {DEVICE}. Loading model (first run downloads ~170 MB)...")
detector.warmup("Fast")
CLASS_NAMES = detector.class_names()
ON_GPU = DEVICE.type == "cuda"


# ----------------------------------------------------------------- logic (unchanged)
def table(counts):
    return [[k, v] for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]


def _path(f):
    return f if isinstance(f, str) else getattr(f, "name", None)


def _status(text, kind="ok"):
    """Small coloured status pill shown under the results."""
    return f'<div class="status {kind}"><span class="dot"></span>{text}</div>'


def run_upload(file, profile, threshold, classes, alpha, max_side, stride, max_frames,
               progress=gr.Progress()):
    path = _path(file)
    if not path:
        raise gr.Error("Upload an image or a video first.")
    ext = os.path.splitext(path)[1].lower()
    predict_fn = detector.make_predict_fn(profile, threshold)
    ids = detector.ids_for(classes)
    cats, cols = detector.categories, detector.colors

    if ext in IMAGE_EXTS:
        frame = cv2.imread(path)
        if frame is None:
            raise gr.Error("Could not read this image.")
        out, counts, dt = process_image(frame, predict_fn, cats, cols, ids, alpha, int(max_side))
        msg = (f"<b>{sum(counts.values())}</b> object(s) found in <b>{dt:.2f}s</b> "
               f"&nbsp;·&nbsp; {profile} mode &nbsp;·&nbsp; {DEVICE.type.upper()}")
        return (gr.update(value=cv2.cvtColor(out, cv2.COLOR_BGR2RGB), visible=True),
                gr.update(value=None, visible=False), table(counts), _status(msg))

    if ext in VIDEO_EXTS:
        out_path, peak, info = process_video(
            path, predict_fn, cats, cols, ids, alpha, int(max_side), int(stride),
            int(max_frames), progress=lambda p, d: progress(p, desc=d))
        fps = info["frames"] / max(info["seconds"], 1e-6)
        msg = (f"Processed <b>{info['frames']}</b> frames in <b>{info['seconds']:.1f}s</b> "
               f"&nbsp;·&nbsp; <b>{fps:.1f}</b> fps &nbsp;·&nbsp; {info['inferences']} model runs")
        kind = "ok"
        if info["hit_limit"]:
            msg += " &nbsp;·&nbsp; stopped at the max-frames limit"
            kind = "warn"
        return (gr.update(value=None, visible=False),
                gr.update(value=out_path, visible=True), table(peak), _status(msg, kind))

    raise gr.Error(f"Unsupported file type: {ext}")


def run_webcam(image, profile, threshold, classes, alpha, max_side):
    if image is None:
        raise gr.Error("Take a webcam snapshot or upload an image first.")
    predict_fn = detector.make_predict_fn(profile, threshold)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out, counts, dt = process_image(bgr, predict_fn, detector.categories, detector.colors,
                                    detector.ids_for(classes), alpha, int(max_side))
    msg = f"<b>{sum(counts.values())}</b> object(s) found in <b>{dt:.2f}s</b>"
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), table(counts), _status(msg)


# ----------------------------------------------------------------- look & feel
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#f4f6fb",
    body_background_fill_dark="#0b1220",
    block_background_fill="#ffffff",
    block_background_fill_dark="#111a2e",
    block_border_color="#e2e8f0",
    block_border_color_dark="#1f2b45",
    block_label_background_fill_dark="#111a2e",
    input_background_fill_dark="#0d1627",
    button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4f46e5 0%, #0891b2 100%)",
    button_primary_text_color="#ffffff",
    button_primary_border_color="transparent",
    slider_color="#6366f1",
    checkbox_background_color_selected="#6366f1",
)

CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* ---------- hero header ---------- */
.hero {
  position: relative; overflow: hidden; padding: 28px 32px; margin-bottom: 8px;
  border-radius: 18px; color: #fff;
  background: radial-gradient(1200px 300px at 90% -20%, rgba(6,182,212,.35), transparent 60%),
              linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #0e7490 100%);
  box-shadow: 0 10px 30px rgba(49,46,129,.35);
}
.hero h1 { margin: 0 0 6px; font-size: 1.9rem; font-weight: 800; letter-spacing: -.02em; color:#fff; }
.hero p  { margin: 0; opacity: .85; font-size: .98rem; max-width: 760px; }
.chips   { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
.chip {
  padding: 5px 12px; border-radius: 999px; font-size: .78rem; font-weight: 600;
  background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22);
  backdrop-filter: blur(6px);
}
.chip.live::before {
  content: ""; display: inline-block; width: 7px; height: 7px; margin-right: 7px;
  border-radius: 50%; background: #34d399; box-shadow: 0 0 8px #34d399;
}

/* ---------- section titles ---------- */
.section-title {
  font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
  color: #6366f1; margin: 4px 0 -2px;
}

/* ---------- tabs ---------- */
.tab-nav button, button[role="tab"] { font-weight: 600 !important; }
button[role="tab"][aria-selected="true"] { color: #6366f1 !important; border-color: #6366f1 !important; }

/* ---------- primary button ---------- */
button.primary {
  font-weight: 700 !important; letter-spacing: .01em; padding: 12px 18px !important;
  box-shadow: 0 6px 18px rgba(99,102,241,.35); transition: transform .12s ease, box-shadow .12s ease;
}
button.primary:hover { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(99,102,241,.45); }

/* ---------- status pill ---------- */
.status {
  display: flex; align-items: center; gap: 10px; padding: 12px 16px; border-radius: 12px;
  font-size: .92rem; border: 1px solid; margin: 4px 0;
}
.status .dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
.status.ok   { background: rgba(16,185,129,.10); border-color: rgba(16,185,129,.35); color: #059669; }
.status.ok .dot   { background: #10b981; box-shadow: 0 0 8px #10b981; }
.status.warn { background: rgba(245,158,11,.10); border-color: rgba(245,158,11,.40); color: #b45309; }
.status.warn .dot { background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }
.dark .status.ok   { color: #6ee7b7; }
.dark .status.warn { color: #fcd34d; }

/* ---------- results table ---------- */
table { border-radius: 12px !important; overflow: hidden; }
thead th { background: rgba(99,102,241,.12) !important; font-weight: 700 !important; }

.foot { text-align: center; opacity: .55; font-size: .8rem; padding: 14px 0 4px; }
"""

# force the dark look regardless of the OS/browser setting
FORCE_DARK_JS = """
() => {
  document.body.classList.add('dark');
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.history.replaceState({}, '', url);
  }
}
"""

# Gradio 6+ takes theme/css/js in launch(); older versions take them in gr.Blocks().
_G6 = int(gr.__version__.split(".")[0]) >= 6
_STYLE = dict(theme=THEME, css=CSS, js=FORCE_DARK_JS)

with gr.Blocks(title="Mask R-CNN Object Detector", **({} if _G6 else _STYLE)) as demo:
    gr.HTML(f"""
    <div class="hero">
      <h1>Mask R-CNN Object Detector</h1>
      <p>Detect and segment every object in images, videos and webcam snapshots —
         tuned for speed, built for accuracy.</p>
      <div class="chips">
        <span class="chip live">Running on {DEVICE.type.upper()}</span>
        <span class="chip">80 COCO classes</span>
        <span class="chip">Image · Video · Webcam</span>
        <span class="chip">Instance segmentation</span>
      </div>
    </div>
    """)

    gr.HTML('<div class="section-title">Settings</div>')
    with gr.Accordion("Detection settings", open=True):
        with gr.Row():
            profile = gr.Radio(list(PROFILES), value="Balanced" if ON_GPU else "Fast",
                               label="Speed mode (Fast = quickest, Accurate = best quality)")
            threshold = gr.Slider(0.1, 0.95, value=0.6, step=0.05, label="Confidence threshold")
            alpha = gr.Slider(0.0, 0.9, value=0.45, step=0.05, label="Mask opacity")
        classes = gr.Dropdown(CLASS_NAMES, multiselect=True, value=[],
                              label="Only detect these objects (empty = all 80 classes)")
        with gr.Accordion("Advanced (speed tuning)", open=False):
            with gr.Row():
                max_side = gr.Slider(320, 1920, value=960, step=32,
                                     label="Resize longest side to (smaller = faster)")
                stride = gr.Slider(1, 10, value=1 if ON_GPU else 3, step=1,
                                   label="Video: run detection every N frames")
                max_frames = gr.Slider(60, 3000, value=900, step=30, label="Video: max frames")

    gr.HTML('<div class="section-title">Workspace</div>')
    with gr.Tabs():
        with gr.Tab("Upload image or video"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4):
                    file_in = gr.File(label="Drop an image or a video",
                                      file_types=["image", "video"], type="filepath")
                    go = gr.Button("Detect objects", variant="primary", size="lg")
                with gr.Column(scale=6):
                    img_out = gr.Image(label="Result", visible=True)
                    vid_out = gr.Video(label="Result", visible=False)
            status = gr.HTML()
            tbl = gr.Dataframe(headers=["Object", "Count"], interactive=False,
                               label="Detected objects (video = peak count at once)")
            go.click(run_upload,
                     [file_in, profile, threshold, classes, alpha, max_side, stride, max_frames],
                     [img_out, vid_out, tbl, status])

        with gr.Tab("Webcam snapshot"):
            with gr.Row():
                cam_in = gr.Image(label="Camera", sources=["webcam"], type="numpy")
                cam_out = gr.Image(label="Result")
            cam_btn = gr.Button("Detect objects", variant="primary", size="lg")
            cam_status = gr.HTML()
            cam_tbl = gr.Dataframe(headers=["Object", "Count"], interactive=False)
            cam_btn.click(run_webcam,
                          [cam_in, profile, threshold, classes, alpha, max_side],
                          [cam_out, cam_tbl, cam_status])

    gr.HTML('<div class="foot">Mask R-CNN · PyTorch · Gradio</div>')

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, **(_STYLE if _G6 else {}))
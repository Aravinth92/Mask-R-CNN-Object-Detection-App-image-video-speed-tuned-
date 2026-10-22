"""Mask R-CNN wrapper tuned for speed."""
import os

import numpy as np
import torch
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
)

from visuals import make_colors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
else:
    torch.set_num_threads(os.cpu_count() or 4)

# Smaller internal image size = much faster. "Accurate" uses the bigger v2 model.
PROFILES = {
    "Fast":     {"model": "v1", "size": 384},
    "Balanced": {"model": "v1", "size": 512},
    "Accurate": {"model": "v2", "size": 800},
}

_FAST_KWARGS = dict(
    rpn_pre_nms_top_n_test=500,    # default 1000
    rpn_post_nms_top_n_test=300,   # default 1000
    box_detections_per_img=50,     # default 100
)


class Detector:
    def __init__(self):
        self._models = {}
        self.categories = None
        self.colors = None

    def _get(self, kind):
        if kind not in self._models:
            if kind == "v1":
                w = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
                m = maskrcnn_resnet50_fpn(weights=w, **_FAST_KWARGS)
            else:
                w = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                m = maskrcnn_resnet50_fpn_v2(weights=w, **_FAST_KWARGS)
            self._models[kind] = m.to(DEVICE).eval()
            if self.categories is None:
                self.categories = list(w.meta["categories"])
                self.colors = make_colors(len(self.categories))
        return self._models[kind]

    def warmup(self, profile="Fast"):
        """Load weights now so the first user request isn't slow."""
        self._get(PROFILES[profile]["model"])

    def class_names(self):
        self.warmup()
        return sorted(c for c in self.categories if c not in ("__background__", "N/A"))

    def ids_for(self, names):
        self.warmup()
        return {i for i, c in enumerate(self.categories) if c in set(names or [])}

    def make_predict_fn(self, profile, threshold):
        cfg = PROFILES[profile]
        model = self._get(cfg["model"])
        model.transform.min_size = (cfg["size"],)
        model.transform.max_size = int(cfg["size"] * 5 / 3)
        model.roi_heads.score_thresh = float(threshold)  # skip masks for weak boxes

        use_amp = DEVICE.type == "cuda"

        @torch.inference_mode()
        def predict(frame_bgr):
            rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            x = torch.from_numpy(rgb).to(DEVICE).permute(2, 0, 1).float().div_(255.0)
            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                out = model([x])[0]
            return {
                "boxes": out["boxes"].float().cpu().numpy(),
                "labels": out["labels"].cpu().numpy(),
                "scores": out["scores"].float().cpu().numpy(),
                "masks": (out["masks"][:, 0] > 0.5).cpu().numpy(),  # threshold on device
            }

        return predict

MASK R-CNN OBJECT DETECTOR  (image + video)
==========================================

QUICK START
  Windows : double-click run.bat
  Mac/Linux: ./run.sh
  (or manually: pip install -r requirements.txt  then  python app.py)

The browser opens at http://127.0.0.1:7860.
First run downloads the model weights (~170 MB). Internet needed only once.

HOW TO USE
  1. Upload an image OR a video (Upload tab), or take a snapshot (Webcam tab).
  2. Click "Detect objects".
  3. See masks + boxes + labels, and the object count table.

MAKING IT FASTER
  * NVIDIA GPU is the biggest speed-up (10-30x). Install the CUDA build of PyTorch:
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  * Use "Fast" speed mode.
  * Advanced: lower "Resize longest side" (e.g. 640) and raise
    "run detection every N frames" (3-5 for video on CPU).
  * Pick only the classes you need in the dropdown.

FILES
  app.py        web UI
  detector.py   Mask R-CNN model (speed-tuned)
  pipeline.py   image / video processing
  visuals.py    mask / box drawing

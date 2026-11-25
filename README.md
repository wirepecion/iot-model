# Eye-state detector (open / closed eyes)

Trains a small YOLO26n object detector that draws a box around each eye and says whether it is
`open_eyes` or `close_eyes`, then exports it to ONNX to run on the Raspberry Pi.

## Run

```bash
pip install -r requirements.txt     # on a PC or Colab, ideally with a GPU
python3 train/train.py              # trains, prints per-class results on the test images, exports ONNX
```

| Output | What it is |
|---|---|
| `models/eyes.onnx` | the model to copy to the Pi |
| `runs/eyes/` | training curves, confusion matrix, `weights/best.pt` |

Settings are the five lines at the top of `train/train.py` (`BASE_MODEL`, `DATASET`, `IMGSZ`, `EPOCHS`, `NAME`).
CPU only? Set `EPOCHS = 30` first just to see it work.

## The dataset (`dataset/`)

Roboflow Universe, *sleep* v6 by a Roboflow user, **CC BY 4.0** (keep the attribution):
https://universe.roboflow.com/yolo-i9xsa/sleep-z8rzd/dataset/6

**Not included in this repo** (third-party images). Download it before training: on that page choose
*Download Dataset*, format **YOLOv11** (or YOLOv8), and unzip it into `dataset/` so that
`dataset/data.yaml` and `dataset/train/`, `valid/`, `test/` exist.

- 593 train / 62 valid / 26 test images, 640x640, YOLO format, classes `close_eyes` and `open_eyes`.
- **One box per eye**, not per face. Many images are close-ups.
- Small: the 26 test images give a noisy score. Judge the model on a short clip of your own camera, not only on the test table.

## On the Raspberry Pi

```bash
# copy models/eyes.onnx and the pi/ folder to the Pi, then:
sudo apt install python3-picamera2
python3 -m venv --system-site-packages ~/eyes-env && . ~/eyes-env/bin/activate
pip install -r pi/requirements.txt

python3 pi/eyes_monitor.py                      # Pi camera
python3 pi/eyes_monitor.py clip.mp4 out.mp4     # or test on a recording and save the annotated video
```

It prints a line each time the label changes and saves a photo to `snapshots/` when it turns `SLEEP`.

| Label | Meaning |
|---|---|
| `FOCUS` | eyes open, or closed less than 3 s (blinks never count) |
| `SLEEP` | eyes closed 3 s in a row |
| `NO_EYES` | no eyes found for 5 s (head down, turned away, left the seat) |

The timings and class names are the constants at the top of `pi/eyes_monitor.py`.
`python3 pi/test_timer.py` checks the timing rules (7 tests, no model needed).

**Tested here (PC):** the timing rules; the ONNX reader gives the same boxes as Ultralytics itself on two photos
(same classes and confidences, boxes within 1 px); the whole loop on a 20 s test video with scripted detections
(FOCUS, SLEEP at 3 s, FOCUS, NO_EYES; snapshot and annotated video saved).
**Not tested:** your trained `eyes.onnx` (it does not exist yet), the Pi camera path, and speed on the Pi.

## Licences

Ultralytics YOLO is AGPL-3.0: fine for coursework and private use, check before you distribute it.

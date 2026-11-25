"""Train YOLO26n on ./dataset (Roboflow: close_eyes / open_eyes) and export it to ONNX.

    pip install ultralytics
    python3 train/train.py

Result:  models/eyes.onnx   (copy this file to the Pi)
         runs/eyes/         training curves, confusion matrix, weights/best.pt

Run it on a PC or Colab with a GPU. On CPU only, set EPOCHS = 30 first to see it work.
"""
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

# ---- settings you may want to change ---------------------------------------------------
BASE_MODEL = "yolo26s.pt"   # downloaded automatically. yolo26n.pt is smaller: less RAM, faster on the Pi
DATASET = ROOT / "dataset"  # folder with data.yaml, train/, valid/, test/
IMGSZ = 640                 # eyes are small in a webcam frame; 416 is faster on the Pi but may miss them
BATCH = 4                   # images per step. "Killed" in the log = out of RAM: lower this (2), or IMGSZ
EPOCHS = 100
NAME = "eyes"               # output name: models/eyes.onnx
# -----------------------------------------------------------------------------------------


def fixed_data_yaml():
    """Roboflow writes '../train/images' in data.yaml, which Ultralytics can't resolve.
    Write a corrected copy with absolute paths and use that."""
    cfg = yaml.safe_load((DATASET / "data.yaml").read_text())
    cfg["path"] = str(DATASET)
    for split in ("train", "val", "test"):
        if split in cfg:
            cfg[split] = cfg[split].removeprefix("../")
    out = ROOT / "runs" / "data.yaml"
    out.parent.mkdir(exist_ok=True)
    out.write_text(yaml.safe_dump(cfg))
    return out


def print_report(metrics):
    """Per-class results on the held-out test images. Read these before trusting the model."""
    print(f"\n{'class':<12} {'precision':>9} {'recall':>7} {'mAP50':>6}")
    for row, class_id in enumerate(metrics.box.ap_class_index):
        p, r, ap50, _ = metrics.box.class_result(row)
        print(f"{metrics.names[class_id]:<12} {p:9.2f} {r:7.2f} {ap50:6.2f}")


def main():
    data = fixed_data_yaml()

    model = YOLO(BASE_MODEL)
    model.train(data=str(data), imgsz=IMGSZ, batch=BATCH, epochs=EPOCHS, patience=20,
                project=str(ROOT / "runs"), name=NAME, exist_ok=True)

    best = YOLO(model.trainer.best)
    print_report(best.val(data=str(data), imgsz=IMGSZ, split="test"))

    onnx = Path(best.export(format="onnx", imgsz=IMGSZ, simplify=True))
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(onnx, ROOT / "models" / f"{NAME}.onnx")
    print(f"\nDone: models/{NAME}.onnx  (input size {IMGSZ}x{IMGSZ}, classes: {best.names})")


if __name__ == "__main__":
    main()

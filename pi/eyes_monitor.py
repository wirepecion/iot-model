"""Raspberry Pi side: camera -> models/eyes.onnx -> "eyes closed for 3 seconds" -> SLEEP.

    python3 pi/eyes_monitor.py                       # Pi camera
    python3 pi/eyes_monitor.py 0                     # USB webcam number 0
    python3 pi/eyes_monitor.py clip.mp4 out.mp4      # test on a video file and save the annotated result

Labels:  FOCUS    eyes open (or closed for less than 3 s, e.g. a blink)
         SLEEP    eyes closed for 3 s in a row          -> saves a photo to snapshots/
         NO_EYES  no eyes found for 5 s (head down, turned away, left the seat)

Needs onnxruntime, opencv and numpy (pi/requirements.txt).  Pi camera also needs: sudo apt install python3-picamera2
"""
import ast
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ---- settings you may want to change ---------------------------------------------------
MODEL = ROOT / "models" / "eyes.onnx"
CLOSED_NAME, OPEN_NAME = "close_eyes", "open_eyes"   # class names in dataset/data.yaml
CONF = 0.35                  # ignore detections less sure than this
CLOSED_SECONDS = 3.0         # eyes closed this long in a row = SLEEP
NO_EYES_SECONDS = 5.0        # no eyes seen this long = NO_EYES
GAP = 0.5                    # a break shorter than this does not restart a timer (detector flicker)
SHOW_WINDOW = False          # True on a Pi with a screen (press q to quit)
# -----------------------------------------------------------------------------------------

COLORS = {"FOCUS": (80, 200, 80), "SLEEP": (60, 60, 230), "NO_EYES": (0, 150, 255)}


class Yolo:
    """Runs an Ultralytics-exported YOLO .onnx. Handles the classic (1, 4+classes, N) output
    and the end-to-end (1, K, 6) output. Class names are read from the model file itself."""

    def __init__(self, path, iou=0.5, threads=3):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.h, self.w = inp.shape[2], inp.shape[3]
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.names = {int(k): v for k, v in ast.literal_eval(meta["names"]).items()}
        self.iou = iou

    def detect(self, bgr, conf=CONF):
        """-> list of (box, confidence, class_name); box = (x1, y1, x2, y2) in frame pixels."""
        x, scale, (dx, dy) = self._letterbox(bgr)
        raw = self.sess.run(None, {self.input_name: x})[0][0]
        H, W = bgr.shape[:2]
        out = []
        for (x1, y1, x2, y2), c, cls in self._decode(raw, conf):
            box = (int(np.clip((x1 - dx) / scale, 0, W - 1)), int(np.clip((y1 - dy) / scale, 0, H - 1)),
                   int(np.clip((x2 - dx) / scale, 0, W - 1)), int(np.clip((y2 - dy) / scale, 0, H - 1)))
            out.append((box, float(c), self.names.get(int(cls), str(int(cls)))))
        return out

    def _letterbox(self, bgr):
        """Resize keeping the aspect ratio, pad with grey to the model's square input."""
        H, W = bgr.shape[:2]
        scale = min(self.w / W, self.h / H)
        nw, nh = round(W * scale), round(H * scale)
        canvas = np.full((self.h, self.w, 3), 114, np.uint8)
        dx, dy = (self.w - nw) // 2, (self.h - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(bgr, (nw, nh))
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0   # BGR -> RGB, CHW
        return np.ascontiguousarray(x), scale, (dx, dy)

    def _decode(self, raw, conf):
        if raw.shape[1] == 6 and raw.shape[0] != 4 + len(self.names):     # end-to-end: x1 y1 x2 y2 conf cls
            return [((r[0], r[1], r[2], r[3]), r[4], r[5]) for r in raw if r[4] >= conf]
        pred = raw.T                                                      # N x (4 + classes)
        cls = pred[:, 4:].argmax(1)
        score = pred[np.arange(len(pred)), 4 + cls]
        keep = score >= conf
        pred, cls, score = pred[keep], cls[keep], score[keep]
        if not len(pred):
            return []
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
        idx = np.array(cv2.dnn.NMSBoxes(boxes.tolist(), score.tolist(), conf, self.iou)).reshape(-1)
        return [((cx[i] - w[i] / 2, cy[i] - h[i] / 2, cx[i] + w[i] / 2, cy[i] + h[i] / 2), score[i], cls[i])
                for i in idx]


class Run:
    """How long a condition has been true in a row. A break shorter than GAP does not reset it."""

    def __init__(self):
        self.start = self.last = None

    def update(self, t, on):
        if on:
            if self.start is None:
                self.start = t
            self.last = t
        elif self.start is not None and t - self.last > GAP:
            self.start = None
        return 0.0 if self.start is None else self.last - self.start


class SleepTimer:
    """Turns per-frame detections into FOCUS / SLEEP / NO_EYES."""

    def __init__(self):
        self.closed, self.missing = Run(), Run()

    def update(self, t, detections):
        closed = sum(c for _, c, n in detections if n == CLOSED_NAME)
        opened = sum(c for _, c, n in detections if n == OPEN_NAME)
        closed_for = self.closed.update(t, closed > opened)
        missing_for = self.missing.update(t, closed == 0 and opened == 0)
        if closed_for >= CLOSED_SECONDS:
            return "SLEEP"
        if missing_for >= NO_EYES_SECONDS:
            return "NO_EYES"
        return "FOCUS"


class Camera:
    """Yields (BGR frame, seconds). Video files use their own clock so a clip is judged by its timeline."""

    def __init__(self, source):
        self.pi = source == "picamera"
        self.live = self.pi or source.isdigit()
        self.n, self.t0 = 0, time.monotonic()
        if self.pi:
            from picamera2 import Picamera2

            self.cam = Picamera2()
            self.cam.configure(self.cam.create_video_configuration(main={"format": "BGR888", "size": (640, 480)}))
            self.cam.start()
            self.fps = 30.0
        else:
            self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
            if not self.cap.isOpened():
                raise SystemExit(f"cannot open video source: {source}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self):
        if self.pi:
            frame = self.cam.capture_array()
        else:
            ok, frame = self.cap.read()
            if not ok:
                return None
        t = time.monotonic() - self.t0 if self.live else self.n / self.fps
        self.n += 1
        return frame, t

    def close(self):
        self.cam.stop() if self.pi else self.cap.release()


def draw(frame, detections, label):
    out = frame.copy()
    for (x1, y1, x2, y2), conf, name in detections:
        color = (60, 60, 230) if name == CLOSED_NAME else (80, 200, 80)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{name} {conf:.2f}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[label], 2)
    return out


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "picamera"
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    model = Yolo(MODEL)
    missing = {CLOSED_NAME, OPEN_NAME} - set(model.names.values())
    if missing:
        raise SystemExit(f"{MODEL.name} has classes {sorted(model.names.values())}; set CLOSED_NAME / OPEN_NAME")

    (ROOT / "snapshots").mkdir(exist_ok=True)
    timer, camera, writer, last = SleepTimer(), Camera(source), None, None
    print(f"watching {source}; Ctrl+C to stop", flush=True)
    try:
        while (item := camera.read()) is not None:
            frame, t = item
            detections = model.detect(frame)
            label = timer.update(t, detections)
            shown = draw(frame, detections, label)

            if label != last:
                print(f"{time.strftime('%H:%M:%S')}  {label:<8} (t = {t:.1f} s)", flush=True)
                if label == "SLEEP":
                    cv2.imwrite(str(ROOT / "snapshots" / f"sleep_{time.strftime('%Y%m%d_%H%M%S')}.jpg"), shown)
                last = label

            if out_path:
                if writer is None:
                    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), camera.fps,
                                             (shown.shape[1], shown.shape[0]))
                writer.write(shown)
            if SHOW_WINDOW:
                cv2.imshow("eyes", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.release()
        camera.close()


if __name__ == "__main__":
    main()

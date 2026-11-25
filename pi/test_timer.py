"""Run:  python3 pi/test_timer.py      (needs opencv + numpy + onnxruntime importable, no model file)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eyes_monitor as em

FPS = 10
CLOSED = [((0, 0, 9, 9), 0.9, "close_eyes"), ((20, 0, 29, 9), 0.8, "close_eyes")]
OPEN = [((0, 0, 9, 9), 0.9, "open_eyes"), ((20, 0, 29, 9), 0.8, "open_eyes")]
NONE = []


class Sim:
    def __init__(self):
        self.timer, self.t = em.SleepTimer(), 0.0

    def run(self, seconds, detections):
        for _ in range(round(seconds * FPS)):
            label = self.timer.update(self.t, detections)
            self.t += 1 / FPS
        return label


def test_open_eyes_are_focus():
    assert Sim().run(10, OPEN) == "FOCUS"


def test_blinks_never_trigger():
    s = Sim()
    for _ in range(20):
        s.run(0.3, CLOSED)
        assert s.run(0.7, OPEN) == "FOCUS"


def test_three_seconds_closed_is_sleep():
    s = Sim()
    assert s.run(2.8, CLOSED) == "FOCUS"
    assert s.run(0.5, CLOSED) == "SLEEP"
    assert s.run(1.0, OPEN) == "FOCUS"          # eyes open again -> recovers


def test_short_detector_dropout_does_not_reset():
    s = Sim()
    s.run(2.0, CLOSED)
    s.run(0.2, OPEN)                            # two noisy frames
    assert s.run(1.5, CLOSED) == "SLEEP"


def test_one_second_open_resets_the_timer():
    s = Sim()
    s.run(2.5, CLOSED)
    s.run(1.0, OPEN)
    assert s.run(2.5, CLOSED) == "FOCUS"


def test_no_eyes_for_five_seconds():
    s = Sim()
    assert s.run(4.5, NONE) == "FOCUS"
    assert s.run(1.0, NONE) == "NO_EYES"
    assert s.run(1.0, OPEN) == "FOCUS"


def test_mixed_detections_use_the_more_confident_side():
    one_each = [((0, 0, 9, 9), 0.9, "close_eyes"), ((20, 0, 29, 9), 0.4, "open_eyes")]
    assert Sim().run(4, one_each) == "SLEEP"


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in tests:
        f()
        print("ok  ", f.__name__)
    print(f"{len(tests)} passed")

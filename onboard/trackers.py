"""
trackers.py - single-object tracker backends.

Every backend answers the same question: "here is a patch the operator picked;
where is that same patch in this frame?"  Nothing here detects or classifies
anything - see CLAUDE.md §5 (class-agnostic) and §7 (this split).  The operator
supplies the target; these only follow it.

Two backends, and the difference that matters is not speed:

  VitTracker    a small transformer (715 KB ONNX).  Learned what makes two
                crops "the same object" across frames, so it survives lighting
                changes and rotation that break a raw-pixel match.  Crucially,
                `update()` also yields a real 0..1 match score.

  PixelTracker  OpenCV's correlation filter (CSRT).  Matches raw pixels.  This
                is what the system used exclusively until now, and it is the
                fallback.  It has no usable score: cv2.TrackerCSRT's
                getTrackingScore() returns -1.0 unconditionally - the method
                exists but is not implemented.

That score asymmetry is the whole reason ViT is primary.  A fallback can only
fire if the primary is able to say "I have lost it", and only one of these two
can say it.  Everything else - the speed, the accuracy - is secondary.

Interface (deliberately the same shape OpenCV already uses, so the detector
does not care which one it holds):

    tracker.init(frame, (x, y, w, h))  -> bool
    tracker.update(frame)              -> (ok, (x, y, w, h), score_or_None)
"""

import logging
import os

import cv2

logger = logging.getLogger(__name__)


# The ONNX weights are not in the repo (*.onnx is gitignored - they are a
# download, not source).  tools/fetch_tracker_model.py puts them here.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VIT_MODEL = os.path.join(REPO_ROOT, 'models', 'vitTracker.onnx')


class PixelTracker:
    """
    Correlation-filter tracking - "follow these pixels".

    Reports no score, so `update()` returns None in that slot and the
    detector's confidence falls back to counting consecutive failures.
    """

    name = "CSRT"
    graded = False

    def __init__(self, kind='CSRT'):
        self.name = str(kind).upper()
        # Resolve candidate factories now (cheap getattr), but do not call them
        # until we have a frame: construction can succeed where init() fails,
        # so the only honest test is to try the real init and see.
        self._factories = []
        for ns in (cv2, getattr(cv2, 'legacy', None)):
            if ns is None:
                continue
            for fname in (f'Tracker{kind}_create', 'TrackerCSRT_create',
                          'TrackerKCF_create', 'TrackerMOSSE_create'):
                fn = getattr(ns, fname, None)
                if fn is not None and fn not in self._factories:
                    self._factories.append(fn)
        self._t = None

    def init(self, frame, bbox):
        bbox = tuple(int(v) for v in bbox)
        for factory in self._factories:
            try:
                t = factory()
                t.init(frame, bbox)
                self._t = t
                self.name = factory.__name__.replace('Tracker', '').replace('_create', '')
                return True
            except Exception:
                continue
        logger.error("No working OpenCV correlation tracker found")
        self._t = None
        return False

    def update(self, frame):
        if self._t is None:
            return False, None, None
        ok, rect = self._t.update(frame)
        if not ok:
            return False, None, None
        return True, tuple(int(v) for v in rect), None


class VitTracker:
    """
    cv2.TrackerVit - the transformer tracker.  Primary backend.

    Raises on construction if the weights or the OpenCV support are missing, so
    that `make_tracker` can fall straight back to pixels at startup.
    """

    name = "ViT"
    graded = True

    def __init__(self, model_path=DEFAULT_VIT_MODEL, score_threshold=0.20):
        if not hasattr(cv2, 'TrackerVit_create'):
            raise RuntimeError(
                f"this OpenCV ({cv2.__version__}) has no TrackerVit; needs >= 4.9.0")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} missing - run tools/fetch_tracker_model.py")

        params = cv2.TrackerVit_Params()
        params.net = model_path
        params.backend = cv2.dnn.DNN_BACKEND_DEFAULT
        params.target = cv2.dnn.DNN_TARGET_CPU
        # ImageNet normalisation, on already-/255 input.
        #
        # Note: OpenCV 4.11-4.14 compute `1.0 / stdvalue` via Scalar division,
        # which is quaternion division - two of the three channels come out
        # sign-flipped.  Fixed upstream but unreleased.  Measured on our
        # footage the difference was not detectable, so this is not worth a
        # magic-number workaround; 4.10.0 is the clean pin if it ever matters.
        params.meanvalue = (0.485, 0.456, 0.406)
        params.stdvalue = (0.229, 0.224, 0.225)
        # update() returns False on its own below this score.
        params.tracking_score_threshold = float(score_threshold)

        self._params = params
        self._t = None
        logger.info(f"ViT tracker: {model_path} (threshold {score_threshold})")

    def init(self, frame, bbox):
        # A fresh tracker per target: re-init on a live TrackerVit is not
        # reliable across OpenCV versions.
        self._t = cv2.TrackerVit_create(self._params)
        self._t.init(frame, tuple(int(v) for v in bbox))
        return True

    def update(self, frame):
        if self._t is None:
            return False, None, None

        ok, rect = self._t.update(frame)
        score = float(self._t.getTrackingScore())

        # opencv#30350: TrackerVit never clips its box to the image.  On a bad
        # frame it can grow without bound - and the score threshold does NOT
        # catch it, the score can stay high while the box swells past the
        # frame.  Left alone this eventually exhausts memory mid-flight, so
        # treat any box that has outgrown the frame as a loss, and clip the
        # rest.  This is a safety clamp, not an optimisation.
        box = self._clip(rect, frame.shape)
        if box is None:
            return False, None, score
        return bool(ok), box, score

    @staticmethod
    def _clip(rect, shape):
        """Clip to the frame; None if the box is degenerate or oversized."""
        if rect is None:
            return None
        fh, fw = shape[:2]
        x, y, w, h = (int(v) for v in rect)
        if w > fw or h > fh or w <= 0 or h <= 0:
            return None
        # Plain intersection with the frame.  Do NOT pull the corner inside
        # first: a box that has drifted entirely off-screen would then survive
        # as a 1x1 sliver at the edge instead of being reported as a loss.
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(x + w, fw), min(y + h, fh)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            return None
        return (x0, y0, x1 - x0, y1 - y0)


def make_tracker(cfg):
    """
    Build (primary, fallback) from the detection config.

    Mirrors `make_proposer`: name a backend, get a soft fallback if it is not
    actually available.  Missing weights or an old OpenCV therefore degrade the
    aircraft to exactly its previous behaviour instead of failing to arm.

    Returns (primary, fallback).  `fallback` is None when the primary already
    *is* the pixel tracker - there is nothing further to fall back to.
    """
    pixels = PixelTracker(cfg.get('fallback_tracker', 'CSRT'))

    backend = str(cfg.get('tracker_backend', 'vit')).lower()
    if backend in ('vit', 'vittrack'):
        try:
            vit = VitTracker(cfg.get('vit_model', DEFAULT_VIT_MODEL),
                             cfg.get('vit_score_threshold', 0.20))
            return vit, pixels
        except Exception as e:
            logger.warning(f"ViT tracker unavailable ({e}) - tracking on pixels only")
    elif backend not in ('csrt', 'pixels', 'pixel'):
        logger.warning(f"Unknown tracker_backend '{backend}' - tracking on pixels only")

    return pixels, None

"""
trackers.py - single-object tracker backends.

Every backend answers the same question: "here is a patch the operator picked;
where is that same patch in this frame?"  Nothing here detects or classifies
anything - see CLAUDE.md §5 (class-agnostic) and §8 (this split).  The operator
supplies the target; these only follow it.

Two backends:

  PixelTracker  OpenCV's correlation filter (CSRT).  Matches raw pixels.  The
                default, and the fallback.  No score at all:
                cv2.TrackerCSRT.getTrackingScore() returns -1.0
                unconditionally - the method exists but is not implemented.

  VitTracker    a small transformer (715 KB ONNX).  Learned what makes two
                crops "the same object" across frames rather than matching raw
                pixels, and it is the only backend here whose `update()` yields
                a real 0..1 score.  Opt-in, not default - see make_tracker.

A note on why, because the obvious argument is wrong and was believed here for
a while: a fallback can only fire if the primary can say "I have lost it", and
ViT's score is the only candidate signal.  But measured, that signal is weak
(AUC ~0.70) and worse, it is weak in the way that matters.  When the tracker
drifts onto a distractor - the realistic failure, far more common than a target
vanishing - the score dips during the transition and then CLIMBS BACK as the
tracker grows confident about the wrong object.  It flags the moment of loss,
not the state of being lost.  Nano's score is outright useless (higher when
wrong than when right).

So the fallback is NOT score-driven.  It fires on consecutive update()
failures, the mechanism that was already there, and the score only derates
confidence and goes into the log.  Lost-lock recovery belongs to
SaliencyDetector re-acquisition, not to any tracker's self-report.

Interface (deliberately the same shape OpenCV already uses, so the detector
does not care which one it holds):

    tracker.init(frame, (x, y, w, h))  -> bool
    tracker.update(frame)              -> (ok, (x, y, w, h), score_or_None)
"""

import logging
import os

import cv2
import numpy as np

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


class StabilizedTracker:
    """
    Cancels camera motion before the inner tracker sees the frame.

    Found on the first real run (Pi 5, handheld, 1280x720): 5 of 6 losses were
    camera jumps of 53-194 px between processed frames, against a measured CSRT
    capture range of ~28 px for the 24 px box most selections get.  When the
    camera moves, the whole picture moves - target included - so it is the
    camera's shift that has to be taken out, not the target's motion predicted.

    Each frame: estimate the global shift against the previous frame
    (phase correlation on a small grey copy, a few ms), accumulate it, and hand
    the inner tracker a frame translated back by the running total.  The inner
    tracker then sees a near-still scene and only has to follow the residual
    plus the target's own motion.  Boxes go out in real frame coordinates.

    Also contains the opencv_contrib CSRT corner bug: a failed update() stores
    (-1,-1) as the target centre, so the next update searches the top-left
    corner and usually "finds" something there.  After any failure the inner
    tracker is re-initialised from the last frame and box it got right, so a
    single blurred frame costs one frame, not the target.
    """

    SMALL_W = 320            # width of the copy used for motion estimation

    # Below this phase-correlation peak the shift estimate is noise and is
    # treated as zero.  Calibrated against LK flow on the first Pi run (3173
    # frames): under 0.2 the estimate was off by >8 px in nearly every frame
    # (median error 15-200 px); at >=0.3 in 1.7%.  A blank frame scores ~0.07
    # with a garbage shift of half the image, which would otherwise be added to
    # the running total for good.  Out-of-range frames for a 24 px box: 292
    # uncompensated, 32 at this threshold.
    MIN_RESPONSE = 0.2

    def __init__(self, inner):
        self.inner = inner
        self.graded = getattr(inner, 'graded', False)
        self._win = None                 # Hanning window, built on first use
        self._start(None)

    @property
    def name(self):
        return self.inner.name

    def _start(self, frame):
        self._prev = None if frame is None else self._small(frame)
        self._off = (0, 0)               # accumulated camera shift, whole px
        self._acc = np.zeros(2)          # same, before rounding
        self._good = None                # (stabilised frame, box) last ok
        self._failed = False

    def _small(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        s = self.SMALL_W / float(frame.shape[1])
        return np.float32(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))

    def _shift(self, frame):
        """Camera shift since the previous frame, in full-res px."""
        cur = self._small(frame)
        prev, self._prev = self._prev, cur
        if prev is None or prev.shape != cur.shape:
            return np.zeros(2)
        if self._win is None or self._win.shape != cur.shape:
            self._win = cv2.createHanningWindow(cur.shape[::-1], cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(prev, cur, self._win)
        if not (resp >= self.MIN_RESPONSE and np.isfinite(dx) and np.isfinite(dy)):
            return np.zeros(2)
        return np.array([dx, dy]) * (frame.shape[1] / float(cur.shape[1]))

    def _stabilise(self, frame):
        """The frame translated back by the accumulated shift."""
        ox, oy = self._off
        if ox == 0 and oy == 0:
            return frame
        m = np.float32([[1, 0, -ox], [0, 1, -oy]])
        return cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_REPLICATE)

    def init(self, frame, bbox):
        self._start(frame)
        ok = self.inner.init(frame, bbox)
        if ok:
            self._good = (frame, tuple(int(v) for v in bbox))
        return ok

    def update(self, frame):
        self._acc += self._shift(frame)
        self._off = tuple(int(round(v)) for v in self._acc)
        view = self._stabilise(frame)

        if self._failed and self._good is not None:
            self.inner.init(*self._good)     # undo the corner bug
        ok, rect, score = self.inner.update(view)
        self._failed = not ok
        if not ok or rect is None:
            return False, None, score

        x, y, w, h = (int(v) for v in rect)
        self._good = (view, (x, y, w, h))
        return True, (x + self._off[0], y + self._off[1], w, h), score


def make_tracker(cfg):
    """
    Build (primary, fallback) from the detection config.

    Mirrors `make_proposer`: name a backend, get a soft fallback if it is not
    actually available.  Missing weights or an old OpenCV therefore degrade the
    aircraft to exactly its previous behaviour instead of failing to arm.

    Returns (primary, fallback).  `fallback` is None when the primary already
    *is* the pixel tracker - there is nothing further to fall back to.
    """
    pixels = StabilizedTracker(PixelTracker(cfg.get('fallback_tracker', 'CSRT')))

    # Default is pixels, NOT ViT.  Measured across 5 terrains at 8/12/20 px on
    # target, ViT is the worst of the deep trackers in exactly our size range:
    # 197/118/137 px median error, 3-10% lock survival, lost in 10 of 10 runs.
    # It is kept selectable because it is the only one whose score carries any
    # information at all, and because it is far better on a tight box than on a
    # loose one - but it does not get to be the default on that.
    backend = str(cfg.get('tracker_backend', 'csrt')).lower()
    if backend in ('vit', 'vittrack'):
        try:
            vit = StabilizedTracker(VitTracker(cfg.get('vit_model', DEFAULT_VIT_MODEL),
                                               cfg.get('vit_score_threshold', 0.20)))
            return vit, pixels
        except Exception as e:
            logger.warning(f"ViT tracker unavailable ({e}) - tracking on pixels only")
    elif backend not in ('csrt', 'pixels', 'pixel'):
        logger.warning(f"Unknown tracker_backend '{backend}' - tracking on pixels only")

    return pixels, None

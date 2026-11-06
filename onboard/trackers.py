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
confidence and goes into the log.  Lost lock is caught, and the target
re-acquired, by StabilizedTracker checking the box against what was selected
- not by any tracker's self-report.

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

    The running total only ever grows while the aircraft follows a moving
    vehicle (the ground streams past at the vehicle's speed), and the
    stabilised canvas is only frame-sized.  So when the box nears the edge of
    the canvas the inner tracker is re-based: re-initialised on the real frame
    at the real box, and the total reset to zero.  Without this a vehicle at
    15 m/s from 250 m slid off the canvas in ~15 s - one that plain CSRT holds
    indefinitely.

    Also contains the opencv_contrib CSRT corner bug: a failed update() stores
    (-1,-1) as the target centre, so the next update searches the top-left
    corner and usually "finds" something there.  After a failure the inner
    tracker is re-initialised from the last frame and box it got right, so a
    single blurred frame costs one frame, not the target.  A re-init costs
    about as much as an update, so during a longer run of failures the retries
    back off (failures 1, 2, 4, 8, 16) and the frames in between are reported
    lost without touching the poisoned tracker.  Only for trackers with no
    score: the bug is CSRT's, and ViT's score-driven failures are a signal the
    detector should keep seeing.
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

    # A rejected frame is skipped, not dropped: the next one is measured
    # against the last frame that was measured, so a jump across a blurred
    # frame is still counted.  After this many rejections in a row the scene
    # has probably changed for good, and the reference moves on.
    MAX_SKIPPED = 3

    # Lost-lock check, because CSRT has none: after a loss on the second Pi
    # run it kept reporting success on blank ceiling, a lookalike lamp or off
    # the frame ("wandering").  The box is compared with what was selected:
    # normalised cross-correlation of the CENTRE half of the init patch,
    # searched within half a box of the reported position.  Below VERIFY_MIN
    # the update is reported as a failure.
    #
    # Centre only, because the box is bigger than the target (INIT_BOX_MIN):
    # for a vehicle the ground under it changes every few frames, and on the
    # whole patch the followed-vehicle test fell to 0.52.  On the centre it
    # stays >= 0.83 while a box landed on other ground scores ~0.57.
    #
    # Labelled frame by frame on both Pi runs (2903 frames on target, 1407
    # off).  At 0.7: run 1 catches 89% of wrong frames with no false alarm;
    # run 2 catches 92% and flags 48 of 1882 correct frames, scattered (runs
    # of 1-3, nearly all on a ~12 px spot) - each costs one frame of coasting.
    VERIFY_MIN = 0.7

    # ...unless the centre is flat.  NCC of a patch with no texture is noise
    # against noise: on the fourth Pi run every window on a facade (dark
    # glass, centre std 9-13 grey levels) scored 0.2-0.5 on the very next
    # frame while CSRT held it, and was declared lost each time.  The whole
    # box scored 0.83-0.90 there.  Below TEXTURE_MIN the whole box is the
    # reference.  Cars on that run had centre std 35-70, so the calibrated
    # path above is unchanged for them.
    TEXTURE_MIN = 20.0

    # The reference the check uses is not frozen at selection: a vehicle that
    # turned ~8 degrees no longer matched it, and was lost for good (found in
    # review; TurningVehicle test).  Each frame that matches at REFRESH_MIN
    # or better blends REFRESH_RATE of what it saw into the reference.  An
    # abrupt jump onto something else scores low and is never blended in.
    REFRESH_MIN = 0.8
    REFRESH_RATE = 0.1

    # Re-acquisition after a loss: the patch taken AT SELECTION (the anchor,
    # never blended) is searched for within SEARCH_RADIUS px of where the last
    # good box should now be.  The jumps that broke the lock on the second
    # run were 96-364 px.  Accepting a match anywhere in that area needs a
    # higher bar than checking one box, and the match must be unique: no
    # second place within UNIQUE_MARGIN of it.  A row of identical lamps, or a
    # dark-over-light edge that a black shirt over a chair repeats, is a
    # guess, not a find.  The anchor rather than the blended reference
    # because a reference that drifted during a wander then "found" leaves.
    #
    # Replayed against the labels, previous code -> this (with the detector's
    # nearest-blob re-acquisition removed): run 2, selections 2-5, frames
    # locked on the wrong thing 310 -> 6, on target 938 -> 1148, "lost" while
    # the target was in view 4 -> 80.  Run 1: wrong-lock frames 34 -> 17, on
    # target unchanged.  Selection 1 of run 2 is left out: its labels are
    # marginal (box beside the puck).  Both runs are indoor and handheld;
    # nothing here is calibrated on a vehicle seen from altitude yet.
    SEARCH_RADIUS = 320
    FOUND_MIN = 0.85
    UNIQUE_MARGIN = 0.1
    COARSE_PX = 16

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
        self._skipped = 0
        self._rebase(frame, None)
        self._fails = 0                  # consecutive failed updates

    def _rebase(self, frame, box):
        """Make `frame` the stabilised canvas: zero offset, `box` the last good."""
        self._off = (0, 0)               # accumulated camera shift, whole px
        self._acc = np.zeros(2)          # same, before rounding
        self._good = None if box is None else (frame, box)

    def _small(self, frame):
        # Shrink first, then drop colour: the conversion is per pixel.
        s = self.SMALL_W / float(frame.shape[1])
        g = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if g.ndim == 3:
            g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        return np.float32(g)

    def _shift(self, frame):
        """Camera shift since the last measured frame, in full-res px."""
        cur = self._small(frame)
        prev = self._prev
        if prev is None or prev.shape != cur.shape:
            self._prev = cur
            return np.zeros(2)
        if self._win is None or self._win.shape != cur.shape:
            self._win = cv2.createHanningWindow(cur.shape[::-1], cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(prev, cur, self._win)
        if not (resp >= self.MIN_RESPONSE and np.isfinite(dx) and np.isfinite(dy)):
            self._skipped += 1
            if self._skipped > self.MAX_SKIPPED:
                self._prev, self._skipped = cur, 0
            return np.zeros(2)
        self._prev, self._skipped = cur, 0
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

    @staticmethod
    def _grey(img):
        return np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img)

    def _centre(self, box):
        """The part of a box the check compares: the centre half, or the
        whole box when the target's centre is flat (TEXTURE_MIN)."""
        if getattr(self, '_whole', False):
            return tuple(box)
        x, y, w, h = box
        cw, ch = max(4, w // 2), max(4, h // 2)
        return x + (w - cw) // 2, y + (h - ch) // 2, cw, ch

    def _match(self, frame, box):
        """Best correlation of the selected centre within half a box of `box`."""
        x, y, w, h = self._centre(box)
        m = max(box[2], box[3]) // 2
        fh, fw = frame.shape[:2]
        xa, ya = max(x - m, 0), max(y - m, 0)
        xb, yb = min(x + w + m, fw), min(y + h + m, fh)
        if xb - xa < w or yb - ya < h:
            return -1.0                  # the box has left the frame
        t = self._tmpl
        if t.shape != (h, w):
            t = cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA)
        win = self._grey(frame[ya:yb, xa:xb])
        _, best, _, (bx, by) = cv2.minMaxLoc(cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED))
        if best >= self.REFRESH_MIN:
            # Follow slow changes of look (a turning vehicle, range, light)
            # while the match is clear.  An abrupt jump onto something else
            # scores low and is never blended in.
            seen = win[by:by + h, bx:bx + w]
            if seen.shape != self._tmpl.shape:
                seen = cv2.resize(seen, self._tmpl.shape[::-1], interpolation=cv2.INTER_AREA)
            self._tmpl = cv2.addWeighted(self._tmpl, 1.0 - self.REFRESH_RATE,
                                         seen, self.REFRESH_RATE, 0.0)
        return float(best)

    def _search(self, frame):
        """After a loss: look for the selected centre around where the last
        good box should be now.  The box, or None."""
        if self._good is None:
            return None
        gx, gy, w, h = self._good[1]
        # Where the last good box's centre should be now, in real coordinates.
        px, py = gx + self._off[0] + w // 2, gy + self._off[1] + h // 2
        t = self._anchor
        th, tw = t.shape
        r = self.SEARCH_RADIUS
        fh, fw = frame.shape[:2]
        xa, ya = max(px - tw // 2 - r, 0), max(py - th // 2 - r, 0)
        xb, yb = min(px + tw // 2 + r, fw), min(py + th // 2 + r, fh)
        if xb - xa < tw or yb - ya < th:
            return None
        win = self._grey(frame[ya:yb, xa:xb])

        # Coarse pass on a shrunk copy.  On the Pi the cost follows the area
        # searched, not the template: ~22 ms for the full 664 px window at
        # any template size, 7.5 ms at half, ~2 ms at a third.  So always at
        # least half, and down until the template is COARSE_PX across.  A
        # coarse template that is too small to be unique only makes the
        # search give up (uniqueness), never accept: FOUND_MIN is checked at
        # full resolution below.
        s = min(0.5, self.COARSE_PX / float(min(tw, th)))
        ws = cv2.resize(win, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ts = cv2.resize(t, (max(4, int(round(tw * s))), max(4, int(round(th * s)))),
                        interpolation=cv2.INTER_AREA)
        if ws.shape[0] < ts.shape[0] or ws.shape[1] < ts.shape[1]:
            return None
        res = cv2.matchTemplate(ws, ts, cv2.TM_CCOEFF_NORMED)
        _, best, _, (bx, by) = cv2.minMaxLoc(res)
        # Only if it is unique: blank out the peak and look for a runner-up.
        sh, sw = ts.shape
        cv2.rectangle(res, (bx - sw, by - sh), (bx + sw, by + sh), -1.0, -1)
        if res.max() > best - self.UNIQUE_MARGIN:
            return None

        # Fine pass at full resolution around the coarse peak.
        pad = int(np.ceil(1.0 / s)) + 2
        fx, fy = int(bx / s), int(by / s)
        ref = win[max(fy - pad, 0):fy + th + pad, max(fx - pad, 0):fx + tw + pad]
        if ref.shape[0] < th or ref.shape[1] < tw:
            return None
        res = cv2.matchTemplate(ref, t, cv2.TM_CCOEFF_NORMED)
        _, best, _, (qx, qy) = cv2.minMaxLoc(res)
        if best < self.FOUND_MIN:
            return None
        cx = xa + max(fx - pad, 0) + qx + tw // 2
        cy = ya + max(fy - pad, 0) + qy + th // 2
        return (cx - w // 2, cy - h // 2, w, h)

    def init(self, frame, bbox):
        self._start(frame)
        ok = self.inner.init(frame, bbox)
        if ok:
            x, y, w, h = (int(v) for v in bbox)
            self._good = (frame, (x, y, w, h))
            self._whole = False
            cx, cy, cw, ch = self._centre((x, y, w, h))
            self._whole = float(self._grey(frame[cy:cy + ch, cx:cx + cw]).std()) < self.TEXTURE_MIN
            cx, cy, cw, ch = self._centre((x, y, w, h))
            self._tmpl = self._anchor = self._grey(frame[cy:cy + ch, cx:cx + cw])
        return ok

    def update(self, frame):
        self._acc += self._shift(frame)
        self._off = tuple(int(round(v)) for v in self._acc)
        view = self._stabilise(frame)

        if self._fails:
            found = self._search(frame)
            if found is not None and self.inner.init(frame, found):
                self._fails = 0
                self._rebase(frame, found)
                return True, found, None

        if self._fails and not self.graded:
            # Corner bug: never update a CSRT whose last update failed.
            # Retry from the last good frame on failures 1, 2, 4, 8, ...
            if self._good is None or self._fails & (self._fails - 1):
                self._fails += 1
                return False, None, None
            self.inner.init(*self._good)

        ok, rect, score = self.inner.update(view)
        if not ok or rect is None:
            self._fails += 1
            return False, None, score
        x, y, w, h = (int(v) for v in rect)
        real = (x + self._off[0], y + self._off[1], w, h)
        if self._match(frame, real) < self.VERIFY_MIN:
            self._fails += 1
            return False, None, score
        self._fails = 0

        # Re-base only when it is the offset that put the box near the edge.
        # A target genuinely at the edge of the real frame gains nothing from
        # it, and would pay a re-init every frame.
        if (self._near_edge((x, y, w, h), frame.shape)
                and not self._near_edge(real, frame.shape)):
            self._rebase(frame, real)
            self.inner.init(frame, real)
        else:
            self._good = (view, (x, y, w, h))
        return True, real, score

    @staticmethod
    def _near_edge(box, shape):
        """True when the inner tracker's search window would leave the canvas.

        CSRT searches a square of side w + 3*sqrt(w*h) around the box, i.e.
        about 2x the box edge each side for a square target."""
        x, y, w, h = box
        margin = 2 * max(w, h)
        cx, cy = x + w / 2.0, y + h / 2.0
        fh, fw = shape[:2]
        return not (margin <= cx <= fw - margin and margin <= cy <= fh - margin)


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

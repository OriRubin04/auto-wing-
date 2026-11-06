"""
Class-agnostic object proposal via visual saliency.

Replaces YOLO for the "find something at/near this point" role.

Why
---
YOLO answers "is there a car / person / dog here?" against 80 fixed COCO
classes, from a dataset of ground-level photographs.  Looking down at terrain
from altitude the answer is almost always "no", so the hint and the
re-acquisition both silently fail.

Saliency asks a different question — "what stands out from its surroundings?"
— which is what the operator actually means when clicking a target.  It needs
no model, no weights and no PyTorch, and runs in single-digit milliseconds.

Backends
--------
contrast  (default)  Difference-of-Gaussians + morphology, at FULL resolution.
spectral             cv2.saliency spectral residual.

The default is deliberately `contrast`.  OpenCV's spectral-residual detector
downsamples internally to 64x64 before computing, so small targets in a large
search window become sub-pixel and are lost.  Measured on synthetic terrain,
480 px re-acquisition window, centroid error vs. ground truth:

    target size    contrast        spectral
      14 px        2.2 px  ok      3.6 px  ok
      10 px        1.4 px  ok     78.6 px  MISS (locked onto clutter)
       6 px       24.8 px  miss   24.1 px  MISS

Spectral is ~3x faster and fine for large targets, but breaks exactly at the
size that matters here.  Below ~8 px neither works - that is a pixels-on-target
problem (resolution / lens FOV), not an algorithm problem.

All proposals are returned in FULL-FRAME pixel coordinates as (x, y, w, h).
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SaliencyDetector:
    """Finds the visually most-distinct blobs near a reference point."""

    def __init__(self, cfg=None):
        cfg = cfg or {}
        # Blob must be at least this many pixels (3x3) to count as an object.
        self.min_area = int(cfg.get('min_area', 9))
        # ...and no larger than this fraction of the search window, which
        # rejects terrain gradients and cloud edges masquerading as objects.
        self.max_area_frac = float(cfg.get('max_area_frac', 0.10))
        # Search window half-width = roi_scale * max_dist.
        self.roi_scale = float(cfg.get('roi_scale', 2.0))
        # Threshold at mean + k*std of the saliency map.  Higher = stricter.
        self.threshold_k = float(cfg.get('threshold_k', 2.5))

        self.method = str(cfg.get('saliency_method', 'contrast')).lower()
        self._spectral = None
        if self.method == 'spectral':
            try:
                self._spectral = cv2.saliency.StaticSaliencySpectralResidual_create()
            except Exception as e:
                logger.warning(f"Spectral saliency unavailable ({e}) - using contrast")
                self.method = 'contrast'
        logger.info(f"Proposal backend: saliency/{self.method}")

    # ------------------------------------------------------------------ #
    #  Public API  (mirrors the old _yolo_nearest contract)                #
    # ------------------------------------------------------------------ #

    def find_nearest(self, frame, ref_x, ref_y, max_dist=None):
        """Nearest salient blob to (ref_x, ref_y), or None."""
        cands = self.propose(frame, ref_x, ref_y, max_dist)
        return cands[0][0] if cands else None

    def propose(self, frame, ref_x, ref_y, max_dist=None):
        """
        Returns [((x, y, w, h), score, distance), ...] sorted nearest-first.
        `score` is mean saliency inside the blob - how strongly it stands out.
        """
        h, w = frame.shape[:2]

        # Work in a window around the reference point.  Faster, and "stands
        # out" is a local question - a blob 400 px away is not competing.
        half = int(max_dist * self.roi_scale) if max_dist else max(w, h)
        x0 = max(0, int(ref_x) - half)
        y0 = max(0, int(ref_y) - half)
        x1 = min(w, int(ref_x) + half)
        y1 = min(h, int(ref_y) + half)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return []
        roi = frame[y0:y1, x0:x1]

        sal = self._saliency_map(roi)

        # Adaptive threshold: scales with how busy the terrain is.
        thresh = float(sal.mean()) + self.threshold_k * float(sal.std())
        mask = (sal >= thresh).astype(np.uint8) * 255

        # Drop isolated noise pixels, then close small gaps within a blob.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        max_area = (x1 - x0) * (y1 - y0) * self.max_area_frac

        cands = []
        for i in range(1, n):                      # 0 is background
            bx, by, bw, bh, area = stats[i]
            if area < self.min_area or area > max_area:
                continue
            cx = x0 + float(centroids[i][0])
            cy = y0 + float(centroids[i][1])
            dist = ((cx - ref_x) ** 2 + (cy - ref_y) ** 2) ** 0.5
            if max_dist is not None and dist > max_dist:
                continue
            # Mean saliency of just this blob, computed inside its bounding
            # box so the mask compare stays cheap.
            sub_lab = labels[by:by + bh, bx:bx + bw]
            sub_sal = sal[by:by + bh, bx:bx + bw]
            score = float(sub_sal[sub_lab == i].mean())
            cands.append(((x0 + int(bx), y0 + int(by), int(bw), int(bh)), score, dist))

        cands.sort(key=lambda c: c[2])             # nearest first
        return cands

    # ------------------------------------------------------------------ #
    #  Saliency maps                                                       #
    # ------------------------------------------------------------------ #

    def _saliency_map(self, bgr):
        if self.method == 'spectral' and self._spectral is not None:
            ok, sal = self._spectral.computeSaliency(bgr)
            if ok:
                return sal.astype(np.float32)
        return self._contrast_map(bgr)

    @staticmethod
    def _contrast_map(bgr):
        """
        Difference of Gaussians at native resolution.

        The narrow blur keeps object-scale detail; the wide blur models the
        local terrain brightness.  Their absolute difference responds to
        anything breaking the local pattern - dark-on-light and light-on-dark
        alike - which is exactly "stands out to the eye".
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        near = cv2.GaussianBlur(gray, (0, 0), 1.2)
        far = cv2.GaussianBlur(gray, (0, 0), 6.0)
        dog = np.abs(near - far)
        peak = float(dog.max())
        return dog / peak if peak > 1e-6 else dog

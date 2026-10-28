# auto-wing — design constraints

Vision-based click-to-track for a fixed-wing UAV (TalonPro). The operator clicks
a target on the GCS video; the onboard computer tracks it and steers the
aircraft via RC override to ArduPilot.

Onboard computer: Raspberry Pi 5 (8 GB) + USB camera.
Operating altitude: mostly ~250 m, up to 500 m.

---

## 1. Autonomy — the binding architectural rule

**All computation runs onboard. Nothing in the tracking or control path may
depend on the radio link.**

The link has exactly two jobs:

| Direction | Purpose | Required? |
|---|---|---|
| GCS → aircraft | send the operator's target selection | once, at selection |
| aircraft → GCS | video, so the operator can see and click | convenience only |

Consequences that follow from this, and must not be traded away:

- Once a target is selected, the aircraft completes the whole task with the
  link dead. Detection, tracking, PID and RC override are all local.
- **Video loss is cosmetic, never a fault.** Dropped frames degrade what the
  operator sees. They must never degrade what the aircraft does.
- The control loop must not block on any network I/O. Streaming and
  registration belong off the hot path.
- Do not "improve" reliability by adding a GCS round-trip to the control path.
  No remote computation, no ack-before-act, no telemetry-gated decisions.

Current state: after target selection the aircraft *does* keep flying with the
link down. But the control loop still pays a network tax — `check_registration()`
blocks ~10 ms every frame, and `send_frame()` does JPEG encode plus `sendto`
inline. Roughly 40% of the frame budget. Moving both to a background thread is
outstanding work, and is what fully honours this rule.

## 2. Flight safety ordering

When logging, streaming or telemetry compete with the control loop, **the
control loop wins.** Drop the log frame, drop the video frame, skip the
telemetry read. Never stall the loop.

Two fixes already made under this rule:
- Mode changes are non-blocking (they used to freeze everything for up to 12 s).
- Flight-log video goes through a bounded queue and drops frames when full.

## 3. ACRO mode semantics

Tracking flies in **ACRO (mode 4)**. In ACRO the roll stick commands a roll
*rate*, not a bank angle — **zero stick does not level the wings**, it holds the
current bank. Wing levelling is therefore done explicitly: read the bank angle
from `ATTITUDE` and command counter-roll.

FBWA would self-level, but has been explicitly rejected.

Stop switches to **RTL (mode 11)** and releases the RC override, so ArduPilot
owns throttle from that point. Throttle is **0%** while tracking in ACRO.

## 4. Confidence decay while coasting

When the tracker fails, the detector keeps reporting the last known box for up
to `coast_limit` (20) frames. `ObjectDetector.confidence()` decays 1.0 → 0.0
across that window and `_send_control` scales roll and pitch by it, so the
aircraft winds down smoothly instead of steering at full authority on a position
that can be ~2 s old.

Attitude-based wing levelling is deliberately **not** derated — it comes from a
live `ATTITUDE` reading, not from tracker pixels.

Lost-lock currently snaps to `roll=0`. Switching it to RTL was considered and
deferred.

## 5. Detection is class-agnostic

Targets are "whatever stands out to the eye", not a fixed class list. YOLO was
replaced by saliency (`onboard/saliency.py`) because COCO classes are close to
useless looking down at terrain from altitude. YOLO remains available as an
opt-in backend (`detection.backend: yolo`) with a lazy import.

Default saliency method is `contrast` (DoG at native resolution). `spectral`
downsamples to 64×64 internally and misses small targets — measured, it loses a
10 px target completely where `contrast` finds it to 1.4 px.

## 6. Pixels on target is the real constraint

Detection range is set by ground sample distance, not by the algorithm:

```
metres_per_pixel = (2 x altitude x tan(HFOV/2)) / horizontal_pixels
```

A 4.5 m vehicle with a 60° lens:

| | 250 m | 500 m |
|---|---|---|
| 640 px | 10 px | 5 px |
| 1920 px | 30 px | 15 px |
| 1920 px + 30° lens | 64 px | 32 px |

Below ~8 px nothing works, and that is not fixable in software. Resolution and
lens FOV buy more altitude than any model change. People (~0.5 m) are not
achievable at these altitudes; vehicles are.

## 7. The init box matters more than the tracker

This is the most useful thing measured so far, and it was found only after
betting wrongly on the opposite.

With the old flat `INIT_PATCH_SIZE = 72`, **every** tracker tested — CSRT, KCF,
MOSSE, Nano, ViT, DaSiamRPN — loses a 10 px target within ~20 frames. The
diagnosis is in one number: they all drift at exactly the aircraft's ground
speed in px/frame, i.e. **they are tracking the terrain inside the box.** An
8 px target is 0.57% of a 72×72 patch, so the correlation peak belongs to the
ground. Scaling the box to the target takes CSRT from 1% to 75% lock survival
at 8 px *with no change of tracker* — a bigger win than any model swap, and it
helps whichever backend runs.

So the box is sized from the proposal now (`_box_size`), not from a constant.

`INIT_BOX_SCALE` is **1.25**, and it is scaled onto the saliency hint, not onto
the target — we never know the true footprint in flight. 1.25 survived a paired
sign test (every scale run on the same 10 clips, so terrain variance cancels):
across 0.5–4.0 at 8/12/20/30 px, on **both** CSRT and ViT, **no scale beats it
anywhere.** ViT additionally collapses above ~2.0 (68% → 4% lock at 8 px), so
widening would break the backend we kept selectable. Leave it at 1.25.

**The thing to actually know: the hint is set by terrain texture, not by the
target.** Same targets, same estimator, different ground:

| true target | 8 | 12 | 20 | 30 |
|---|---|---|---|---|
| hint over textured ground | 7 | 11 | 14 | 20 |
| hint over smooth ground | 19 | 25 | 41 | 59 |

So at scale 1.25 the box lands at **0.85–1.17× the true target over textured
ground and 2.5–3.0× over smooth ground** — a 3× swing that no single scale
value can absorb. `threshold_k` moves it as well (k=1.0 → 12/18/27/37;
k=3.0 → 6/9/13/6).

An earlier version of this section gave only the smooth-ground numbers and
attributed them to the DoG filter scale ("bottoms out near 20 px"). That was a
property of the calibration scene, not of the estimator. **box/true is the
rig-independent quantity; the hint is not.** Map through your own footage
before trusting any scale number here.

Consequence of `INIT_BOX_MIN = 24`: over textured ground it clamps 8, 12 *and*
20 px — the whole range §6 cares about — so the scale is inert below ~30 px
there and a flat 24 px box is what actually flies. Not harmful on the evidence
(scale 2.0 did not separate from 1.25 in the paired test), but tuning the scale
at those sizes will look like it does nothing, which is confusing rather than
wrong.

## 8. Tracking backends, and the argument that did not survive

`onboard/trackers.py` holds two backends behind one interface.
`detection.tracker_backend` picks the primary — **default `csrt`**, with `vit`
opt-in.

The tempting argument, which was believed here and acted on: a fallback can
only fire if the primary can say "I have lost it"; `cv2.TrackerCSRT.
getTrackingScore()` returns `-1.0` unconditionally (exists, not implemented);
ViT is the only backend reporting a real score; therefore ViT must be primary.

Both halves failed measurement:

- **ViT is the worst deep tracker in our size range.** Across 5 terrains at
  8/12/20 px: 197/118/137 px median error, 3–10% lock survival, lost in 10 of
  10 runs. It is very sensitive to box scale — much better tight than loose —
  but it does not get to be the default on that.
- **Its score flags the wrong moment.** AUC ~0.70, and when the tracker drifts
  onto a distractor — the realistic failure, not a target vanishing — the score
  dips during the transition and then *climbs back* as it grows confident about
  the wrong object. Nano's score is worse than useless: higher when wrong
  (0.854) than when right (0.822).

**No tracker here has a usable lost-lock signal.** The fallback therefore fires
on consecutive `update()` failures, the mechanism already in place; the score
only derates confidence and goes in the log. Recovery belongs to
`SaliencyDetector` re-acquisition.

Most accurate measured at 8–20 px was **TrackerNano** (2.5–3.7 px, 89–100% lock,
0/10 losses at 20 px) at ~0.2× CSRT's cost. Not adopted yet: that ratio is x86,
OpenCV's NEON path is weaker than its AVX2 path, and there is still **no Pi
measurement of anything.**

Two failures, deliberately handled differently:

- **Weights or OpenCV missing** → `make_tracker` logs it and runs pixels only;
  never blocks arming. The `.onnx` is gitignored, so a fresh clone behaves as
  before until `tools/fetch_tracker_model.py` runs.
- **Lost lock in flight** → `_handover` gives the last good box to CSRT and
  carries on, one-way per target, instead of abandoning the target.

Confidence after a handover is capped at `degraded_confidence` (0.60): the
handover resets `fail_count`, which would otherwise hand full authority straight
back to the tracker that cannot report being wrong.

Two upstream ViT bugs are worked around in `VitTracker`:
- **opencv#30350** — it never clips its box to the frame, and the score
  threshold does not catch the runaway: 30×30 → 1416×912 observed, with latency
  2.6 → 47.6 ms exactly when the tracker is already lost. The old `w > 5 and
  h > 5` guard catches the collapse-to-zero case but *not* the growth. `_clip`
  rejects any box that has outgrown the frame. Safety, not tidiness.
- **opencv#30164** — `1.0 / Scalar` is quaternion division, so two of three
  normalisation channels are sign-flipped in 4.11–4.14. No detectable
  difference measured; 4.10.0 is the clean pin if it ever matters.

Caveat on all of the above: **synthetic terrain, x86, 5 s clips.** No rolling
shutter, JPEG artefacts, haze, parallax or real vehicle signature. It is enough
to rule things out, not enough to pick a flight configuration. Real footage
through `tools/benchmark_detector.py` is still the gate.

## 9. Practical notes

- `config/config.yaml` has **uncommitted local modifications** on the operator's
  machine. Prefer code defaults (`cfg.get(...)`) over editing it, to avoid pull
  conflicts.
- `origin` is GitHub. Work happens on `claude/tender-hamilton-Dz0Se`.
- Never commit `__pycache__` — a stale `.pyc` shadowing an updated `.py` once
  made `git pull` appear to succeed while old code kept running.
- Development is currently on Windows + SITL. The Pi is not in use yet, so
  `CAP_DSHOW` is the correct backend for now. MJPEG fourcc is still required
  the moment resolution exceeds 640×480 — uncompressed 720p exceeds USB 2.0
  bandwidth and the driver silently drops to ~5 fps rather than erroring.

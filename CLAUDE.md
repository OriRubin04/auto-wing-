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

## 7. Tracking: deep primary, pixels as the fallback

`onboard/trackers.py` holds two backends behind one interface. `detection.
tracker_backend` picks the primary (default `vit`); `csrt` forces pixels only.

| | reports a score? |
|---|---|
| `cv2.TrackerVit` — 715 KB transformer, primary | **yes, 0…1** |
| `cv2.TrackerCSRT` — correlation filter, fallback | no |

**The score is the reason ViT is primary, not the speed and not the accuracy.**
A fallback can only fire if the primary can say "I have lost it".
`cv2.TrackerCSRT.getTrackingScore()` returns `-1.0` unconditionally — the method
exists but is not implemented — so a CSRT-only system cannot know when to
degrade. NanoTrack is faster than both and was rejected for the same reason: its
score barely separates present from absent, and `update()` always returns true.

Two failures, deliberately handled differently:

- **Weights or OpenCV missing** → `make_tracker` logs it and runs pixels only.
  Degrades to the previous behaviour; never blocks arming. The `.onnx` is
  gitignored, so a fresh clone behaves exactly as before until
  `tools/fetch_tracker_model.py` has been run.
- **Lost lock in flight** → `_handover` gives the last good box to CSRT and
  carries on. One-way per target: no flapping between backends on a target the
  deep tracker has already shown it cannot hold.

Confidence after a handover is capped at `degraded_confidence` (0.60). Without
that cap the handover resets `fail_count` and hands full authority straight back
to the tracker that *cannot* report being wrong.

Score band: measured, a correct ViT lock scores ~0.47–0.55 and an absent target
collapses to ~0.07, so `score_confident` is 0.35 — high enough to leave a
healthy lock at full authority, low enough to derate a marginal one. Setting it
near a good lock's own score taxes every command for no reason.

Two upstream bugs are worked around, both in `VitTracker`:
- **opencv#30350** — ViT never clips its box to the frame and the score
  threshold does not catch the runaway, so it can grow until memory runs out.
  `_clip` rejects any box that has outgrown the frame. Safety, not tidiness.
- **opencv#30164** — `1.0 / Scalar` is quaternion division, so two of three
  normalisation channels are sign-flipped in 4.11–4.14. Measured no detectable
  difference on our footage; 4.10.0 is the clean pin if it ever matters.

Still unknown, and it is the open question: **no published measurement exists of
any deep tracker below ~20 px on target.** These nets were trained on objects
filling a good fraction of the crop. ViT may well do *worse* than CSRT's HOG at
§6 sizes. Hence pixels are kept, and `INIT_PATCH_SIZE` (72 px, tuned for CSRT)
is deliberately left alone — a deep tracker squeezes the init box into a fixed
template, so 72 px around a 10 px target is almost all background. Shrinking it
is the next experiment, not part of the swap.

## 8. Practical notes

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

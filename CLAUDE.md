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

When CSRT fails, the detector keeps reporting the last known box for up to
`coast_limit` (20) frames. `ObjectDetector.confidence()` decays 1.0 → 0.0 across
that window and `_send_control` scales roll and pitch by it, so the aircraft
winds down smoothly instead of steering at full authority on a position that can
be ~2 s old.

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

## 7. Practical notes

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

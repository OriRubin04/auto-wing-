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

Current state: **done.** Both halves of the network tax are off the loop.

- `check_registration()` is a non-blocking `recvfrom` on a non-blocking socket.
  It used to carry a 10 ms timeout and pay it on nearly every frame, since the
  common case is that no packet is waiting.
- `send_frame()` now only computes `last_scale` and hands a copy of the frame
  to a sender thread. The resize, the JPEG encode and the `sendto` all happen
  there. Measured at 1280x720 on x86: the loop went from **1.19 ms to 0.25 ms**
  per streamed frame, and that 0.25 ms is flat whether the GCS is listening,
  gone, or a blackhole address. It is one memcpy and nothing else.

`last_scale` is deliberately computed on the loop rather than on the thread: it
depends only on the frame shape and `max_width`, and `_handle_gcs_message`
reads it to map a click back to frame pixels. Setting it from the thread would
be a race on the one value that decides where the operator actually pointed.

An earlier version of this section put the tax at "roughly 40% of the frame
budget". That was the 10 ms `check_registration()`, and it has not been true
since that was fixed. The remaining win measured at under 1 ms on x86. The
reason to take it anyway is that the encode is the part that grows on the Pi's
weaker NEON path, and that a stalled `sendto` can no longer reach the loop at
all — which is the rule, not the millisecond.

**Not worth doing: shrinking the UDP chunk size.** It is listed as a known
issue on PR #1 and the arithmetic does not support it. A ~30 kB JPEG is ~21 IP
fragments at a 1500 byte MTU, or ~23 packets if chunked at 1400 by hand, and a
JPEG needs every byte either way — one lost packet loses the frame in both
schemes. Robustness against loss needs FEC or a lower quality setting, not a
smaller chunk. Fewer, larger packets at least cost fewer syscalls.

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

### Large targets: the hint was a speck (third Pi run)

Cars from a 20th floor are ~40x80 px. The DoG sees structure up to ~12 px, so
a car is its rim and its glints in fragments, and the nearest fragment - often
a 5x3 px reflection - set the box: the 24 px floor on one part of the car.
`SaliencyDetector.object_at` now sizes the box first: the pixels around the
click that stand out from a sigma-25 background the same way, at least half as
strongly, snapped to within 10 px of the click; refused if it leaks to the
window edge, and the DoG blob is the fallback. The box is no longer square -
each side is 1.25x the object's own side, clamped to 24..96.

On the run's 8 car selections it gets 0.87-1.0 of the true length on 6 (DoG
hint: 0.06-0.5). Replayed on the Pi: the old box drifted onto the road and
kept reporting lock (sel 19, 21); the new one stays on the car or reports lost.
Cost: CSRT on the bigger box, 29.6 -> 39.0 ms median per tracked frame. Seen
but not fixed: on the Pi CSRT's scale estimate sometimes grows the box well
past the car (sel 19, ~60x130).

Synthetic 8-30 px targets come out at their true size (the DoG hint was
+10-15 px), so below ~19 px the box is the 24 px floor as before. Not yet
checked on real footage from altitude.

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
only derates confidence and goes in the log. Lost lock is caught by checking
the box against what was selected, not by the tracker (see "Wandering" below).

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

### Camera shake: the first real-footage finding

The first Pi run (handheld, 1280x720, 13 selections) lost lock 6 times, and
the cause was not the tracker's appearance model. Two things, both measured:

- **The picture jumps further than CSRT can search.** For a 24 px box CSRT
  holds ~28 px of shift between frames and loses everything past ~48 px
  (search side is `w + 3·√(wh)`, measured range is below its ±2w maximum).
  The losses were at 53-194 px. When the camera moves the whole image moves,
  so `StabilizedTracker` takes the camera's shift out (phase correlation on a
  320 px grey copy) before the inner tracker sees the frame.
- **CSRT has a corner bug.** A failed `update()` stores (-1,-1) as the centre
  before returning false, so the next update searches the top-left corner and
  usually reports ok there. The wrapper never updates a CSRT whose last update
  failed: it re-inits from the last good frame, backing off on failures
  1, 2, 4, 8, 16. Note what this does *not* change: CSRT still reports ok after
  a re-init, now near the last good box rather than in the corner, so
  `fail_count` still resets and still says little about whether the lock is
  right (no lost-lock signal, above).

One trap the first version fell into, caught in review: the accumulated
offset only grows while the aircraft follows a moving vehicle, because the
ground streams past at the vehicle's speed. The stabilised canvas is
frame-sized, so the target slid off it - a 15 m/s vehicle at 250 m was lost in
~15 s where plain CSRT held it indefinitely. The wrapper now re-bases (re-init
on the real frame, offset back to zero) when the offset pushes the box near
the canvas edge. `tests/` covers it; any change to the stabiliser has to keep
the followed-vehicle test green, not just the shake ones.

Replayed on the same clip: old code reproduces all 6 live losses at the same
frames; new code holds 5 and loses the sixth (heavy blur) 100 frames later.
`MIN_RESPONSE = 0.2` is calibrated against LK flow on that clip. Cost on the
Pi: +5.4 ms median per update for the first version, against a tracking loop
that already ran at 33.6 ms. Tests: `python -m unittest discover tests`.

### Wandering: the second real-footage finding

Second Pi run (handheld, 6 selections, labelled frame by frame by agents).
The stabiliser helped, but after a jump it could not measure (rotation, heavy
blur, 96-364 px) CSRT lost the target and kept reporting success on blank
ceiling, a neighbouring lamp, or off the frame. Worse, after 20 failed frames
the detector re-initialised on the nearest salient blob and reported a clean
lock on a table, a vase, a shirt. Now:

- **Check every box.** `StabilizedTracker` compares the centre half of the
  box with a reference patch (NCC within half a box); below 0.7 the update is
  a failure. Centre only, because with `INIT_BOX_MIN` the ground under a
  vehicle fills the rest and changes as it drives.
- **The reference follows slow change.** Frozen at selection, a vehicle that
  turned ~8° no longer matched and was lost for good - caught in review,
  `TurningVehicle` test. Clear matches (≥0.8) blend 10% into it.
- **Re-acquire by appearance, against the selection.** During a loss the
  patch taken at selection (never blended) is searched within 320 px,
  coarse-to-fine, and only a unique match counts. Using the blended reference
  here let a drifted one "find" a leaf.
- **The nearest-blob re-acquisition in the detector is gone.**

Replayed against the labels (run 2, selections 2-5): wrong-lock frames
310 → 6, on target 938 → 1148, "lost" while in view 4 → 80. Run 1: 34 → 17.
Pi 5: +1 ms per locked frame; the search is 3.2 ms median, 8.3 ms p95 (cost
follows the area searched, ~22 ms at full resolution, hence coarse first).

Flat targets (fourth Pi run, windows on a facade): the centre of a window
is dark glass, centre std 9-13 grey levels, and NCC there is noise against
noise - 0.2-0.5 on the next frame while CSRT held the window, so every
window was declared lost at once. Below `TEXTURE_MIN = 20` the whole box is
the reference instead (0.83-0.90 there). Replayed: windows 1 frame -> held
until the next click; labelled runs 1 and 2 unchanged or better (run 2 false
lock 11 -> 3, 11 fewer hits).

At night that was not enough (fifth Pi run): the picture is dark and noisy and
even the whole box scored 0.57-0.74 on the next frame. A fixed blur before
the NCC cleaned that up but let wrong locks through on run 2 (3 -> 68), so the
bar is per target instead: the first 3 updates after a selection measure what
"the same thing" scores in this scene, and the bar is 0.85x their median,
clamped to 0.45..0.7. Runs 1 and 2 are identical (their targets score >= 0.82,
so the bar stays 0.7); night windows went from 1 locked frame to held while in
view (e.g. 1/62 -> 62/62).

Open, deliberately: the search only runs during the 20-frame coast, so a
target out of view for longer still needs a re-click. And every threshold
here is from indoor handheld footage - recalibrate on a vehicle from altitude
before trusting them in flight.

## 9. Verify before asserting

**Operator's standing instruction: research and check before giving an answer,
every time.** Hardware behaviour, device names, parameter names and API
signatures all move, and this project is now at the point where a wrong one
gets wired into an aircraft.

The rule, concretely:

- Check a primary source before stating a fact about hardware, an OS release,
  ArduPilot parameters or a library API. Not memory.
- If the source cannot be reached, say that it could not be reached and that
  the claim is unverified. Do not fill the gap from memory and present the
  result as settled.
- Do not let uncertainty about a detail drive a design choice. Resolve the
  detail.

That last one is from a real failure here. The Pi guide recommended buying a
USB-TTL adapter and routed the whole wiring section around it, because
`/dev/serial0` on a Pi 5 was unclear. The actual answer is one search away: on
a Pi 5 `serial0` aliases the separate 3-pin debug connector, the GPIO14/15
UART is **`/dev/ttyAMA0`**, and the Linux console already sits on the debug
connector — so unlike Pi 3/4 there is nothing to free in `raspi-config`, and
the direct three-wire connection every companion-computer setup uses is both
correct and simpler. An unverified worry had been turned into a shopping list
and an extra flight-critical connector.

## 10. Practical notes

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
- Camera frame rate (sixth Pi run, USB global-shutter camera): one capture
  buffer made V4L2 skip every other frame (15 fps); two give 30. And a
  camera's own auto-exposure lengthens exposure in the dark until the frame
  rate falls (2.5 fps measured). `onboard/exposure.py` sets exposure itself,
  gain first and never past 90% of the frame period, with ranges read from
  whatever camera is plugged in; a camera without manual exposure keeps its
  own. Linux only. Measured after: 29.8 fps, worst gap 65 ms, in the same
  room. Not yet checked outdoors or in real darkness.
- Pi deployment (install, wiring to the flight controller, FC parameters,
  bench-test order) is written up in `docs/raspberry-pi-setup.md`, and
  `tools/install_pi.sh` is the one-command setup. Nothing there has been run on
  real hardware yet either — it is the plan, not a report.

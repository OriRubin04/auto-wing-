# Auto-Wing — Vision Tracking for Fixed-Wing Aircraft

Track any object with your fixed-wing drone using a camera + AI.  
Click a target on your laptop screen → the aircraft banks and follows it.

---

## What You Need

- Windows laptop (for SITL simulation / GCS)
- Python 3.10 or 3.11 installed
- Git installed
- A webcam (USB) — or just use a video file for testing

---

## Part 1 — Install Python & Dependencies

### 1. Clone the repo
```bash
git clone https://github.com/OriRubin04/auto-wing-.git
cd auto-wing-
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac
```

### 3. Install Python packages
```bash
pip install -r requirements.txt
```

This installs:
- `ultralytics` — YOLOv8 object detection
- `opencv-python` — camera, video, tracking
- `pymavlink` — talk to ArduPilot flight controller
- `pyyaml` — config file reading
- `numpy` — math

> First run will auto-download the YOLOv8 nano model (~6 MB). You need internet once.

---

## Part 2 — Install ArduPlane SITL (Simulation)

SITL = Software In The Loop. It simulates a real flight controller on your PC.  
You test everything safely before touching real hardware.

### Option A — WSL2 (Best for Windows)

**Step 1: Enable WSL2**
Open PowerShell as Administrator and run:
```powershell
wsl --install
```
Restart your PC when asked.

**Step 2: Open Ubuntu (WSL) and install ArduPilot**
```bash
git clone https://github.com/ArduPilot/ardupilot.git --depth 1
cd ardupilot
git submodule update --init --recursive
Tools/environment_install/install-prereqs-ubuntu.sh -y
source ~/.profile
```
This takes 5-15 minutes.

**Step 3: Run the simulator**
```bash
cd ardupilot/ArduPlane
sim_vehicle.py -v ArduPlane --console --map
```

A map window and console will open. The simulator listens on `TCP port 5760`.  
Leave this running.

---

### Option B — Windows Binary (Easier, less features)

1. Download Mission Planner: https://ardupilot.org/planner/docs/mission-planner-installation.html
2. Open Mission Planner → go to **Simulation** tab → select **Plane** → click **Start**
3. SITL starts automatically on port 5760

---

## Part 3 — Run the Tracking System

You need **two terminal windows** open at the same time.

### Terminal 1 — Onboard Script (runs on Pi in real life, laptop for SITL)

```bash
cd auto-wing-
venv\Scripts\activate
python onboard/main.py
```

You should see:
```
Connecting to tcp:127.0.0.1:5760 ...
Heartbeat received from sysid=1
Mode set to GUIDED
Ready. Waiting for GCS target selection.
```

### Terminal 2 — GCS Script (your laptop screen)

```bash
cd auto-wing-
venv\Scripts\activate
python gcs/gcs_client.py
```

A window opens showing the live camera feed (or black if no camera yet).

---

## Part 4 — Select a Target and Track

1. Point your webcam at something (a person, a bottle, anything)
2. In the GCS window — **left-click on the object**
3. A green box appears around it
4. Watch the SITL map — the simulated aircraft will bank toward it

**Keyboard shortcuts in GCS window:**
| Key | Action |
|-----|--------|
| Left click | Select / change target |
| `S` | Stop tracking (hold wings level) |
| `Q` | Quit |

---

## Part 5 — Test Without a Camera

If you don't have a webcam handy, point the camera at a video file:

Edit `config/config.yaml`:
```yaml
camera:
  index: "path/to/your/video.mp4"   # change this line
```

Or use your laptop's built-in camera:
```yaml
camera:
  index: 0   # 0 = first camera, 1 = second, etc.
```

---

## Part 6 — Tune the PID (When It Overshoots or Is Sluggish)

Edit `config/config.yaml` under `control:`:

```yaml
control:
  roll_pid:
    kp: 0.4    # increase = reacts faster, but may oscillate
    ki: 0.01   # increase = removes steady error, but may overshoot
    kd: 0.05   # increase = dampens oscillation
  pitch_pid:
    kp: 0.2
    ki: 0.005
    kd: 0.02
  throttle: 0.6   # 0.0 to 1.0
```

**Start conservative** (low kp) and increase slowly.

---

## Moving to Real Hardware (Raspberry Pi)

When you're ready to go from SITL to real flight:

1. Copy the whole `auto-wing-` folder to your Raspberry Pi
2. Install the same dependencies on the Pi (`pip install -r requirements.txt`)
3. Connect Pi to flight controller via USB or UART
4. Edit `config/config.yaml`:
   ```yaml
   mavlink:
     connection_string: "/dev/ttyUSB0"   # USB → FC
     # or: "/dev/ttyAMA0"               # UART GPIO pins
     baudrate: 115200

   gcs:
     onboard_ip: "192.168.1.xxx"   # Pi's IP address on your WiFi
   ```
5. Run `python onboard/main.py` on the Pi
6. Run `python gcs/gcs_client.py` on your laptop

---

## Common Errors

| Error | Fix |
|-------|-----|
| `No module named 'ultralytics'` | Run `pip install -r requirements.txt` again |
| `Camera not found` | Check `camera.index` in config.yaml (try 0, 1, 2) |
| `Connection refused tcp:127.0.0.1:5760` | SITL is not running — start it first |
| `Heartbeat timeout` | Wrong connection string or SITL crashed — restart SITL |
| GCS window is black | Camera index wrong, or onboard script not running yet |
| Aircraft not moving | Check ArduPilot is in GUIDED mode in Mission Planner |

---

## File Structure

```
auto-wing-/
├── config/config.yaml        ← all settings (edit this)
├── onboard/
│   ├── main.py               ← run this on Pi / laptop
│   ├── detector.py           ← YOLO + CSRT tracker
│   ├── controller.py         ← PID math
│   ├── mavlink_interface.py  ← talks to flight controller
│   └── streamer.py           ← sends video to GCS
├── gcs/
│   └── gcs_client.py         ← run this on your laptop
├── simulation/
│   └── sitl_setup.md         ← detailed SITL notes
└── requirements.txt
```

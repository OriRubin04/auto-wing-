# Auto-Wing — Vision Tracking for Fixed-Wing Aircraft

Track any object with your fixed-wing drone using a camera + AI.  
Click a target on your laptop screen → the aircraft banks and follows it.

> **All commands below are for Windows CMD** (the black Command Prompt window).  
> To open CMD: press `Windows + R`, type `cmd`, press Enter.

---

## What You Need

- Windows laptop
- Python 3.11 installed (see Part 1)
- Git installed (see Part 1)
- A webcam (USB or built-in)

---

## Part 1 — One-Time Installs

### Install Git
1. Go to: **https://git-scm.com/download/win**
2. Download and run the installer — click **Next** on everything, don't change anything
3. After install, close and reopen CMD

### Install Python 3.11
1. Go to: **https://www.python.org/downloads/**
2. Click the big **Download Python 3.11.x** button
3. Run the installer
4. **IMPORTANT:** Check the box **"Add Python to PATH"** at the bottom before clicking Install
5. After install, close and reopen CMD

### Verify both installed
Open CMD and type:
```
python --version
git --version
```
Both should show a version number. If not, restart your PC and try again.

---

## Part 2 — Download the Project

In CMD, type these commands one by one:
```
cd Desktop
git clone https://github.com/OriRubin04/auto-wing-.git
cd auto-wing-
```

Now you have an `auto-wing-` folder on your Desktop.

---

## Part 3 — Install Python Packages

Still in CMD inside the `auto-wing-` folder:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

You will see `(venv)` appear at the start of the line — that means it worked.  
The install takes a few minutes. Wait until it finishes.

> The first time you run the tracking script, it will auto-download the AI model (~6 MB). You need internet once for that.

---

## Part 4 — Install the Flight Simulator (SITL)

SITL = Software In The Loop. It simulates a real flight controller on your PC so you can test safely without touching real hardware.

**Use Mission Planner — easiest on Windows:**

1. Download Mission Planner: **https://ardupilot.org/planner/docs/mission-planner-installation.html**
2. Install it (click Next on everything)
3. Open Mission Planner
4. At the top, click the **Simulation** tab
5. Under vehicle type, select **Plane**
6. Click **Start** — a map and console window will open
7. Leave it running in the background

The simulator is now running on your PC on port 5760. The tracking scripts connect to it automatically.

---

## Part 5 — Run the Tracking System

You need **two CMD windows** open at the same time.

### CMD Window 1 — Onboard script

```
cd Desktop\auto-wing-
venv\Scripts\activate
python onboard\main.py
```

You should see:
```
Connecting to tcp:127.0.0.1:5760 ...
Heartbeat received from sysid=1
Mode set to GUIDED
Ready. Waiting for GCS target selection.
```

If you see that — great, it connected to the simulator!

### CMD Window 2 — GCS (your screen)

Open a new CMD window and type:
```
cd Desktop\auto-wing-
venv\Scripts\activate
python gcs\gcs_client.py
```

A window opens showing your webcam feed.

---

## Part 6 — Select a Target and Start Tracking

1. Point your webcam at any object (a person, a bottle, your hand)
2. In the GCS window — **left-click directly on the object**
3. A green box appears around it
4. Look at the Mission Planner map — the simulated aircraft will bank toward the target

**Keys in the GCS window:**
| Key | What it does |
|-----|-------------|
| Left click | Pick / change target |
| `S` | Stop tracking |
| `Q` | Quit |

---

## Part 7 — Getting Updates (When I Push New Code)

Whenever I update the code, just open CMD and run:
```
cd Desktop\auto-wing-
git pull
```

That's it — you get the latest version automatically.

---

## Common Errors

| Error message | What to do |
|---------------|-----------|
| `'python' is not recognized` | Python not installed or PATH not set — reinstall Python and check the "Add to PATH" box |
| `'git' is not recognized` | Git not installed — install it from git-scm.com |
| `No module named 'ultralytics'` | You forgot to activate venv — run `venv\Scripts\activate` first |
| `Connection refused tcp:127.0.0.1:5760` | SITL not running — start Mission Planner simulation first |
| `Heartbeat timeout` | SITL crashed — restart Mission Planner simulation |
| GCS window is black | Wrong camera — edit `config\config.yaml`, try `index: 1` or `index: 2` |
| Aircraft not moving in SITL | Make sure Mission Planner shows mode = GUIDED |

---

## Tune the PID (if the aircraft overshoots or is too slow)

Open `config\config.yaml` with Notepad:
```
notepad config\config.yaml
```

Find this section and adjust the numbers:
```yaml
control:
  roll_pid:
    kp: 0.4    # higher = reacts faster (but may shake)
    ki: 0.01   # higher = fixes steady drift
    kd: 0.05   # higher = reduces shaking
  pitch_pid:
    kp: 0.2
    ki: 0.005
    kd: 0.02
  throttle: 0.6   # engine power: 0.0 to 1.0
```

Start with small numbers and increase slowly.

---

## Moving to Real Hardware (Raspberry Pi) — Later

When simulation works and you're ready for real flights:

1. Copy the `auto-wing-` folder to your Raspberry Pi (via USB stick or `scp`)
2. On the Pi, open a terminal and run:
   ```
   pip install -r requirements.txt
   python onboard/main.py
   ```
3. On your laptop, run:
   ```
   python gcs\gcs_client.py
   ```
4. Edit `config\config.yaml`:
   ```yaml
   mavlink:
     connection_string: "/dev/ttyUSB0"   # Pi connected to FC via USB
     baudrate: 115200
   gcs:
     onboard_ip: "192.168.1.xxx"         # your Pi's IP address
   ```

---

## File Structure

```
auto-wing-\
├── config\config.yaml        ← all settings (edit this)
├── onboard\
│   ├── main.py               ← run this on Pi / laptop
│   ├── detector.py           ← YOLO + backup tracker
│   ├── controller.py         ← PID math
│   ├── mavlink_interface.py  ← talks to flight controller
│   └── streamer.py           ← sends video to GCS
├── gcs\
│   └── gcs_client.py         ← run this on your laptop
├── simulation\
│   └── sitl_setup.md         ← extra SITL notes
└── requirements.txt
```

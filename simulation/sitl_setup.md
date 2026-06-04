# SITL Setup Guide for Auto-Wing Tracker

## Windows Setup (ArduPlane SITL)

### Option A: WSL2 (Recommended)
1. Install WSL2: `wsl --install`
2. Inside WSL:
   ```bash
   git clone https://github.com/ArduPilot/ardupilot.git
   cd ardupilot && git submodule update --init --recursive
   Tools/environment_install/install-prereqs-ubuntu.sh -y
   . ~/.profile
   cd ArduPlane && sim_vehicle.py -v ArduPlane --console --map
   ```
3. SITL listens on TCP port 5760 (accessible from Windows as 127.0.0.1:5760)

### Option B: Pre-built Windows Binary
Download from: https://firmware.ardupilot.org/Tools/MissionPlanner/
Run: `ArduPlane-4.x.x-SITL.exe --model plane`

## After SITL starts:
- Connect Mission Planner to TCP 127.0.0.1:5760 to verify
- Run onboard script: `python -m onboard.main`
- Run GCS script: `python gcs/gcs_client.py`

## SITL with virtual camera (for testing without hardware):
Pass a video file as camera:
Edit config/config.yaml: camera.index = "path/to/test_video.mp4"

## Key SITL parameters to set (via Mission Planner or MAVProxy):
```
param set ARMING_CHECK 0      # disable pre-arm checks for SITL
param set TKOFF_ALT 50        # takeoff altitude
param set GUIDED_OPTIONS 0
```

## Testing sequence:
1. Start SITL
2. Start onboard/main.py (connects via TCP to SITL)
3. Start gcs/gcs_client.py
4. In GCS window: click on an object to begin tracking
5. Watch Mission Planner map for aircraft response

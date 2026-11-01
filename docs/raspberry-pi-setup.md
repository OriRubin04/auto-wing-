# Raspberry Pi 5 — install and wiring

Everything the aircraft needs runs on the Pi: camera, saliency proposal,
tracker, PID, RC override. The ground station only sends the target click and
receives video (CLAUDE.md §1). So the Pi has to work with the radio link dead,
and that is what this guide sets up.

Two halves:

- **Part A — software**, one command.
- **Part B — wiring**, Pi ↔ flight controller ↔ power.

> **About a ready-made image:** there is no `.img` to download. Building a
> bootable Pi image is not something that can be produced from this repo, and a
> 2–4 GB file is not something to pass around anyway. `tools/install_pi.sh` is
> the equivalent: one command, same result, and it survives `git pull`.
> Once a card is set up and flying, **clone that card** — that is your image.
> See [Making your own image](#making-your-own-image) at the end.

---

## Part A — software

### A1. Flash the card

Use **Raspberry Pi Imager**.

- OS: **Raspberry Pi OS (64-bit) Lite**, Bookworm or newer.
  Lite, not Desktop — nothing onboard draws a window. `onboard/main.py` never
  calls `imshow`; the GUI lives on your laptop in `gcs/gcs_client.py`.
- Click the gear / **Edit Settings** before writing and set:
  - hostname: `autowing`
  - username / password (this guide assumes user `pi`)
  - **Enable SSH** → password authentication
  - Wi-Fi: your home network *and* your field hotspot if Imager lets you
    (it only takes one; add the second later with `nmcli`).

Boot the Pi, then from your laptop:

```
ssh pi@autowing.local
```

### A2. Run the installer

```bash
curl -fsSL https://raw.githubusercontent.com/OriRubin04/auto-wing-/main/tools/install_pi.sh | bash
```

Prefer not to pipe a script from the network into bash? Clone first and read it:

```bash
git clone https://github.com/OriRubin04/auto-wing-.git ~/auto-wing-
bash ~/auto-wing-/tools/install_pi.sh
```

> **Until this is merged into `main`**, that URL 404s. Use the branch instead:
> ```bash
> git clone -b claude/tender-hamilton-Dz0Se https://github.com/OriRubin04/auto-wing-.git ~/auto-wing-
> bash ~/auto-wing-/tools/install_pi.sh
> ```

It is idempotent — safe to re-run. It:

1. installs apt packages (`python3-venv`, `v4l-utils`, the libs the OpenCV
   wheel links against, the headers `pymavlink` may need to build);
2. clones or updates the repo at `~/auto-wing-`;
3. builds a venv at `~/auto-wing-/venv` and installs `requirements.txt`;
4. adds `usb_max_current_enable=1` to `/boot/firmware/config.txt` — see
   [A6](#a6-pi-5-specific-traps), this one actually matters for the camera;
5. writes a `systemd` unit `autowing.service`, **left disabled**;
6. prints what to do next.

Takes a few minutes, most of it the OpenCV wheel.

> **If pip starts compiling OpenCV, stop it (Ctrl-C).** A source build takes
> hours on a Pi and means PyPI had no wheel for your Python version. Fallback:
> ```bash
> sudo apt install -y python3-opencv
> rm -rf ~/auto-wing-/venv
> python3 -m venv --system-site-packages ~/auto-wing-/venv
> ~/auto-wing-/venv/bin/pip install pymavlink numpy pyyaml
> ```
> The apt build is older OpenCV. CSRT is in it and CSRT is the default
> tracker, so nothing is lost. The optional ViT backend needs ≥ 4.9 and will
> log that it is unavailable and track on CSRT (CLAUDE.md §8) — which is what
> it does on the default config anyway.

### A3. Find the camera

```bash
v4l2-ctl --list-devices
```

A USB camera usually gives two nodes (`/dev/video0`, `/dev/video1`); the
*first* is normally the capture one. The number is the `camera.index` in the
config. Check it:

```bash
cd ~/auto-wing- && source venv/bin/activate
python tools/camera_test.py
```

It prints which indices open and what resolution they actually deliver.
`_open_camera` picks `CAP_V4L2` on Linux by itself (`onboard/main.py:94`), so
there is nothing to set for that.

### A4. Edit the config

```bash
nano ~/auto-wing-/config/config.yaml
```

The four things that are different from your Windows/SITL copy:

```yaml
camera:
  index: 0                 # whatever A3 reported
  width: 1280              # see the note below
  height: 720

mavlink:
  connection_string: "/dev/ttyUSB0"   # Part B decides this
  baudrate: 921600                    # must match SERIAL2_BAUD on the FC

streaming:
  host: "0.0.0.0"          # already correct
```

**On resolution:** 640×480 gives a 4.5 m vehicle 10 px at 250 m, which is
close to the floor where anything works at all (CLAUDE.md §6). 1280×720
doubles that for free. The MJPEG fourcc that 720p needs over USB is already
set (`onboard/main.py:105`) — without it the driver silently drops to ~5 fps
instead of erroring. Watch `loop_ms` in the flight log after the change; if
the Pi cannot hold the rate, go back to 640×480. The link is unaffected either
way: `streaming.max_width` downscales the video for the GCS separately, so
processing runs full-resolution while the radio carries a small frame.

On your laptop, in *its* `config/config.yaml`:

```yaml
gcs:
  onboard_ip: "192.168.1.xxx"   # the Pi's address — `hostname -I` on the Pi
```

### A5. First run, by hand

Not as a service yet — you want to see the log.

```bash
cd ~/auto-wing- && source venv/bin/activate
python onboard/main.py
```

Expected:

```
Camera opened: index=0 resolution=1280x720
Camera backend: 200  delivering 1280x720
Connecting to /dev/ttyUSB0
Heartbeat from sysid=1 compid=1
Ready. Waiting for GCS target selection.
```

No heartbeat → the wiring or the FC parameters are wrong. Part B.

Then on the laptop: `python gcs\gcs_client.py`, and click a target.

Ports, if a firewall is in the way:

| Where | Port | Purpose |
|---|---|---|
| Pi, inbound | 15000/udp | GCS `REGISTER` packet |
| Pi, inbound | 15001/tcp | target select / stop |
| Laptop, inbound | 15100/udp | video (Pi sends to `port + 100`) |

### A6. Pi 5 specific traps

- **USB current limit.** A Pi 5 that does not negotiate USB-PD — which
  includes every Pi powered from a BEC through the GPIO pins or a plain 5 V
  supply — caps *total* USB peripheral current at 600 mA. A USB camera can sit
  right at that edge and the symptom is not an error; it is a camera that
  vanishes mid-flight. `usb_max_current_enable=1` in `/boot/firmware/config.txt`
  lifts the cap, and the installer adds it. Your 5 V supply then has to
  actually deliver — see Part B.
- **Heat.** The Pi 5 throttles at 85 °C and OpenCV will peg all four cores.
  Fit the active cooler or a heatsink with real airflow. Check with
  `vcgencmd measure_temp` and `vcgencmd get_throttled` (`0x0` is clean).
- **No clock.** No RTC battery means flight-log timestamps are wrong until the
  Pi sees a network. Harmless for the data, annoying for naming runs. A CR2032
  on the RTC header fixes it if it bothers you.
- **Undervoltage shows up as weirdness, not as a message.** If anything is
  flaky, check `vcgencmd get_throttled` first.

### A7. Autostart, once it works

```bash
sudo systemctl enable --now autowing
journalctl -u autowing -f        # live log
sudo systemctl disable --now autowing
```

Leave it disabled until you have flown it by hand at least once. The unit is
`Restart=always`, which is what you want in the air and a nuisance on the
bench.

---

## Part B — wiring

### B1. Parts

| Part | Note |
|---|---|
| Raspberry Pi 5, 8 GB | with the active cooler |
| USB camera | the one you have been testing with |
| 5 V BEC, **5 A** | from the main battery, dedicated to the Pi |
| USB↔TTL serial adapter, 3.3 V | CP2102 or FTDI. **Option A below** |
| 6-pin JST-GH cable | for the FC's TELEM2 port |

### B2. Data: Pi ↔ flight controller

Three options. The pin numbers for the TELEM port are the Pixhawk standard —
**check your board's own silkscreen or docs**, pin 1 is the marked one:

```
FC TELEM1 / TELEM2 (6-pin JST-GH)
  1 = +5V     2 = TX     3 = RX     4 = CTS     5 = RTS     6 = GND
```

#### Option A — USB↔TTL adapter (recommended for flight)

```
  Raspberry Pi 5                  USB-TTL adapter              FC TELEM2
  ┌──────────────┐                ┌───────────┐                ┌─────────┐
  │   USB-A      ├────── USB ─────┤           │                │         │
  │              │                │   TXD  ───┼────────────────►│ 3  RX   │
  │              │                │   RXD  ◄──┼────────────────┤ 2  TX   │
  │              │                │   GND  ───┼────────────────┤ 6  GND  │
  │              │                │   VCC   × │  leave unused  │ 1  +5V  │  ×
  └──────────────┘                └───────────┘                └─────────┘
```

- **TX goes to RX, RX goes to TX.** Crossed. Getting this backwards gives you
  exactly the symptom of a dead port: no heartbeat, no error.
- **Leave the adapter's VCC disconnected** and leave TELEM pin 1 alone. The Pi
  has its own supply; tying two 5 V rails together backfeeds one of them.
- GND **must** be connected. It is the shared reference the signals are
  measured against, and without it the link works unreliably or not at all.
- Both sides are 3.3 V logic, so no level shifter. Do not substitute a 5 V
  adapter.
- Device: `/dev/ttyUSB0`. Confirm with `ls -l /dev/ttyUSB*` after plugging in.

Pin it down so it cannot wiggle out — this adapter is now a flight-critical
connector.

#### Option B — the Pi's own GPIO UART

One less part, two more problems.

```
  Pi 40-pin header                    FC TELEM2
  pin 8   GPIO14 / TXD  ─────────────►  3  RX
  pin 10  GPIO15 / RXD  ◄─────────────  2  TX
  pin 6   GND           ──────────────  6  GND
```

Setup:

```bash
sudo raspi-config     # Interface Options > Serial Port
                      #   login shell over serial?  NO
                      #   serial port hardware?     YES
sudo reboot
ls -l /dev/serial0
```

That last line is the point. On the Pi 5 under Bookworm `/dev/serial0` does not
reliably point where it did on a Pi 4 — reports have it aliasing `ttyAMA10`
rather than `ttyAMA0`, and the fix depends on the OS release. Use whatever the
symlink resolves to as `connection_string`, and if the symlink is missing,
Option A instead of fighting it.

#### Option C — plain USB cable, Pi → FC's USB port

Device `/dev/ttyACM0`, baud 115200, and **no FC parameters to change at all** —
ArduPilot's USB serial is SERIAL0 and already speaks MAVLink2. That makes it
the fastest way to get a first heartbeat on the bench.

Not for flight: the FC's USB port is the one you want free for a laptop at the
field, the connector is not retention-locked, and on some boards plugging or
unplugging it resets the autopilot.

### B3. Power

```
  Main battery ──┬── ESC ── motor
                 │
                 ├── FC power module ──► flight controller
                 │
                 └── 5 V / 5 A BEC ────► Pi 5
                                          USB-C,  or
                                          GPIO pin 2 or 4 (+5V) and pin 6 (GND)
```

- **One dedicated BEC for the Pi.** Not the FC's 5 V rail, not the TELEM 5 V
  pin — that pin is typically limited to about 1 A and shared with whatever
  else is on it. A Pi 5 is rated 5 A at 5 V; it will not draw that with one
  camera, but it brownouts on a supply that sags, and a brownout mid-flight
  loses the tracker.
- Powering through the GPIO 5 V pins works and is common, and it bypasses both
  PD negotiation (hence `usb_max_current_enable=1`, A6) and the Pi's inrush
  protection. Clean BEC, short thick leads.
- **Common ground between Pi and FC** — already there via TELEM pin 6. Do not
  omit it on the theory that both are grounded to the battery.
- Put the camera on its own USB port, not behind a hub.

### B4. Flight-controller parameters

For TELEM2 = `SERIAL2`. In Mission Planner: **Config → Full Parameter List**.
Using TELEM1 instead? Then `SERIAL1_*`.

| Parameter | Value | Why |
|---|---|---|
| `SERIAL2_PROTOCOL` | `2` | MAVLink 2 |
| `SERIAL2_BAUD` | `921` | 921600, must match `baudrate` in config.yaml. Drop to `115` if you see dropouts |
| `BRD_SER2_RTSCTS` | `0` | flow control off — only 3 wires are connected |

Then reboot the FC and re-run A5. You should get a heartbeat.

**RC override acceptance.** The autopilot ignores `RC_CHANNELS_OVERRIDE` from a
system ID that is not its configured GCS. `pymavlink` sends as sysid 255, which
is also the default on the FC side, so normally there is nothing to do. If the
heartbeat is fine but the aircraft ignores the commands, that check is the first
suspect: the parameter is `SYSID_MYGCS` up to 4.6 and was renamed
`MAV_GCS_SYSID` in 4.7 (`SYSID_ENFORCE` became a `MAV_OPTIONS` bit at the same
time). ArduPilot's own RC-input page still shows the old names, so trust your
firmware's parameter list over any document — including this one.

**`RC_OVERRIDE_TIME`** is the safety net worth knowing about: if the MAVLink
override stream stops — Pi crash, cable out, software killed — the autopilot
falls back to the real RC receiver within a few seconds on its own.

### B5. Bench test, in this order

**Props off. All of it.**

1. FC on USB to the laptop (Mission Planner) *and* Pi on TELEM2 at the same
   time. Both links can be live together.
2. Transmitter on. **Confirm a mode switch pulls the aircraft back to MANUAL
   or FBWA, and confirm it still does while the Pi is commanding.** This is
   your abort and it is the first thing to test, not the last.
3. Run `onboard/main.py`, click a target in the GCS. In Mission Planner watch:
   - mode changes to **ACRO**;
   - Flight Data → Status: `ch1in`/`ch2in` move with the target;
   - the control surfaces move the right way. Point the camera so the target
     is to the left and check the aircraft wants to roll left.
4. Move the target to the centre and confirm the ailerons settle rather than
   hunting. That is the PID; tune `control.roll_pid` from there.
5. `logs/<timestamp>/flight.csv` — check `loop_ms`, `confidence`,
   `track_score` and `target_w`. `target_w` is also how you will find out
   whether shadows are inflating the size estimate in real footage, which no
   simulation here can tell you.

### B6. Two things to know before the first flight

- **Zero stick does not level the wings in ACRO.** It holds the current bank
  (CLAUDE.md §3). Levelling is done explicitly from the `ATTITUDE` reading.
- **Lost lock currently commands roll = 0**, i.e. hold bank, rather than
  switching to RTL. That is a deliberate deferral, not an oversight, but it
  means a lost target in a turn keeps the turn. Fly with altitude and with a
  finger on the mode switch.

---

## Making your own image

Once a card is set up, configured and flown, that card *is* the image:

- **From the Pi**, to a second card in a USB reader:
  ```bash
  sudo apt install -y rpi-clone      # or: git clone https://github.com/billw2/rpi-clone
  sudo rpi-clone sda
  ```
- **From a laptop**, card in a reader:
  - Linux/macOS: `sudo dd if=/dev/sdX of=autowing.img bs=4M status=progress`
  - Windows: Win32DiskImager → *Read*

Shrink it with [PiShrink](https://github.com/Drewsif/PiShrink) before storing
it; an un-shrunk image is the full card size, mostly zeroes.

Worth doing once things work — restoring a known-good card at the field beats
debugging an install there.

#!/usr/bin/env bash
#
# One-command setup for auto-wing on a Raspberry Pi 5 (Raspberry Pi OS Lite,
# Bookworm or newer).  Idempotent: safe to re-run after a git pull.
#
#   curl -fsSL https://raw.githubusercontent.com/OriRubin04/auto-wing-/main/tools/install_pi.sh | bash
#
# or, having cloned already:
#
#   bash ~/auto-wing-/tools/install_pi.sh
#
# Deliberately does NOT: edit config/config.yaml (that is per-aircraft, and
# clobbering it is how you lose a tuned PID), or enable the service (you want
# to watch the first run by hand).  See docs/raspberry-pi-setup.md.

set -euo pipefail

REPO_URL="https://github.com/OriRubin04/auto-wing-.git"
REPO_DIR="${AUTOWING_DIR:-$HOME/auto-wing-}"
VENV_DIR="$REPO_DIR/venv"
BOOT_CONFIG="/boot/firmware/config.txt"     # pre-Bookworm path handled below
SERVICE="/etc/systemd/system/autowing.service"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[x] %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] && die "Run as the normal user (e.g. pi), not root. It calls sudo where needed."
command -v sudo >/dev/null || die "sudo not found."

# ---------------------------------------------------------------- 0. sanity --
if [ "$(uname -m)" != "aarch64" ]; then
    warn "Architecture is $(uname -m), not aarch64 - this script targets 64-bit"
    warn "Raspberry Pi OS on a Pi 5.  Continuing, but you are off the tested path."
fi
MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
say "Model: $MODEL"

# ----------------------------------------------------------- 1. apt packages --
# python3-venv : Bookworm's Python is externally managed (PEP 668), so a venv
#                is not optional.
# v4l-utils    : v4l2-ctl, to find which /dev/video* is the camera.
# libgl1 +
# libglib2.0-0 : the opencv-contrib-python wheel links against these even
#                though nothing onboard opens a window.
# *-dev, build : pymavlink has no aarch64 wheel for every Python version and
#                falls back to building.
say "Installing apt packages"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    git python3-venv python3-dev python3-pip build-essential \
    v4l-utils libgl1 libglib2.0-0 libxml2-dev libxslt1-dev

# ----------------------------------------------------------------- 2. clone --
if [ -d "$REPO_DIR/.git" ]; then
    say "Updating $REPO_DIR"
    git -C "$REPO_DIR" pull --ff-only || warn "git pull failed (local changes to a tracked file?) - continuing with what is on disk"
else
    say "Cloning into $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
fi
[ -f "$REPO_DIR/requirements.txt" ] || die "$REPO_DIR does not look like the auto-wing repo"

# ------------------------------------------------------------------ 3. venv --
if [ ! -x "$VENV_DIR/bin/python" ]; then
    say "Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    say "venv already present at $VENV_DIR"
fi

say "Installing Python requirements (the OpenCV wheel is the slow part)"
"$VENV_DIR/bin/pip" install --upgrade pip wheel
# --only-binary on opencv: if PyPI has no wheel for this Python, fail fast
# instead of starting a multi-hour source build that looks like a hang.
if ! "$VENV_DIR/bin/pip" install --only-binary=opencv-contrib-python -r "$REPO_DIR/requirements.txt"; then
    warn "No prebuilt OpenCV wheel for this Python ($("$VENV_DIR/bin/python" -V))."
    warn "Falling back to the system OpenCV from apt.  Older build; CSRT"
    warn "(the default tracker) is present, the optional ViT backend is not."
    sudo apt-get install -y python3-opencv
    rm -rf "$VENV_DIR"
    python3 -m venv --system-site-packages "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip wheel
    "$VENV_DIR/bin/pip" install pymavlink numpy pyyaml
fi

say "Checking the imports the aircraft needs"
"$VENV_DIR/bin/python" - <<'PY'
import cv2, numpy, yaml, pymavlink
print(f"  opencv     {cv2.__version__}")
print(f"  numpy      {numpy.__version__}")
print(f"  TrackerCSRT {'yes' if hasattr(cv2, 'TrackerCSRT_create') else 'NO'}"
      f"   TrackerVit {'yes' if hasattr(cv2, 'TrackerVit_create') else 'no (optional)'}")
PY

# ------------------------------------------- 4. boot config (USB + UART) --
# usb_max_current_enable: a Pi 5 that cannot negotiate USB-PD - which is every
#   Pi fed from a BEC - limits TOTAL USB peripheral current to 600 mA.  A USB
#   camera sits close to that line and the failure mode is not an error
#   message, it is a camera that disappears in flight.
# dtparam=uart0=on: brings up the GPIO14/15 UART as /dev/ttyAMA0, which is how
#   the Pi talks to the flight controller.  NOT uart0_console, which would put
#   the Linux console on those pins and fight us for the port.  On a Pi 5 the
#   console already lives on the separate 3-pin debug connector, so there is
#   nothing to free first - unlike every Pi 3/4 guide.
#   Note /dev/serial0 is NOT this port on a Pi 5; it aliases the debug
#   connector.  main.py is pointed at /dev/ttyAMA0 explicitly.
[ -f "$BOOT_CONFIG" ] || BOOT_CONFIG="/boot/config.txt"
if [ -f "$BOOT_CONFIG" ]; then
    # Match the WHOLE setting, not the key before the first '=': splitting
    # "dtparam=uart0=on" on '=' gives "dtparam", which matches an unrelated
    # "dtparam=audio=on" already in the file - and then the UART line is
    # silently skipped and there is no /dev/ttyAMA0 to reach the FC.
    for line in "usb_max_current_enable=1" "dtparam=uart0=on"; do
        if grep -qFx "$line" "$BOOT_CONFIG"; then
            say "$line already set in $BOOT_CONFIG"
        else
            say "Adding $line to $BOOT_CONFIG (needs a reboot)"
            printf '\n# auto-wing\n%s\n' "$line" | sudo tee -a "$BOOT_CONFIG" >/dev/null
            REBOOT_NEEDED=1
        fi
    done
else
    warn "No config.txt found - skipping the USB-current and UART settings."
    warn "Without dtparam=uart0=on there will be no /dev/ttyAMA0 to reach the FC."
fi

# ----------------------------------------------------- 5. serial port access --
if ! id -nG "$USER" | tr ' ' '\n' | grep -qx dialout; then
    say "Adding $USER to the dialout group (serial port access)"
    sudo usermod -aG dialout "$USER"
    warn "Group change takes effect on next login - log out and back in."
fi

# --------------------------------------------------------- 6. systemd unit --
say "Installing autowing.service (left DISABLED)"
sudo tee "$SERVICE" >/dev/null <<UNIT
[Unit]
Description=auto-wing onboard tracking
# Not After=network-online.target on purpose: the whole point is that the
# aircraft does not wait for, or depend on, the link (CLAUDE.md section 1).
After=multi-user.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO_DIR
ExecStart=$VENV_DIR/bin/python $REPO_DIR/onboard/main.py
Restart=always
RestartSec=3
# Journal only; the flight log writes its own files under logs/.
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload

# ------------------------------------------------------------------- done ----
cat <<EOF

$(printf '\033[1;32m')Installed.$(printf '\033[0m')  Now continue in the guide:
$REPO_DIR/docs/raspberry-pi-setup.md

  step 3   wire the Pi to the flight controller
  step 4   set SERIAL2_* on the flight controller
  step 5   edit config/config.yaml on the Pi
  step 6   bench test, props off:
               cd $REPO_DIR && source venv/bin/activate && python onboard/main.py
  step 7   only then, autostart:  sudo systemctl enable --now autowing

EOF
if [ "${REBOOT_NEEDED:-0}" = "1" ]; then
    printf '\033[1;33m[!] Reboot now - config.txt changed.  sudo reboot\033[0m\n\n'
fi

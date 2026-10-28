#!/usr/bin/env python3
"""
Download the TrackerVit ONNX weights into models/.

The weights are not in the repo - they are a 715 KB binary download, not
source, and *.onnx is gitignored.  Until this has been run, the onboard
software falls back to pixel tracking and logs that it did; it does not fail.

    python tools/fetch_tracker_model.py

Verifies the SHA-256 and refuses to install a file that does not match, so a
truncated or substituted download cannot end up flying.
"""

import hashlib
import os
import ssl
import sys
import urllib.request

# opencv_extra's copy, which is the one cv2.TrackerVit is tested against.
URL = ("https://raw.githubusercontent.com/opencv/opencv_extra/4.x/"
       "testdata/dnn/onnx/models/vitTracker.onnx")
SHA256 = "4cbc4639c6967746a0523c22fcd284833cb0d0011ba1d811e00dfe9327ad2b02"
SIZE = 715081

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(REPO_ROOT, 'models', 'vitTracker.onnx')


def main():
    if os.path.exists(DEST):
        with open(DEST, 'rb') as f:
            if hashlib.sha256(f.read()).hexdigest() == SHA256:
                print(f"already present and verified: {DEST}")
                return 0
        print(f"{DEST} exists but does not match the expected hash — replacing")

    os.makedirs(os.path.dirname(DEST), exist_ok=True)

    print(f"downloading {URL}")
    ctx = ssl.create_default_context()          # honours SSL_CERT_FILE
    with urllib.request.urlopen(URL, context=ctx, timeout=60) as r:
        blob = r.read()

    digest = hashlib.sha256(blob).hexdigest()
    if digest != SHA256:
        print(f"HASH MISMATCH — refusing to install.\n"
              f"  expected {SHA256}\n  got      {digest}  ({len(blob)} bytes)",
              file=sys.stderr)
        return 1

    # Write to a temp name and rename, so an interrupted run cannot leave a
    # half-written model that passes the exists() check at startup.
    tmp = DEST + '.part'
    with open(tmp, 'wb') as f:
        f.write(blob)
    os.replace(tmp, DEST)

    print(f"ok — {len(blob)} bytes (expected {SIZE}) -> {DEST}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

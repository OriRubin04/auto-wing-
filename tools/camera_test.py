"""
Run this to find which camera index works on your laptop.
Usage: python tools\camera_test.py
Press Q to close each window.
"""
import cv2

print("Testing cameras 0-4 with DirectShow backend...\n")

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"  index {i}: NOT found")
        continue

    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"  index {i}: found but CANNOT grab frame (probably in use by another app)")
        cap.release()
        continue

    h, w = frame.shape[:2]
    print(f"  index {i}: OK  —  {w}x{h}  ← use this one")
    cv2.imshow(f"Camera {i} — press Q to close", frame)
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {i} — press Q to close", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

print("\nDone. Edit config\\config.yaml and set camera.index to the working number.")

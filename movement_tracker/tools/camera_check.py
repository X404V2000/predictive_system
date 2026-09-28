#!/usr/bin/env python3
"""
Webcam Compatibility Check for the Predictive Movement System.

Verifies the local camera can be opened, negotiated at useful resolutions and
frame rates, and actually delivers frames. Run this before wiring live pose
input into the pipeline.

Usage:
    python3 camera_check.py                 # probe device 0
    python3 camera_check.py --device 1      # probe a different device
    python3 camera_check.py --preview       # show a live window (needs a display)

Requires OpenCV. On this machine the system interpreter already provides it:
    python3 -c "import cv2; print(cv2.__version__)"
"""

import argparse
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check webcam compatibility.")
    parser.add_argument("--device", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--preview", action="store_true", help="show a live preview window")
    parser.add_argument("--frames", type=int, default=60, help="frames to sample for FPS")
    parser.add_argument(
        "--width", type=int, default=640, help="capture width (default 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="capture height (default 480)"
    )
    parser.add_argument(
        "--fourcc",
        default="MJPG",
        help="pixel format: MJPG (default), YUYV, or auto",
    )
    parser.add_argument(
        "--manual-exposure",
        type=int,
        default=None,
        metavar="N",
        help=(
            "disable auto-exposure and force this exposure value. Low light makes "
            "UVC cameras drop frame rate to lengthen exposure; forcing a short "
            "exposure often restores full FPS."
        ),
    )
    return parser.parse_args()


def check_opencv() -> bool:
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[FAIL] OpenCV (cv2) is not importable by this interpreter.")
        print("       Try the system interpreter:  python3 camera_check.py")
        print("       Or install:                  pip install opencv-python")
        return False
    return True


def _configure(cap, args) -> None:
    """Applies pixel format, resolution and optional manual exposure."""
    import cv2

    if args.fourcc.lower() != "auto":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc.upper()))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.manual_exposure is not None:
        # 1 = manual mode on V4L2 (varies by device); 3 = aperture-priority auto.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, args.manual_exposure)


def probe(args) -> bool:
    """Opens the camera, reads frames, and reports measured throughput."""
    import cv2

    device = args.device
    width, height, frames = args.width, args.height, args.frames

    print(f"[..] Opening /dev/video{device} ...")
    # V4L2 backend is the correct one on Linux; MJPG avoids USB bandwidth limits.
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"[FAIL] Could not open /dev/video{device}.")
        print("       - Confirm the device exists:  ls -l /dev/video*")
        print("       - Confirm you are in the video group:  groups | grep video")
        print("       - If the camera is busy, close other apps using it.")
        return False

    _configure(cap, args)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    backend = cap.getBackendName()
    print(f"[ OK] Opened. Backend={backend}")
    print(f"[ OK] Negotiated resolution: {actual_w}x{actual_h}")
    print(f"[ OK] Reported FPS: {reported_fps:.1f}")

    if actual_w != width or actual_h != height:
        print(f"[WARN] Requested {width}x{height} but got {actual_w}x{actual_h}.")
        print("       The device does not support the requested mode.")

    # Warm up: the first few frames from UVC cameras are often blank.
    for _ in range(5):
        cap.read()

    # Measure real throughput over `frames` reads.
    start = time.perf_counter()
    good = 0
    last_frame = None
    for _ in range(frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            good += 1
            last_frame = frame
    elapsed = time.perf_counter() - start
    measured_fps = good / elapsed if elapsed > 0 else 0.0

    print(f"[{' OK' if good == frames else 'WARN'}] Frames captured: {good}/{frames}")
    print(f"[ OK] Measured throughput: {measured_fps:.1f} FPS")

    if last_frame is not None:
        mean_brightness = float(last_frame.mean())
        print(f"[ OK] Frame shape: {last_frame.shape}, mean brightness: {mean_brightness:.1f}")
        if mean_brightness < 1.0:
            print("[WARN] Frame appears black — check the lens cover / privacy switch.")
    else:
        print("[FAIL] No usable frames were returned.")
        cap.release()
        return False

    overall = good > frames * 0.8 and measured_fps > 10.0
    print(
        f"\n[{'PASS' if overall else 'WARN'}] Camera is "
        f"{'suitable' if overall else 'marginal'} for real-time pose input."
    )

    if measured_fps < 30:
        print("       Note: pose estimation typically wants >= 30 FPS for smooth tracking.")
        print("       If the reported FPS above was higher than measured, the camera is")
        print("       likely lengthening exposure in dim light. Fixes, in order:")
        print("         1. Add more light on the subject.")
        print("         2. Force a short exposure:  --manual-exposure 100")
        print("         3. Lower resolution:        --width 320 --height 240")

    cap.release()
    return overall


def preview(args) -> None:
    """Shows a live window. Press 'q' to exit."""
    import cv2

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    _configure(cap, args)
    print("[..] Live preview — press 'q' in the window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame grab failed; stopping preview.")
            break
        cv2.imshow("Camera check - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    if not check_opencv():
        return 1

    if args.preview:
        preview(args)
        return 0

    return 0 if probe(args) else 1


if __name__ == "__main__":
    sys.exit(main())

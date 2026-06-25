"""Paint the first-frame object mask for FoundationPose run_demo."""

import argparse
import glob
import os

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paint the first-frame mask (object=white, background=black) for FP run_demo."
    )
    default_scene = os.path.expanduser("~/Documents/FoundationPose/demo_data/peg_test")
    parser.add_argument(
        "--scene_dir",
        default=default_scene,
        help=f"Scene dir with rgb/. Default: {default_scene}",
    )
    parser.add_argument(
        "--brush",
        type=int,
        default=12,
        help="Brush radius in pixels.",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Continue from the saved mask file if it exists. Default: start blank.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scene_dir = os.path.abspath(args.scene_dir)
    rgb_files = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.png")))
    if not rgb_files:
        raise FileNotFoundError(f"No RGB frames found under {scene_dir}/rgb")

    rgb_path = rgb_files[0]
    rgb_name = os.path.basename(rgb_path)
    mask_dir = os.path.join(scene_dir, "masks")
    mask_path = os.path.join(mask_dir, rgb_name)

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"Failed to read {rgb_path}")

    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    if args.load and os.path.exists(mask_path):
        existing = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if existing is not None and existing.shape == mask.shape:
            mask = existing.copy()
            print(f"Loaded existing mask from {mask_path}")
    elif os.path.exists(mask_path):
        print(f"Note: {mask_path} exists but not loaded (use --load to continue editing).")

    state = {"drawing": False, "erase": False, "brush": args.brush, "stroke_active": False}
    max_undo = 50
    history = [mask.copy()]

    def checkpoint():
        history.append(mask.copy())
        if len(history) > max_undo:
            history.pop(0)

    def undo():
        if len(history) <= 1:
            print("Nothing to undo")
            return
        history.pop()
        mask[:] = history[-1]
        print(f"Undo ({len(history) - 1} step(s) left)")

    def on_mouse(event, x, y, _flags, _param):
        color = 0 if state["erase"] else 255
        radius = state["brush"]
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["erase"] = False
            state["stroke_active"] = True
            cv2.circle(mask, (x, y), radius, color, -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["drawing"] = True
            state["erase"] = True
            state["stroke_active"] = True
            cv2.circle(mask, (x, y), radius, 0, -1)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            cv2.circle(mask, (x, y), radius, color, -1)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            if state["stroke_active"]:
                checkpoint()
                state["stroke_active"] = False
            state["drawing"] = False

    window = "Paint mask"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    print(f"RGB frame : {rgb_path}")
    print(f"Mask save : {mask_path}")
    print(
        "Controls  : left=paint, right=erase, [/]=brush, z=undo, s=save, r=reset, q=quit"
    )
    print(f"Brush     : {state['brush']} px radius (use --brush N to set initial size)")

    while True:
        preview = rgb.copy()
        preview[mask > 0] = (0.4 * preview[mask > 0] + 0.6 * np.array([0, 255, 0])).astype(
            np.uint8
        )
        hint = f"brush:{state['brush']}px  [/] size  z undo  s save  r reset  q quit"
        cv2.putText(
            preview,
            hint,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            hint,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("]") or key == ord("="):
            state["brush"] = min(state["brush"] + 2, 80)
            print(f"Brush size: {state['brush']} px")
        elif key == ord("["):
            state["brush"] = max(state["brush"] - 2, 1)
            print(f"Brush size: {state['brush']} px")
        elif key == ord("z"):
            undo()
        elif key == ord("s"):
            os.makedirs(mask_dir, exist_ok=True)
            cv2.imwrite(mask_path, mask)
            print(f"Saved mask to {mask_path}")
        elif key == ord("r"):
            mask[:] = 0
            checkpoint()
            print("Mask reset")
        elif key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

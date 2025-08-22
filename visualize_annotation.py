"""
visualize_one.py

• Copies the original frame into a new “viz” folder
• Saves an extra *_overlay.png with key-points drawn (supports 1- or 2-tool frames)

Examples
--------
python visualize_annotation.py --ann_file /home/student/Project/synthetic_data/ann/000073.json
 
"""
import argparse
import json
import cv2
import pathlib
import shutil
import sys

# ---------- drawing style ----------
DOT_RADIUS = 5
DOT_THICK  = -1             # filled circle
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
TEXT_THICK = 1
COLORS = [               
    (255,   0,   0),        # blue
    (  0, 255,   0),        # green
    (  0,   0, 255),        # red
    (255, 255,   0),        # cyan
    (255,   0, 255),        # magenta
    (  0, 255, 255),        # yellow
]
# -----------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ann_file", required=True, type=pathlib.Path,
                   help="Path to one annotation JSON (e.g. synthetic_dataset/ann/000123.json)")
    p.add_argument("--no_show", action="store_true",
                   help="Skip the pop-up window (still saves images)")
    return p.parse_args()

def main():
    args = parse_args()

    with open(args.ann_file, "r") as f:
        meta = json.load(f)

    dataset_dir = args.ann_file.parent.parent     
    img_path    = dataset_dir / meta["image"]   

    viz_dir = dataset_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    original_copy = viz_dir / img_path.name
    if not original_copy.exists():
        shutil.copy2(img_path, original_copy)    

    # ---- draw overlay ----
    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit(f"Could not open image: {img_path}")

    # Iterate over each instrument object
    for obj_idx, obj in enumerate(meta["objects"]):
        color = COLORS[obj_idx % len(COLORS)]
        instr_name = obj["instrument"]

        for kp_name, xy in obj["keypoints"].items():
            x, y = map(int, xy)
            cv2.circle(img, (x, y), DOT_RADIUS, color, DOT_THICK)
            label = f"{instr_name}:{kp_name}"
            cv2.putText(img, label, (x + 6, y - 6),
                        FONT, FONT_SCALE, color, TEXT_THICK, cv2.LINE_AA)

    overlay_path = viz_dir / f"{img_path.stem}_overlay.png"
    cv2.imwrite(str(overlay_path), img)

    # ---- optional preview ----
    if not args.no_show:
        cv2.imshow("annotation", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"[OK] original saved to:  {original_copy}")
    print(f"[OK] overlay  saved to:  {overlay_path}")

if __name__ == "__main__":
    main()

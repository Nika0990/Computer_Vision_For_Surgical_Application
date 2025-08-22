import argparse, cv2, torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path

# --- Keypoint schema ---
KPT_NAMES = ["left_tip", "right_tip", "hinge_center", "left_ring", "right_ring"]
L = {n: i for i, n in enumerate(KPT_NAMES)}

SKELETON = [
    (L["left_tip"],     L["hinge_center"]),
    (L["right_tip"],    L["hinge_center"]),
    (L["hinge_center"], L["left_ring"]),
    (L["hinge_center"], L["right_ring"]),
]

KPT_COLOR = {
    "left_tip":     (0,   0, 255),  # red
    "right_tip":    (0, 255, 255),  # yellow
    "hinge_center": (0, 255,   0),  # green
    "left_ring":    (255, 0,   0),  # blue
    "right_ring":   (255, 0, 255),  # magenta
}
EDGE_COLOR = (0, 0, 0)       # black lines
BOX_COLOR  = (36, 255, 12)   # light green
TEXT_COLOR = (0, 0, 0)

def draw_boxes(img, boxes_xyxy, confs=None, thickness=2):
    if boxes_xyxy is None:
        return img
    h, w = img.shape[:2]
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [int(v) for v in b]
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))

        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, thickness, cv2.LINE_AA)

        if confs is not None:
            score = float(confs[i])
            label = f"{score:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), BOX_COLOR, -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    return img

def draw_pose(img, xy, conf=None, kpt_conf_thres=0.2, radius=5, thickness=2, draw_labels=False):
    """
    xy: (K, 2) float32 in pixels
    conf: (K,) or None — per-kpt confidence if available
    """
    K = xy.shape[0]

    def ok(i):
        if not (0 <= i < K): return False
        if conf is None: return True
        return float(conf[i]) >= kpt_conf_thres

    # lines first
    for u, v in SKELETON:
        if ok(u) and ok(v):
            p1 = tuple(np.int32(xy[u]))
            p2 = tuple(np.int32(xy[v]))
            if any(np.isnan(np.array(p1 + p2))):  
                continue
            cv2.line(img, p1, p2, EDGE_COLOR, thickness, cv2.LINE_AA)

    # points
    for i, name in enumerate(KPT_NAMES):
        if ok(i):
            p = tuple(np.int32(xy[i]))
            if any(np.isnan(np.array(p))):
                continue
            cv2.circle(img, p, radius, KPT_COLOR[name], -1, cv2.LINE_AA)
            if draw_labels:
                cv2.putText(img, name, (p[0] + 6, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img

def main():
    ap = argparse.ArgumentParser("YOLOv8-Pose image annotator (custom skeleton & colors)")
    ap.add_argument("--weights", "-w", required=True, help="Path to trained pose .pt (e.g., best.pt)")
    ap.add_argument("--source", "-s", required=True, help="Input image path")
    ap.add_argument("--output", "-o", default="annotated.png", help="Output image path")
    ap.add_argument("--conf", "-c", type=float, default=0.4, help="Box confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inference size (multiple of 32)")
    ap.add_argument("--device", default="0", help="'cpu' or CUDA id like '0'")
    ap.add_argument("--kpt_conf", type=float, default=0.20, help="Min keypoint confidence to draw")
    ap.add_argument("--max_det", type=int, default=20, help="Max detections")
    ap.add_argument("--labels", action="store_true", help="Draw keypoint names")
    ap.add_argument("--no_boxes", action="store_true", help="Do not draw bounding boxes")
    ap.add_argument("--thickness", type=int, default=2, help="Skeleton/box thickness")
    ap.add_argument("--radius", type=int, default=5, help="Keypoint circle radius")
    args = ap.parse_args()

    model = YOLO(args.weights)

    img = cv2.imread(args.source, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.source}")
    annotated = img.copy()

    with torch.no_grad():
        res = model.predict(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            max_det=args.max_det
        )[0]

    if not args.no_boxes and res.boxes is not None and len(res.boxes) > 0:
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") and res.boxes.conf is not None else None
        draw_boxes(annotated, boxes_xyxy, confs=confs, thickness=args.thickness)

    if res.keypoints is not None and len(res.keypoints) > 0:
        k_xy = res.keypoints.xy.cpu().numpy()
        k_cf = res.keypoints.conf.cpu().numpy() if hasattr(res.keypoints, "conf") and res.keypoints.conf is not None else None
        for i in range(k_xy.shape[0]):
            xy_i = k_xy[i]
            cf_i = None if k_cf is None else k_cf[i]
            draw_pose(
                annotated, xy_i, conf=cf_i,
                kpt_conf_thres=args.kpt_conf,
                radius=args.radius,
                thickness=args.thickness,
                draw_labels=args.labels
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), annotated)
    if not ok:
        raise RuntimeError(f"Failed to write: {out_path}")

    print(f"Saved annotated image to {out_path}")

if __name__ == "__main__":
    main()

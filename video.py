import argparse, cv2, torch, time
from ultralytics import YOLO
import numpy as np

KPT_NAMES = ["left_tip", "right_tip", "hinge_center", "left_ring", "right_ring"]
L = {n:i for i,n in enumerate(KPT_NAMES)}

SKELETON = [
    (L["left_tip"],   L["hinge_center"]),
    (L["right_tip"],  L["hinge_center"]),
    (L["hinge_center"], L["left_ring"]),
    (L["hinge_center"], L["right_ring"]),
]

KPT_COLOR = {
    "left_tip":     (0,   0, 255),   # red
    "right_tip":    (0, 255, 255),   # yellow
    "hinge_center": (0, 255,   0),   # green
    "left_ring":    (255, 0,   0),   # blue
    "right_ring":   (255, 0, 255),   # magenta
}
EDGE_COLOR = (0, 0, 0)      # black lines
BOX_COLOR  = (36, 255, 12)  # light green
TEXT_COLOR = (0, 0, 0)

def draw_boxes(frame, boxes_xyxy, confs=None, cls_ids=None, thickness=2):
    """
    boxes_xyxy: (N,4) array-like [x1,y1,x2,y2]
    confs: (N,) or None
    cls_ids: (N,) or None  (unused here, class-agnostic)
    """
    if boxes_xyxy is None:
        return frame

    h, w = frame.shape[:2]
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [int(v) for v in b]
        x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, thickness, cv2.LINE_AA)

        if confs is not None:
            score = float(confs[i])
            label = f"{score:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), BOX_COLOR, -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    return frame


def draw_pose(frame, xy, conf=None, kpt_conf_thres=0.2,
              radius=4, thickness=2, text=False):
    """
    xy: (K,2) float ndarray in pixels
    conf: (K,) or None
    """
    K = xy.shape[0]
    h, w = frame.shape[:2]

    def ok(i):
        if not (0 <= i < K): return False
        if conf is None: return True
        return float(conf[i]) >= kpt_conf_thres

    for u, v in SKELETON:
        if ok(u) and ok(v):
            p1 = tuple(np.int32(xy[u]))
            p2 = tuple(np.int32(xy[v]))
            cv2.line(frame, p1, p2, EDGE_COLOR, thickness, cv2.LINE_AA)

    for i, name in enumerate(KPT_NAMES):
        if ok(i):
            p = tuple(np.int32(xy[i]))
            cv2.circle(frame, p, radius, KPT_COLOR[name], -1, cv2.LINE_AA)
            if text:
                cv2.putText(frame, name, (p[0]+5, p[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)

    return frame


def main():
    ap = argparse.ArgumentParser("YOLOv11-Pose video annotator (class-agnostic)")
    ap.add_argument("--weights", "-w", required=True, help="Path to trained pose .pt (e.g., best.pt)")
    ap.add_argument("--source", "-s", required=True, help="Input video path or webcam index")
    ap.add_argument("--output", "-o", default="annotated.mp4", help="Output video path")
    ap.add_argument("--conf", "-c", type=float, default=0.4, help="Box confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inference size (multiple of 32)")
    ap.add_argument("--device", default="0", help="'cpu' or CUDA id like '0'")
    ap.add_argument("--stride_frames", type=int, default=1, help="Process every Nth frame")
    ap.add_argument("--kpt_conf", type=float, default=0.20, help="Min keypoint confidence to draw")
    ap.add_argument("--max_det", type=int, default=20, help="Max detections per frame")
    ap.add_argument("--show", action="store_true", help="Preview window (ESC to quit)")
    ap.add_argument("--labels", action="store_true", help="Draw keypoint names")
    args = ap.parse_args()

    model = YOLO(args.weights)

    src = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Cannot open writer: {args.output}")

    print(f"{args.source} → {args.output} | imgsz={args.imgsz} conf={args.conf} iou={args.iou} device={args.device}")
    frame_id = 0
    t0 = time.time()

    try:
        with torch.no_grad():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if args.stride_frames > 1 and (frame_id % args.stride_frames):
                    out.write(frame)
                    frame_id += 1
                    continue

                res = model.predict(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    verbose=False,
                    max_det=args.max_det
                )[0]

                annotated = frame.copy()

                # --- draw bounding boxes ---
                if res.boxes is not None and len(res.boxes) > 0:
                    boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") and res.boxes.conf is not None else None
                    cls_ids = res.boxes.cls.cpu().numpy() if hasattr(res.boxes, "cls") and res.boxes.cls is not None else None
                    draw_boxes(annotated, boxes_xyxy, confs, cls_ids, thickness=2)

                # --- draw keypoints & skeleton ---
                if res.keypoints is not None and len(res.keypoints) > 0:
                    k_xy = res.keypoints.xy.cpu().numpy()
                    k_cf = res.keypoints.conf.cpu().numpy() if hasattr(res.keypoints, "conf") and res.keypoints.conf is not None else None
                    for i in range(k_xy.shape[0]):
                        xy_i = k_xy[i]
                        cf_i = None if k_cf is None else k_cf[i]
                        draw_pose(annotated, xy_i, cf_i, kpt_conf_thres=args.kpt_conf,
                                  radius=5, thickness=2, text=args.labels)

                out.write(annotated)

                # progress
                n = int(getattr(res.boxes, "shape", [0])[0]) if res.boxes is not None else 0
                print(f"Frame {frame_id+1}/{total_frames or '?'} | detections: {n}")
                if total_frames:
                    dt = time.time() - t0
                    eta = (dt / (frame_id+1)) * (total_frames - frame_id - 1)
                    print(f"ETA: {eta/60:.1f} min")

                if args.show:
                    cv2.imshow("YOLOv11-Pose (class-agnostic)", annotated)
                    if cv2.waitKey(1) & 0xFF == 27:  
                        break

                frame_id += 1
    finally:
        cap.release()
        out.release()
        if args.show:
            cv2.destroyAllWindows()
    print("Done:", args.output)


if __name__ == "__main__":
    main()

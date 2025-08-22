"""
video_to_yolo_pose.py
Run YOLOv11-pose on a video and export frames + YOLO keypoint labels.

YOLO keypoint label format per line:
  <cls> <cx> <cy> <w> <h>  <kpt1_x> <kpt1_y> <kpt1_v> ... <kptK_x> <kptK_y> <kptK_v>
All coordinates are normalized to [0,1]. Visibility 'v' is:
  2 = visible (we set 2 if kpt confidence >= --kpt_conf)
  1 = labeled but not visible (we set 1 if below threshold but predicted)
  0 = not labeled (we set 0 only if the model did not return the keypoint)

Example usage:
  python video_to_yolo_pose.py \
    --weights runs/pose/train_or_aug_ca_handgrip/weights/best.pt \
    --source /datashare/project/vids_tune/20_2_24_1.mp4 \
    --out_dir exported_yolo \
    --imgsz 1280 --conf 0.1 --iou 0.5 --device 0 \
    --kpt_conf 0.2 --stride_frames 1 --save_empty
"""
import argparse, os, cv2, numpy as np, torch
from ultralytics import YOLO
from pathlib import Path

def xyxy_to_cxcywh_norm(box_xyxy, w, h):
    x1, y1, x2, y2 = box_xyxy
    bw = max(0.0, float(x2 - x1))
    bh = max(0.0, float(y2 - y1))
    cx = float(x1 + x2) / 2.0
    cy = float(y1 + y2) / 2.0
    # normalize
    return cx / w, cy / h, bw / w, bh / h

def clip01(a):
    return float(np.clip(a, 0.0, 1.0))

def main():
    ap = argparse.ArgumentParser("Export YOLO keypoint labels from video")
    ap.add_argument("--weights", "-w", required=True, help="Path to YOLO pose .pt (e.g., best.pt)")
    ap.add_argument("--source", "-s", required=True, help="Input video path or webcam index")
    ap.add_argument("--out_dir", "-o", required=True, help="Output root dir for images/ and labels/")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")  
    ap.add_argument("--max_det", type=int, default=50)
    ap.add_argument("--stride_frames", type=int, default=1, help="Export every Nth frame")
    ap.add_argument("--kpt_conf", type=float, default=0.20, help="Min keypoint confidence → v=2 else v=1")
    ap.add_argument("--save_empty", action="store_true", help="Also save frames with no detections (empty .txt)")
    args = ap.parse_args()

    model = YOLO(args.weights)

    src = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")

    out_root = Path(args.out_dir)
    img_dir = out_root / "images"
    lbl_dir = out_root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    frame_id = 0
    saved = 0

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.stride_frames > 1 and (frame_id % args.stride_frames):
                frame_id += 1
                continue

            H, W = frame.shape[:2]
            res = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                max_det=args.max_det,
                verbose=False
            )[0]

            # Build label lines
            lines = []
            n_kpts = 0
            if res.keypoints is not None and len(res.keypoints) > 0:
                k_xy = res.keypoints.xy.cpu().numpy()               
                k_cf = (res.keypoints.conf.cpu().numpy()
                        if hasattr(res.keypoints, "conf") and res.keypoints.conf is not None
                        else None)                                   
                n_kpts = k_xy.shape[1]

            if res.boxes is not None and len(res.boxes) > 0:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()            
                confs = (res.boxes.conf.cpu().numpy()
                         if hasattr(res.boxes, "conf") and res.boxes.conf is not None else None)
                cls_ids = (res.boxes.cls.cpu().numpy().astype(int)
                           if hasattr(res.boxes, "cls") and res.boxes.cls is not None else np.zeros((len(boxes_xyxy),), dtype=int))

                for i, b in enumerate(boxes_xyxy):
                    cx, cy, ww, hh = xyxy_to_cxcywh_norm(b, W, H)
                    cls_i = int(cls_ids[i]) if cls_ids is not None else 0

                    row = [str(cls_i),
                           f"{clip01(cx):.6f}", f"{clip01(cy):.6f}",
                           f"{clip01(ww):.6f}", f"{clip01(hh):.6f}"]

                    if res.keypoints is not None and len(res.keypoints) > 0:
                        kxy = k_xy[i] if i < k_xy.shape[0] else None
                        kcf = k_cf[i] if (k_cf is not None and i < k_cf.shape[0]) else None
                        if kxy is not None:
                            for k in range(kxy.shape[0]):
                                x, y = float(kxy[k, 0]), float(kxy[k, 1])
                                x_n, y_n = x / W, y / H
                                # visibility flag from confidence
                                v = 2
                                if kcf is not None:
                                    v = 2 if float(kcf[k]) >= args.kpt_conf else 1
                                row += [f"{clip01(x_n):.6f}", f"{clip01(y_n):.6f}", str(int(v))]
                    lines.append(" ".join(row))

            if lines:
                base = f"{saved:06d}"
                img_path = img_dir / f"{base}.jpg"
                lbl_path = lbl_dir / f"{base}.txt"

                cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                with open(lbl_path, "w", encoding="utf-8") as f:
                    if lines:
                        f.write("\n".join(lines))
                saved += 1

            print(f"Frame {frame_id:06d} → labels: {len(lines)}")
            frame_id += 1

    cap.release()
    print(f"Saved {saved} frames to: {str(out_root)}")

if __name__ == "__main__":
    main()

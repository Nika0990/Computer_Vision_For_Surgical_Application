"""
Train YOLOv11-pose directly from an existing YOLO dataset.

Expected layout (example):
yolo_mixed/
  ├─ train/
  │   ├─ images/*.jpg|png
  │   └─ labels/*.txt
  ├─ val/
  │   ├─ images/*.jpg|png
  │   └─ labels/*.txt
  ├─ data.yaml
  └─ manifest.csv   (optional; ignored)

If data.yaml is missing pose keys, they'll be patched in-place:
  - names: ["instrument"]
  - nc: 1
  - kpt_shape: [5, 3]
  - flip_idx: [1, 0, 2, 4, 3]
"""
import argparse, os
from pathlib import Path
from ultralytics import YOLO

NAMES = ["instrument"]
KPT_SHAPE = [5, 3]
FLIP_IDX  = [1, 0, 2, 4, 3]

def patch_data_yaml(data_yaml: Path):
    import yaml
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f) or {}

    root = data_yaml.parent

    def _fix_path(key, default_rel):
        v = cfg.get(key, default_rel)
        p = Path(v)
        if not p.is_absolute():
            p = root / v
        cfg[key] = str(p.relative_to(root)) if str(p).startswith(str(root)) else str(p)

    _fix_path("train", "train/images")
    _fix_path("val",   "val/images")

    # Pose-specific fields 
    if "names" not in cfg or not cfg["names"]:
        cfg["names"] = NAMES
    if "nc" not in cfg:
        cfg["nc"] = len(cfg["names"])
    if "kpt_shape" not in cfg:
        cfg["kpt_shape"] = KPT_SHAPE
    if "flip_idx" not in cfg:
        cfg["flip_idx"] = FLIP_IDX

    cfg["path"] = str(root)

    with open(data_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return data_yaml

def main():
    ap = argparse.ArgumentParser("Train YOLOv11-pose from an existing YOLO dataset")
    ap.add_argument("--data", required=True,
                    help="Path to data.yaml OR to the dataset folder containing data.yaml")
    ap.add_argument("--model", default="yolo11l-pose.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--run_project", default="runs/pose")
    ap.add_argument("--run_name", default="train_from_yolo_mixed")
    ap.add_argument("--resume", action="store_true", help="Resume last checkpoint in this run folder")
    args = ap.parse_args()

    # Resolve data.yaml
    data_path = Path(args.data)
    if data_path.is_dir():
        data_yaml = data_path / "data.yaml"
    else:
        data_yaml = data_path
    assert data_yaml.exists(), f"data.yaml not found at {data_yaml}"

    data_yaml = patch_data_yaml(data_yaml)

    if args.workers is None:
        args.workers = max(2, min(os.cpu_count() or 6, 8))

    # Train
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
        fliplr=0.0, flipud=0.0, perspective=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        lr0=5e-3, lrf=0.1, momentum=0.937, weight_decay=5e-4,
        cos_lr=True, close_mosaic=10, amp=True, pretrained=True,
        cache="ram", patience=20, plots=True, save_period=-1,
        freeze=0,
        project=args.run_project, name=args.run_name, exist_ok=True, resume=args.resume
    )

    # Validate on val split with plots
    model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.run_project, name=args.run_name, exist_ok=True, plots=True
    )

if __name__ == "__main__":
    main()

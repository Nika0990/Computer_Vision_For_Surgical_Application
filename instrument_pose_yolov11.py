"""
YOLOv11-Pose: PREP (OR-lighting + bands + gloves + hand-grip + video) + TRAIN

- Converts dataset (images/ + ann/) to YOLOv11-pose format (single class: "instrument")
- Augments TRAIN images with label-safe occlusions:
    • band occlusions (shaft-aligned, often BETWEEN keypoints, with lateral offset)
    • glove/skin patches (often BETWEEN keypoints, with perpendicular jitter)
    • hand-grip occluder (palm + fingers wrapping the shaft BETWEEN hinge and rings/tips)
    • OR spotlight ("lightning") and video-like compression/noise
  (All augmentations avoid geometric warps, so keypoint labels remain valid.)

Run examples
------------
Prepare:
  python instrument_pose_yolov11.py --step prepare --root ./synthetic_dataset --out_dir /prepared_data_pose \
    --aug_copies 1 --tweezer_boost 0 \
    --band_prob 0.35 --glove_prob 0.55 --grip_prob 0.45 \
    --band_per_image 0.55 --glove_per_image 0.55 --grip_per_image 0.55 \
    --or_prob 0.50 --video_prob 0.35

Train:
  python instrument_pose_yolov11.py --step train --out_dir /prepared_data_pose \
    --model yolo11l-pose.pt --epochs 100 --imgsz 1024 --batch 16 --device 0
"""
import os, json, hashlib, argparse, shutil, time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ----- schema & constants (CLASS-AGNOSTIC) -----
KEYPOINTS = ["left_tip", "right_tip", "hinge_center", "left_ring", "right_ring"]
NUM_KPTS  = len(KEYPOINTS)

# Map both tools to the SAME class id = 0 (class-agnostic)
CLS2ID = {"needle_holder": 0, "tweezers": 0}
NAMES  = ["instrument"]  # single class

# horizontal flip swaps L/R consistently
FLIP_IDX = [1, 0, 2, 4, 3]

_rng = np.random.default_rng(2025)


# =========================
# Utility helpers
# =========================
def read_json(p: Path) -> Dict:
    with open(p, "r") as f:
        return json.load(f)

def gather_pairs(root: Path) -> List[Tuple[Path, Path]]:
    img_dir, ann_dir = root/"images", root/"ann"
    pairs = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in (".png",".jpg",".jpeg"):
            j = ann_dir/(p.stem + ".json")
            if j.exists():
                pairs.append((p, j))
    return pairs

def fingerprint(pairs: List[Tuple[Path,Path]], params: Dict) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(params, sort_keys=True).encode())
    for img_p, ann_p in pairs:
        for q in (img_p, ann_p):
            st = q.stat()
            h.update(str(q).encode()); h.update(str(st.st_size).encode()); h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()

def load_meta(meta_p: Path) -> Dict:
    if meta_p.exists():
        try:
            return json.loads(meta_p.read_text())
        except Exception:
            return {}
    return {}

def save_meta(meta_p: Path, meta: Dict):
    meta_p.parent.mkdir(parents=True, exist_ok=True)
    meta_p.write_text(json.dumps(meta, indent=2))

def write_txt(p: Path, lines: List[str]):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + ("\n" if lines else ""))

def split_pairs(pairs: List[Tuple[Path,Path]], val_pct: float, seed: int):
    import random
    rnd = random.Random(seed); rnd.shuffle(pairs)
    n_val = max(1, int(len(pairs)*val_pct))
    return pairs[n_val:], pairs[:n_val]


# =========================
# Label building
# =========================
def build_kvec(kdict: Dict[str, List[float]], w: int, h: int) -> List[float]:
    """Return [x/w, y/h, v]*NUM_KPTS in fixed KEYPOINTS order; v=2 if present else 0."""
    v = []
    for name in KEYPOINTS:
        if name in kdict:
            x,y = float(kdict[name][0]), float(kdict[name][1])
            x = max(0.0, min(x, w-1)); y = max(0.0, min(y, h-1))
            v += [x/w, y/h, 2]
        else:
            v += [0.0, 0.0, 0]
    return v

def compute_bbox_from_kpts(kpts_xy: np.ndarray, img_w: int, img_h: int, pad=0.30, min_wh: int = 16):
    """
    BBox from visible keypoints, padded and clamped, with a minimum size to
    ensure slender tweezers still get context.
    """
    if kpts_xy.size == 0:
        return 0.5, 0.5, 1.0, 1.0
    x0, y0 = kpts_xy[:,0].min(), kpts_xy[:,1].min()
    x1, y1 = kpts_xy[:,0].max(), kpts_xy[:,1].max()
    w = max(x1 - x0, min_wh); h = max(y1 - y0, min_wh)
    x0 -= w*pad; x1 += w*pad; y0 -= h*pad; y1 += h*pad
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(img_w-1, x1); y1 = min(img_h-1, y1)
    cx, cy = (x0+x1)/2.0, (y0+y1)/2.0
    return cx/img_w, cy/img_h, (x1-x0)/img_w, (y1-y0)/img_h


# =========================
# Visibility guards & helpers for occlusions
# =========================
def _dist_point_to_seg(px, py, ax, ay, bx, by):
    apx, apy = px-ax, py-ay
    abx, aby = bx-ax, by-ay
    ab2 = abx*abx + aby*aby + 1e-6
    t = max(0.0, min(1.0, (apx*abx + apy*aby)/ab2))
    cx, cy = ax + t*abx, ay + t*aby
    dx, dy = px-cx, py-cy
    return (dx*dx + dy*dy) ** 0.5

def _covers_two_critical_line(kdict, p0, p1, thickness):
    crit = ["left_tip","right_tip","hinge_center"]
    ax,ay = float(p0[0]), float(p0[1])
    bx,by = float(p1[0]), float(p1[1])
    hit = 0
    for n in crit:
        if n in kdict:
            x,y = float(kdict[n][0]), float(kdict[n][1])
            if _dist_point_to_seg(x,y,ax,ay,bx,by) <= 0.55*thickness:
                hit += 1
    return hit >= 2

def _covers_two_critical_ellipse(kdict, center, ax, ay):
    crit = ["left_tip","right_tip","hinge_center"]
    cx,cy = center
    ax=max(1,ax); ay=max(1,ay)
    hit = 0
    for n in crit:
        if n in kdict:
            x,y = float(kdict[n][0]), float(kdict[n][1])
            u = (x-cx)/ax; v = (y-cy)/ay
            if (u*u + v*v) <= 1.0: hit += 1
    return hit >= 2

def _seg_with_offset(p1, p2, t0, t1, off):
    p1 = np.array(p1, np.float32); p2 = np.array(p2, np.float32)
    v = p2 - p1
    n = np.array([-v[1], v[0]], np.float32)  # perpendicular
    nv = n / (np.linalg.norm(n) + 1e-6)
    a = p1*(1-t0) + p2*t0 + nv*off
    b = p1*(1-t1) + p2*t1 + nv*off
    return a, b


# =========================
# Medium-strength occlusions + OR spotlight + video-like
# (class-agnostic; density controls how busy each occluded image is)
# =========================
def apply_band_occlusions(img, objects, prob=0.35,
                          density=0.6,
                          tmin=12, tmax=28,
                          between_p=0.75, max_offset=22.0,
                          soft_alpha=0.80):
    colors = [(255,255,255),(255,245,235)]
    out = img.copy()
    for obj in objects:
        if _rng.random() > prob:
            continue
        k = obj.get("keypoints", {})
        segs = []
        def add_seg(a, b):
            if a in k and b in k:
                segs.append((k[a], k[b]))
        add_seg("hinge_center", "left_tip")
        add_seg("hinge_center", "right_tip")
        add_seg("hinge_center", "left_ring")
        add_seg("hinge_center", "right_ring")
        if not segs:
            continue

        base = int(_rng.integers(1, 4))  # 1..3
        n_strokes = max(1, int(round(base * max(0.25, density))))
        len_scale = 0.65 + 0.35*density
        th_scale  = 0.55 + 0.45*density
        a_scale   = 0.55 + 0.45*density

        for _ in range(n_strokes):
            p1, p2 = segs[int(_rng.integers(0, len(segs)))]
            p1 = np.array(p1, np.float32); p2 = np.array(p2, np.float32)
            if np.linalg.norm(p2 - p1) < 10:
                continue
            t0 = float(_rng.uniform(0.10, 0.70))
            span = float(_rng.uniform(0.22, 0.38)) * len_scale
            t1 = min(1.0, t0 + span)

            if _rng.random() < between_p:
                offset = float(_rng.uniform(-max_offset, max_offset))
                c0, c1 = _seg_with_offset(p1, p2, t0, t1, offset)
            else:
                c0 = p1*(1 - t0) + p2*t0
                c1 = p1*(1 - t1) + p2*t1

            th = max(6, int(_rng.integers(tmin, tmax+1) * th_scale))
            color = colors[int(_rng.integers(0, len(colors)))]
            if _covers_two_critical_line(k, c0, c1, th):
                continue

            ov = out.copy()
            cv2.line(ov, tuple(np.int32(c0)), tuple(np.int32(c1)), color, th, cv2.LINE_AA)
            out = cv2.addWeighted(ov, soft_alpha*a_scale, out, 1.0-soft_alpha*a_scale, 0)
    return out


def apply_glove_occlusions(img, objects, prob=0.55,
                           density=0.6,
                           main_p=0.80, blue_p=0.28,
                           main_axes=(28, 64), blue_axes=(22, 44),
                           soft_alpha_main=0.62, soft_alpha_blue=0.55,
                           between_p=0.60, max_offset=22.0):
    whites=[(228,228,228),(242,242,242)]
    blues=[(208,220,238)]
    out = img.copy()

    for obj in objects:
        if _rng.random() > prob:
            continue
        k = obj.get("keypoints", {})
        centers=[]

        # anchors at kpts (fewer at low density)
        for name in ("hinge_center","left_tip","right_tip"):
            if name in k and _rng.random() < (0.75*density + 0.15):
                centers.append((float(k[name][0]), float(k[name][1])))

        # between-kpt midpoints
        pairs=[]
        if "hinge_center" in k and "left_tip" in k:
            pairs.append((k["hinge_center"], k["left_tip"]))
        if "hinge_center" in k and "right_tip" in k:
            pairs.append((k["hinge_center"], k["right_tip"]))
        for a,b in pairs:
            if _rng.random() < between_p * (0.7 + 0.3*density):
                a = np.array(a, np.float32); b = np.array(b, np.float32)
                m = 0.5*(a+b)
                v = b-a; n = np.array([-v[1], v[0]], np.float32)
                n = n/(np.linalg.norm(n)+1e-6)
                m = m + n*float(_rng.uniform(-max_offset, max_offset))
                centers.append((float(m[0]), float(m[1])))

        size_scale = 0.6 + 0.4*density
        alpha_scale = 0.6 + 0.4*density

        for cx, cy in centers:
            if _rng.random() < (main_p * (0.6 + 0.4*density)):
                ax = int(_rng.integers(main_axes[0], main_axes[1]+1) * size_scale)
                ay = int(_rng.integers(main_axes[0], main_axes[1]+1) * size_scale)
                if not _covers_two_critical_ellipse(k, (cx,cy), ax, ay):
                    ov = out.copy()
                    cv2.ellipse(ov, (int(cx),int(cy)), (ax,ay), _rng.uniform(0,180), 0, 360,
                                whites[int(_rng.integers(0,len(whites)))], -1, cv2.LINE_AA)
                    out = cv2.addWeighted(ov, soft_alpha_main*alpha_scale, out, 1.0-soft_alpha_main*alpha_scale, 0)

            if _rng.random() < (blue_p * (0.6 + 0.4*density)):
                ax = int(_rng.integers(blue_axes[0], blue_axes[1]+1) * size_scale)
                ay = int(_rng.integers(blue_axes[0], blue_axes[1]+1) * size_scale)
                if _covers_two_critical_ellipse(k, (cx,cy), ax, ay):
                    continue
                ov = out.copy()
                cv2.ellipse(ov, (int(cx),int(cy)), (ax,ay), _rng.uniform(0,180), 0, 360,
                            blues[0], -1, cv2.LINE_AA)
                out = cv2.addWeighted(ov, soft_alpha_blue*alpha_scale, out, 1.0-soft_alpha_blue*alpha_scale, 0)
    return out


def video_like(img, prob=0.35):
    if _rng.random()>prob: return img
    a=float(_rng.uniform(0.9,1.1)); b=float(_rng.uniform(-10,10))
    img=cv2.convertScaleAbs(img, alpha=a, beta=b)
    if _rng.random()<0.3:
        h,w=img.shape[:2]; sigma=float(_rng.uniform(2,5))
        noise=_rng.normal(0,sigma,(h,w,3)).astype(np.float32)
        img=np.clip(img.astype(np.float32)+noise,0,255).astype(np.uint8)
    q=int(_rng.integers(60,85))
    _,enc=cv2.imencode(".jpg",img,[int(cv2.IMWRITE_JPEG_QUALITY),q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def or_spotlight_center(img,
                        core_sigma_ratio=0.18,
                        halo_sigma_ratio=0.42,
                        gain_core=0.70,
                        gain_halo=0.35,
                        warm_cast=(1.00,1.05,1.12),
                        edge_vignette=0.25,
                        bloom_strength=0.55,
                        bloom_threshold=225):
    h,w=img.shape[:2]
    out=img.astype(np.float32).copy()
    out=np.clip(out*np.array(warm_cast,np.float32),0,255)
    cx = w*(0.50+float(_rng.uniform(-0.03,0.03)))
    cy = h*(0.55+float(_rng.uniform(-0.03,0.03)))
    Y,X=np.ogrid[:h,:w]
    R2=(X-cx)**2 + (Y-cy)**2
    smin=float(min(h,w))
    core_sigma=(core_sigma_ratio*smin)**2
    halo_sigma=(halo_sigma_ratio*smin)**2
    core=gain_core*np.exp(-R2/(2.0*core_sigma))
    halo=gain_halo*np.exp(-R2/(2.0*halo_sigma))
    gain=1.0+core+halo
    r_edge=np.sqrt((X-w/2)**2 + (Y-h/2)**2)
    vign=1.0 - edge_vignette*((r_edge/(r_edge.max()+1e-6))**2)
    out=np.clip(out*gain[...,None]*vign[...,None],0,255)
    luma=0.114*out[...,0]+0.587*out[...,1]+0.299*out[...,2]
    hot=(luma>bloom_threshold).astype(np.float32)
    k=int(max(7, round(0.03*smin))) | 1
    hot_blur=cv2.GaussianBlur(hot,(k,k),0)
    glow=cv2.GaussianBlur(out,(k,k),0)*(hot_blur[...,None]*bloom_strength)
    out=np.clip(out+glow,0,255)
    return out.astype(np.uint8)

def apply_hand_grip_occluder(img, objects, prob=0.40,
                             density=0.6,
                             palm_scale=(52, 78), finger_len=(42, 64),
                             finger_w=(10, 16), n_fingers=(2, 4),
                             wrap_jitter=12.0, tone=("white","blue")):
    whites=[(228,228,228),(242,242,242)]
    blues=[(208,220,238)]
    out=img.copy()

    for obj in objects:
        if _rng.random() > prob:
            continue
        k = obj.get("keypoints", {})
        if "hinge_center" not in k:
            continue

        targets=[]
        if "left_ring" in k:  targets.append(k["left_ring"])
        if "right_ring" in k: targets.append(k["right_ring"])
        if "left_tip" in k:   targets.append(k["left_tip"])
        if "right_tip" in k:  targets.append(k["right_tip"])
        if not targets:
            continue

        t = np.array(targets[int(_rng.integers(0,len(targets)))], np.float32)
        h0 = np.array(k["hinge_center"], np.float32)
        v  = t - h0
        if np.linalg.norm(v) < 8:
            continue
        v  = v/(np.linalg.norm(v)+1e-6)
        n  = np.array([-v[1], v[0]], np.float32)
        cx, cy = (h0 + 0.45*(t-h0))
        cx += n[0]*float(_rng.uniform(-wrap_jitter, wrap_jitter))
        cy += n[1]*float(_rng.uniform(-wrap_jitter, wrap_jitter))

        # palette
        mode = tone if isinstance(tone, str) else (tone[int(_rng.integers(0,2))] if isinstance(tone, tuple) else "white")
        if mode == "blue":
            color = blues[0]; alpha_palm, alpha_finger = 0.55, 0.50
        else:
            color = whites[int(_rng.integers(0,len(whites)))]; alpha_palm, alpha_finger = 0.60, 0.55

        size_scale  = 0.65 + 0.35*density
        alpha_scale = 0.6  + 0.4*density

        # palm
        palml = int(_rng.integers(palm_scale[0], palm_scale[1]+1) * size_scale)
        palmw = int(_rng.integers(palm_scale[0]//2, palm_scale[1]//2 + 1) * size_scale)
        angle = float(np.degrees(np.arctan2(v[1], v[0])) + 90.0)
        ov = out.copy()
        cv2.ellipse(ov, (int(cx),int(cy)), (palmw, palml), angle, 0, 360, color, -1, cv2.LINE_AA)
        out = cv2.addWeighted(ov, alpha_palm*alpha_scale, out, 1.0-alpha_palm*alpha_scale, 0)

        # fingers
        base_nf = int(_rng.integers(n_fingers[0], n_fingers[1]+1))
        nf = max(1, int(round(base_nf * max(0.3, density))))
        spacing = palml / max(nf,1)
        for i in range(nf):
            base = np.array([cx, cy], np.float32) + v*((i - (nf-1)/2.0)*spacing*0.55)
            base += n*float(_rng.uniform(-wrap_jitter, wrap_jitter))
            fl = int(_rng.integers(finger_len[0], finger_len[1]+1) * size_scale)
            fw = int(_rng.integers(finger_w[0], finger_w[1]+1) * size_scale)
            angle_f = float(np.degrees(np.arctan2(v[1], v[0])) + _rng.uniform(65, 115))
            ov = out.copy()
            cv2.ellipse(ov, (int(base[0]), int(base[1])), (fw, fl), angle_f, 0, 360, color, -1, cv2.LINE_AA)
            out = cv2.addWeighted(ov, alpha_finger*alpha_scale, out, 1.0-alpha_finger*alpha_scale, 0)
    return out


# =========================
# Conversion (to YOLO-pose labels/images)
# =========================
def _convert_one(img_p: Path, ann_p: Path, out_img: Path, out_lbl: Path):
    d = read_json(ann_p)
    img = cv2.imread(str(img_p)); h,w = img.shape[:2]
    lines = []
    for obj in d.get("objects", []):
        inst = obj.get("instrument")
        if inst not in CLS2ID:
            continue
        cls = CLS2ID[inst] 
        kdict = obj.get("keypoints", {})
        kvec = build_kvec(kdict, w, h)
        vis_xy = np.array([[kdict[k][0], kdict[k][1]] for k in kdict], np.float32) if kdict else np.zeros((0,2))
        cx,cy,bw,bh = compute_bbox_from_kpts(vis_xy, w, h, pad=0.30, min_wh=16)
        fields = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"] + \
                 [f"{v:.6f}" if i%3!=2 else str(int(v)) for i,v in enumerate(kvec)]
        lines.append(" ".join(fields))
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_p, out_img)
    write_txt(out_lbl, lines)


# =========================
# PREPARE (OR spotlight + occlusions + hand-grip + video-like)
# =========================
def _prepare(root: Path, out_dir: Path, val_pct: float, seed: int,
             aug_copies: int, band_prob: float, glove_prob: float,
             or_prob: float, video_prob: float, grip_prob: float,
             band_density: float, glove_density: float, grip_density: float,
             tweezer_boost: int, force: bool):
    yolo_root = out_dir/"yolo"
    meta_p = yolo_root/".prep_meta.json"

    pairs = gather_pairs(root)
    assert pairs, f"No (image,json) pairs found in {root}"

    params = dict(
        aug_copies=aug_copies, band_prob=band_prob, glove_prob=glove_prob,
        or_prob=or_prob, video_prob=video_prob, grip_prob=grip_prob,
        band_density=band_density, glove_density=glove_density, grip_density=grip_density,
        tweezer_boost=tweezer_boost, seed=seed, val_pct=val_pct,
        class_agnostic=True,
        schema="prep_v6_or_spotlight_bands_gloves_handgrip_video_CA"
    )
    fp = fingerprint(pairs, params)
    old = load_meta(meta_p)
    if (not force) and old.get("fingerprint") == fp and (yolo_root/"data.yaml").exists():
        print("[prepare] up-to-date; skipping conversion."); return yolo_root/"data.yaml"

    if yolo_root.exists(): shutil.rmtree(yolo_root)
    for s in ["train","val"]:
        (yolo_root/f"{s}/images").mkdir(parents=True, exist_ok=True)
        (yolo_root/f"{s}/labels").mkdir(parents=True, exist_ok=True)

    train_pairs, val_pairs = split_pairs(pairs, val_pct, seed)

    # VAL (plain)
    for img_p, ann_p in val_pairs:
        _convert_one(img_p, ann_p, yolo_root/"val/images"/img_p.name, yolo_root/"val/labels"/f"{img_p.stem}.txt")

    # TRAIN (base + prepared copies; optional extra copies if tweezers present)
    for img_p, ann_p in train_pairs:
        base_img = yolo_root/"train/images"/img_p.name
        base_lbl = yolo_root/"train/labels"/f"{img_p.stem}.txt"
        _convert_one(img_p, ann_p, base_img, base_lbl)

        d = read_json(ann_p)
        has_tweezers = any(o.get("instrument") == "tweezers" for o in d.get("objects", []))
        total_copies = max(0, aug_copies) + (tweezer_boost if has_tweezers else 0)

        for k in range(total_copies):
            aug = cv2.imread(str(img_p))

            # --- occlusions (keypoint-aware, often BETWEEN keypoints) ---
            aug = apply_band_occlusions(aug, d.get("objects", []), prob=band_prob, density=band_density)
            aug = apply_glove_occlusions(aug, d.get("objects", []), prob=glove_prob, density=glove_density)
            aug = apply_hand_grip_occluder(aug, d.get("objects", []), prob=grip_prob, density=grip_density)

            # --- OR spotlight + video-like ---
            if _rng.random() < or_prob:
                aug = or_spotlight_center(aug)
            aug = video_like(aug, prob=video_prob)

            name = f"{img_p.stem}_aug{k+1}{img_p.suffix}"
            cv2.imwrite(str(yolo_root/"train/images"/name), aug)
            base_label_lines = (yolo_root/"train/labels"/f"{img_p.stem}.txt").read_text().splitlines()
            write_txt(yolo_root/"train/labels"/f"{img_p.stem}_aug{k+1}.txt", base_label_lines)

    # data.yaml for YOLOv11-pose (single class)
    (yolo_root/"data.yaml").write_text(
        f"""# generated by instrument_pose_yolov11.py
        path: {yolo_root}
        train: train/images
        val: val/images
        names: {NAMES}
        nc: {len(NAMES)}
        kpt_shape: [{NUM_KPTS}, 3]
        flip_idx: {FLIP_IDX}
        """
    )
    save_meta(meta_p, {"fingerprint": fp, "created": int(time.time()),
                       "counts": {"train": len(train_pairs), "val": len(val_pairs)}, "params": params})
    print(f"[prepare] wrote {yolo_root/'data.yaml'}")
    return yolo_root/"data.yaml"


# =========================
# TRAIN (YOLOv11-pose)
# =========================
def _train(model_name: str, data_yaml: Path, epochs: int, imgsz: int, batch: int,
           device: str, workers: int, run_project: str, run_name: str):
    assert data_yaml.exists(), f"Missing {data_yaml}. Run --step prepare first."

    model = YOLO(model_name)

    model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
        device=device, workers=workers,
        # Light geometry aug (Ultralytics will transform keypoints correctly)
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
        fliplr=0.0, flipud=0.0, perspective=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
        # Modest color jitter
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        # Optim / schedule
        lr0=5e-3, lrf=0.1, momentum=0.937, weight_decay=5e-4,
        cos_lr=True, close_mosaic=10, amp=True, pretrained=True,
        cache="ram", patience=20, plots=True, save_period=-1,
        freeze=0,
        project=run_project, name=run_name, exist_ok=True, resume=False
    )

    model.val(
        data=str(data_yaml), imgsz=imgsz, device=device, workers=workers,
        project=run_project, name=run_name, exist_ok=True, plots=True
    )


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser("Class-agnostic YOLOv11-Pose with OR-lighting + hand-grip aug")
    ap.add_argument("--step", choices=["prepare","train","prepare+train"], required=True)
    ap.add_argument("--root", type=str, help="dataset root with images/ and ann/ (required for prepare)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--val_pct", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--aug_copies", type=int, default=1)
    ap.add_argument("--tweezer_boost", type=int, default=0, help="extra copies ONLY for images containing tweezers")
    ap.add_argument("--band_prob", type=float, default=0.35)
    ap.add_argument("--glove_prob", type=float, default=0.55)
    ap.add_argument("--grip_prob", type=float, default=0.40, help="probability to draw a hand-grip occluder")
    ap.add_argument("--or_prob", type=float, default=0.50)
    ap.add_argument("--video_prob", type=float, default=0.35)

    ap.add_argument("--band_per_image",  type=float, default=0.60, help="0..1: how many/strong band occlusions per occluded image")
    ap.add_argument("--glove_per_image", type=float, default=0.60, help="0..1: how many/strong glove patches per occluded image")
    ap.add_argument("--grip_per_image",  type=float, default=0.60, help="0..1: size/opacity/#fingers for hand-grip per image")

    # training
    ap.add_argument("--model", type=str, default="yolo11l-pose.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--workers", type=int, default=None)

    # run folder control
    ap.add_argument("--run_project", type=str, default="runs/pose")
    ap.add_argument("--run_name", type=str, default="train_hdri_background")

    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    yolo_root = out_dir/"yolo"
    if args.workers is None:
        args.workers = max(2, min(os.cpu_count() or 6, 8))

    if args.step in ("prepare","prepare+train"):
        assert args.root, "--root is required for prepare"
        data_yaml = _prepare(
            Path(args.root), out_dir, args.val_pct, args.seed,
            args.aug_copies, args.band_prob, args.glove_prob,
            args.or_prob, args.video_prob, args.grip_prob,
            args.band_per_image, args.glove_per_image, args.grip_per_image,
            args.tweezer_boost, args.force
        )
    else:
        data_yaml = yolo_root/"data.yaml"

    if args.step in ("train","prepare+train"):
        _train(args.model, data_yaml, args.epochs, args.imgsz,
               args.batch, args.device, args.workers,
               run_project=args.run_project, run_name=args.run_name)
        print("Training done.")

if __name__ == "__main__":
    main()

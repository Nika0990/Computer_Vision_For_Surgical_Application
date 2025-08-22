#!/usr/bin/env python3
"""
synthetic_data_generator.py 
------------------------------------------------------------------------
Pinned defaults:
- Camera cap: 24 mm (wide)
- HDRI zoom-out: mapping scale 800–1600
- Size buckets (fraction of image diagonal):
    small    0.25–0.35   ( 5%)
    medium   0.35–0.50   (30%)
    large    0.50–0.75   (40%)
    closeup  0.75–0.95   (25%)
- If any object is close-up, force single-tool for that frame.
"""

from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
from typing import Dict, List, Tuple

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Vector
from PIL import Image, ImageDraw, ImageFilter

# ──────────────────────────────────────────────────────────────────────────────
# Caches
# ──────────────────────────────────────────────────────────────────────────────
_mesh_data_cache: Dict[str, bpy.types.Mesh] = {}
_material_cache: Dict[str, bpy.types.Material] = {}
_hdri_cache: List[Path] = []

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_cmd() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser("Surgical-tool renderer (extra HDRI zoom-out + bigger tools)")
    p.add_argument("--models_dir", required=True, type=Path)
    p.add_argument("--hdris_dir", required=True, type=Path)
    p.add_argument("--camera", required=True, type=Path)
    p.add_argument("--keypoints", required=True, type=Path)
    p.add_argument("--output_dir", type=Path, default=Path("./synthetic_dataset"))
    p.add_argument("--num_images", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--render_samples", type=int, default=32)
    p.add_argument("--fast_mode", action="store_true")

    p.add_argument("--lens_cap_mm", type=float, default=24.0)

    p.add_argument("--hdri_zoom_min", type=float, default=800.0)
    p.add_argument("--hdri_zoom_max", type=float, default=1600.0)

    p.add_argument("--small_lo",  type=float, default=0.25)
    p.add_argument("--small_hi",  type=float, default=0.35)
    p.add_argument("--med_lo",    type=float, default=0.35)
    p.add_argument("--med_hi",    type=float, default=0.50)
    p.add_argument("--large_lo",  type=float, default=0.50)
    p.add_argument("--large_hi",  type=float, default=0.75)
    p.add_argument("--close_lo",  type=float, default=0.75)
    p.add_argument("--close_hi",  type=float, default=0.95)

    p.add_argument("--prob_small",  type=float, default=0.05)
    p.add_argument("--prob_medium", type=float, default=0.30)
    p.add_argument("--prob_large",  type=float, default=0.40)
    p.add_argument("--prob_close",  type=float, default=0.25)

    p.add_argument("--two_tools_prob", type=float, default=0.40)
    return p.parse_args(argv)

# ──────────────────────────────────────────────────────────────────────────────
# Render & camera
# ──────────────────────────────────────────────────────────────────────────────
def setup_render(res_x: int, res_y: int, samples: int, fast_mode: bool) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    addon = bpy.context.preferences.addons.get("cycles")
    if addon and hasattr(addon, "preferences") and addon.preferences.compute_device_type != "NONE":
        scene.cycles.device = "GPU"
    else:
        scene.cycles.device = "CPU"
    scene.cycles.samples = samples if not fast_mode else max(16, samples // 2)
    for name, val in [
        ("max_bounces", 2 if fast_mode else 4),
        ("diffuse_bounces", 1 if fast_mode else 2),
        ("glossy_bounces", 1 if fast_mode else 2),
        ("transmission_bounces", 1 if fast_mode else 2),
        ("volume_bounces", 0),
    ]:
        if hasattr(scene.cycles, name):
            setattr(scene.cycles, name, val)
    if hasattr(scene.cycles, 'use_denoising'):
        scene.cycles.use_denoising = True
        if hasattr(scene.cycles, 'denoiser'):
            scene.cycles.denoiser = 'OPTIX' if scene.cycles.device == 'GPU' else 'OPENIMAGEDENOISE'
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

def setup_camera_for_surgery(cam_json: Path, lens_cap_mm: float) -> bpy.types.Object:
    with cam_json.open() as f:
        camd = json.load(f)
    cam = bpy.data.objects.get("SyntheticCam")
    if cam is None:
        cam_data = bpy.data.cameras.new("SyntheticCam")
        cam = bpy.data.objects.new("SyntheticCam", cam_data)
        bpy.context.collection.objects.link(cam)
    else:
        cam_data = cam.data

    w, h, fx = camd["width"], camd["height"], camd["fx"]
    sensor_w = 36.0
    sensor_h = sensor_w * h / w
    cam_data.sensor_width, cam_data.sensor_height = sensor_w, sensor_h
    calculated_lens = fx * sensor_w / w
    cam_data.lens = min(calculated_lens, max(8.0, lens_cap_mm))
    cam.data.shift_x = -(camd["cx"] - w/2) / w
    cam.data.shift_y =  (camd["cy"] - h/2) / h
    cam.location = (0.0, 0.0, 0.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    cam.data.clip_start = 0.01
    cam.data.clip_end   = 200.0
    bpy.context.scene.camera = cam
    print(f"Camera lens: {cam_data.lens:.1f} mm | FOV: {math.degrees(cam_data.angle):.1f}°")
    return cam

# ──────────────────────────────────────────────────────────────────────────────
# World / HDRI
# ──────────────────────────────────────────────────────────────────────────────
def initialize_hdri_cache(root: Path) -> None:
    global _hdri_cache
    if not _hdri_cache:
        _hdri_cache = list(root.glob("**/*.hdr"))
        if not _hdri_cache:
            raise FileNotFoundError(f"No .hdr files under {root}")
        print(f"Cached {len(_hdri_cache)} HDRI files")

def random_hdri() -> Path:
    return random.choice(_hdri_cache)

def apply_hdri_zoomed(world: bpy.types.World, hdr: Path, strength: float,
                      zoom_min: float, zoom_max: float) -> None:
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    tex = nt.nodes.new("ShaderNodeTexEnvironment")
    tex.image = bpy.data.images.get(hdr.name) or bpy.data.images.load(str(hdr), check_existing=True)
    tex.projection = 'EQUIRECTANGULAR'
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputWorld")
    mapping = nt.nodes.new("ShaderNodeMapping")
    texcoord = nt.nodes.new("ShaderNodeTexCoord")

    s = random.uniform(zoom_min, zoom_max)
    mapping.inputs["Scale"].default_value = (s, s, s)
    mapping.inputs["Rotation"].default_value = (0, 0, random.uniform(0, 2*math.pi))
    mapping.inputs["Location"].default_value = (random.uniform(-0.25, 0.25),
                                                random.uniform(-0.25, 0.25), 0.0)

    nt.links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

def setup_environment_for_surgery(world: bpy.types.World, zoom_min: float, zoom_max: float) -> None:
    hdr = random_hdri()
    strength = random.uniform(0.35, 0.7)
    apply_hdri_zoomed(world, hdr, strength, zoom_min, zoom_max)

def add_minimal_lighting(scene: bpy.types.Scene) -> None:
    name = "FastLight"
    if name not in bpy.data.lights:
        ldat = bpy.data.lights.new(name, type='AREA')
        ldat.energy = random.uniform(60, 100)
        ldat.color = (0.95, 0.98, 1.0)
        ldat.size = 2.0
        lobj = bpy.data.objects.new(name, ldat)
        scene.collection.objects.link(lobj)
    else:
        lobj = bpy.data.objects[name]
    lobj.location = (random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(4, 6))

# ──────────────────────────────────────────────────────────────────────────────
# Assets
# ──────────────────────────────────────────────────────────────────────────────
def get_cached_material(name: str) -> bpy.types.Material:
    if name not in _material_cache:
        m = bpy.data.materials.new(name=name)
        m.use_nodes = True
        bsdf = m.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.9, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.9
        bsdf.inputs["Roughness"].default_value = 0.1
        _material_cache[name] = m
    return _material_cache[name]

def import_mesh_cached(obj_path: Path) -> bpy.types.Object:
    key = str(obj_path)
    if key in _mesh_data_cache:
        mesh_copy = _mesh_data_cache[key].copy()
        obj = bpy.data.objects.new(f"{obj_path.stem}_{random.randint(1000,9999)}", mesh_copy)
        bpy.context.collection.objects.link(obj)
        if not obj.data.materials and mesh_copy.materials:
            for mat in mesh_copy.materials:
                obj.data.materials.append(mat)
    else:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.import_scene.obj(filepath=str(obj_path), axis_forward="-Z", axis_up="Y", split_mode="OFF")
        obj = bpy.context.selected_objects[0]
        obj.name = f"{obj_path.stem}_{random.randint(1000,9999)}"
        obj.data.name = f"{obj_path.stem}_mesh"
        if not obj.data.materials:
            obj.data.materials.append(get_cached_material(f"{obj_path.stem}_material"))
        mesh_to_cache = obj.data.copy()
        mesh_to_cache.name = f"{obj_path.stem}_cached_mesh"
        _mesh_data_cache[key] = mesh_to_cache
    return obj

def load_keypoints(kp_json: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    data = json.loads(kp_json.read_text())
    m: Dict[str, Dict[str, Dict[str, int]]] = {}
    for inst in data["instruments"]:
        m.setdefault(inst["instrument"], {})[inst["mesh_file"]] = inst["keypoints"]
    return m

# ──────────────────────────────────────────────────────────────────────────────
# Projection & framing
# ──────────────────────────────────────────────────────────────────────────────
def project_world_to_image(world_co: Vector, cam: bpy.types.Object, scene: bpy.types.Scene) -> Tuple[float, float]:
    ndc = world_to_camera_view(scene, cam, world_co)
    x_px = ndc.x * scene.render.resolution_x
    y_px = (1.0 - ndc.y) * scene.render.resolution_y
    return float(x_px), float(y_px)

def fast_screen_bounds_check(obj: bpy.types.Object, cam: bpy.types.Object, scene: bpy.types.Scene,
                             margin_ratio: float = 0.01) -> bool:
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    pts = [project_world_to_image(c, cam, scene) for c in corners]
    xs, ys = zip(*pts)
    min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
    w, h = scene.render.resolution_x, scene.render.resolution_y
    mx, my = w*margin_ratio, h*margin_ratio
    return (min_x >= mx and max_x <= w-mx and min_y >= my and max_y <= h-my)

def _bbox_on_image_px(obj: bpy.types.Object, cam: bpy.types.Object, scene: bpy.types.Scene) -> Tuple[float, float, float, float]:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    pts = [project_world_to_image(c, cam, scene) for c in corners]
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)

def adapt_object_size(obj: bpy.types.Object, cam: bpy.types.Object, scene: bpy.types.Scene,
                      target_lo: float, target_hi: float, max_iters: int = 12) -> None:
    w, h = scene.render.resolution_x, scene.render.resolution_y
    img_diag = (w*w + h*h) ** 0.5
    for _ in range(max_iters):
        bpy.context.view_layer.update()
        x0, y0, x1, y1 = _bbox_on_image_px(obj, cam, scene)
        bw, bh = (x1 - x0), (y1 - y0)
        frac = ((bw*bw + bh*bh) ** 0.5) / img_diag
        if target_lo <= frac <= target_hi:
            return
        step = 0.90 if frac > target_hi else 1.12
        obj.scale = tuple(s * step for s in obj.scale)
        if not fast_screen_bounds_check(obj, cam, scene, margin_ratio=0.01):
            obj.scale = tuple(s / step for s in obj.scale)
            return

def choose_bucket(args) -> Tuple[str, Tuple[float,float]]:
    probs = np.array([args.prob_small, args.prob_medium, args.prob_large, args.prob_close], dtype=float)
    probs = probs / probs.sum()
    name = np.random.choice(["small","medium","large","close"], p=probs)
    if name == "small":  return name, (args.small_lo,  args.small_hi)
    if name == "medium": return name, (args.med_lo,    args.med_hi)
    if name == "large":  return name, (args.large_lo,  args.large_hi)
    return "close", (args.close_lo, args.close_hi)

def random_pose_with_bucket(obj: bpy.types.Object, cam: bpy.types.Object, scene: bpy.types.Scene,
                            bucket: str, target: Tuple[float,float], max_attempts: int = 30) -> None:
    t_lo, t_hi = target
    for _ in range(max_attempts):
        if bucket == "close":
            z = -random.uniform(2.8, 4.5); s0 = random.uniform(0.55, 0.95)
        elif bucket == "large":
            z = -random.uniform(3.8, 6.5); s0 = random.uniform(0.45, 0.80)
        elif bucket == "medium":
            z = -random.uniform(5.5, 8.5); s0 = random.uniform(0.35, 0.65)
        else:  # small
            z = -random.uniform(7.5, 10.0); s0 = random.uniform(0.28, 0.50)

        obj.scale = (s0, s0, s0)
        obj.rotation_euler = Euler((
            random.uniform(-math.pi/3,  math.pi/3),
            random.uniform(-math.pi/10, math.pi/10),
            random.uniform(0, 2*math.pi)
        ), "XYZ")

        fov_x, fov_y = cam.data.angle_x, cam.data.angle_y
        max_x = math.tan(fov_x * 0.46) * (-z)
        max_y = math.tan(fov_y * 0.46) * (-z)
        obj.location = (random.uniform(-max_x, max_x),
                        random.uniform(-max_y, max_y),
                        z)

        bpy.context.view_layer.update()
        if fast_screen_bounds_check(obj, cam, scene, margin_ratio=0.01):
            adapt_object_size(obj, cam, scene, t_lo, t_hi)
            return

    obj.location = (0, 0, -5.0)
    obj.rotation_euler = (0, 0, 0)
    obj.scale = (0.7, 0.7, 0.7)

# ──────────────────────────────────────────────────────────────────────────────
# Post-processing (optional)
# ──────────────────────────────────────────────────────────────────────────────
def add_blood_stains_fast(img_path: Path, prob: float = 0.2) -> None:
    if random.random() >= prob: return
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for _ in range(random.randint(1, 2)):
            cx = random.randint(int(w*0.1), int(w*0.9))
            cy = random.randint(int(h*0.1), int(h*0.9))
            r  = random.randint(int(min(w,h)*0.01), int(min(w,h)*0.03))
            draw.ellipse([cx-r, cy-r, cx+r, cy+r],
                         fill=(random.randint(80,120), random.randint(5,15),
                               random.randint(5,15), random.randint(100,150)))
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.5))
        Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB").save(img_path, optimize=True)
    except Exception as e:
        print(f"[WARN] Blood stain processing failed: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────────────────────────────────────
def reset_scene_fast(exclude: Tuple[str, ...] = ("SyntheticCam", "FastLight")) -> None:
    for obj in [o for o in bpy.data.objects if o.name not in exclude]:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0 and not mesh.name.endswith("_cached_mesh"):
            bpy.data.meshes.remove(mesh)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_cmd()
    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed)

    (args.output_dir / "images").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ann").mkdir(exist_ok=True)

    print("Initializing caches...")
    initialize_hdri_cache(args.hdris_dir)

    cam = setup_camera_dfor_surgery(args.camera, lens_cap_mm=args.lens_cap_mm)
    scene = bpy.context.scene
    if not bpy.data.worlds:
        bpy.data.worlds.new("World")
    scene.world = bpy.data.worlds[0]

    setup_render(scene.render.resolution_x, scene.render.resolution_y,
                 args.render_samples, args.fast_mode)

    kp_map = load_keypoints(args.keypoints)

    # Build inventory
    inventory: List[Tuple[str, Path]] = []
    for inst_name, meshes in kp_map.items():
        for mesh_file in meshes:
            p = args.models_dir / inst_name / mesh_file
            if p.exists(): inventory.append((inst_name, p))
            else: print(f"[WARN] missing OBJ skipped: {p}")
    if not inventory:
        sys.exit("No OBJ files found – check paths!")

    world = scene.world
    print(f"Starting generation of {args.num_images} images...")

    for idx in range(args.num_images):
        reset_scene_fast()

        bucket_name, target = choose_bucket(args)
        two_tools_ok = (bucket_name != "close") and (random.random() < args.two_tools_prob)
        k = 2 if two_tools_ok else 1

        chosen = random.sample(inventory, k=k)
        obj_info = []

        for inst_name, obj_path in chosen:
            obj = import_mesh_cached(obj_path)
            random_pose_with_bucket(obj, cam, scene, bucket_name, target)
            obj_info.append((inst_name, obj, obj_path))

        setup_environment_for_surgery(world, args.hdri_zoom_min, args.hdri_zoom_max)
        add_minimal_lighting(scene)

        # Render
        fname = f"{idx:06d}.png"
        fpath = args.output_dir / "images" / fname
        scene.render.filepath = str(fpath)
        bpy.ops.render.render(write_still=True)
        if not args.fast_mode:
            add_blood_stains_fast(fpath, prob=0.2)

        # Annotations
        objects_ann: List[Dict] = []
        for inst_name, obj, mesh_path in obj_info:
            kp_src = kp_map[inst_name][mesh_path.name]
            kp_idx = {k: v - 1 for k, v in kp_src.items()}
            img_kps: Dict[str, List[float]] = {}
            for name, vidx in kp_idx.items():
                vert = obj.data.vertices[vidx]
                world_co = obj.matrix_world @ vert.co
                img_kps[name] = list(project_world_to_image(world_co, cam, scene))
            objects_ann.append({"instrument": inst_name, "keypoints": img_kps})

        ann = {"image": str(Path("images") / fname), "objects": objects_ann}
        with (args.output_dir / "ann" / f"{idx:06d}.json").open("w") as f:
            json.dump(ann, f, separators=(',', ':'))

        if (idx + 1) % 10 == 0:
            print(f"[{idx + 1:>4}/{args.num_images}] Generated (bucket={bucket_name})")

    print("\nDone!")
    print(f"Cache stats: {len(_mesh_data_cache)} meshes, {len(_material_cache)} materials; HDRIs: {len(_hdri_cache)}")

if __name__ == "__main__":
    main()

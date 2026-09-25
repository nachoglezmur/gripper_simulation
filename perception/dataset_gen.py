"""Genera dataset YOLO sintético auto-etiquetado del CDPR.

Render EGL 640x480 con la cámara fija del vídeo + domain randomization:
jitter de cámara, ruido/brillo a nivel imagen, poses aleatorias.
Etiquetas: 0=cubo_robot (esquinas de box_body), 1=calcetin (extent del sock).

Uso:
  python perception/dataset_gen.py --n-train 3000 --n-val 500 --seed 7
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image

from control.model_data import build_model_data
from perception.camera import CameraRig, LOOKAT, DISTANCE, ELEVATION, AZIMUTH

IMG_W, IMG_H = 640, 480
CLASSES = ["cubo_robot", "calcetin"]


def project_corners(rig, center, half):
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pts.append(center + np.array([sx * half[0], sy * half[1], sz * half[2]]))
    us, vs = [], []
    for p in pts:
        u, v = rig.project(p)
        us.append(u)
        vs.append(v)
    return (min(us), min(vs), max(us), max(vs))


def to_yolo(box):
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(IMG_W, x1), min(IMG_H, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    cx = (x0 + x1) / 2 / IMG_W
    cy = (y0 + y1) / 2 / IMG_H
    w = (x1 - x0) / IMG_W
    h = (y1 - y0) / IMG_H
    return cx, cy, w, h


def main():
    import mujoco as mj

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="data/yolo")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("Compilando modelo (texturas una sola vez)...", flush=True)
    model, data = build_model_data(texture_seed=args.seed)
    sock_jid = model.joint("sock_joint").id
    sock_dof = model.jnt_dofadr[sock_jid]
    carrier_id = model.body("carrier").id
    sock_id = model.body("sock_target").id

    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    vopt = mj.MjvOption()
    vopt.flags[mj.mjtVisFlag.mjVIS_TENDON] = True

    import pathlib

    out = pathlib.Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out / "preview").mkdir(parents=True, exist_ok=True)

    with open(out / "cdpr.yaml", "w") as f:
        f.write(
            f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
            f"names: {{0: cubo_robot, 1: calcetin}}\n"
        )

    total = {"train": args.n_train, "val": args.n_val}
    idx = {"train": 0, "val": 0}
    n_done = 0
    with mj.Renderer(model, height=IMG_H, width=IMG_W) as renderer:
        for split, n in total.items():
            for i in range(n):
                # --- pose aleatoria ---
                sx, sy = rng.uniform(-0.9, 0.9, size=2)
                data.qpos[sock_dof : sock_dof + 3] = [sx, sy, 0.026]
                data.qpos[sock_dof + 3 : sock_dof + 7] = [1, 0, 0, 0]
                data.joint("drive_x").qpos[0] = float(rng.uniform(-0.8, 0.8))
                data.joint("drive_y").qpos[0] = float(rng.uniform(-0.8, 0.8))
                data.joint("drive_z").qpos[0] = float(rng.uniform(0.9, 1.6))
                data.ctrl[:] = 0
                mj.mj_forward(model, data)

                # --- jitter de cámara (DR) ---
                cam.lookat[:] = [
                    LOOKAT[0] + rng.uniform(-0.03, 0.03),
                    LOOKAT[1] + rng.uniform(-0.03, 0.03),
                    LOOKAT[2] + rng.uniform(-0.03, 0.03),
                ]
                cam.distance = DISTANCE + rng.uniform(-0.1, 0.1)
                cam.elevation = ELEVATION + rng.uniform(-2, 2)
                cam.azimuth = AZIMUTH + rng.uniform(-2, 2)
                renderer.update_scene(data, camera=cam, scene_option=vopt)
                frame = renderer.render().copy()

                # --- DR a nivel imagen ---
                gain = rng.uniform(0.85, 1.15)
                frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
                noise = rng.normal(0, 4, frame.shape).astype(np.float32)
                frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

                # --- etiquetas con el MISMO rig perturbado ---
                rig = CameraRig(
                    model, data,
                    lookat=tuple(cam.lookat), distance=cam.distance,
                    elevation=cam.elevation, azimuth=cam.azimuth,
                )
                carrier_xyz = data.xpos[carrier_id].copy()
                sock_xyz = data.xpos[sock_id].copy()
                boxes = [
                    (0, project_corners(rig, carrier_xyz, (0.065, 0.065, 0.065))),
                    (1, project_corners(rig, sock_xyz, (0.035, 0.075, 0.035))),
                ]
                lines = []
                for cls, box in boxes:
                    y = to_yolo(box)
                    if y is not None:
                        lines.append(f"{cls} {y[0]:.5f} {y[1]:.5f} {y[2]:.5f} {y[3]:.5f}")

                name = f"{split}_{i:05d}"
                Image.fromarray(frame).save(out / "images" / split / f"{name}.png")
                with open(out / "labels" / split / f"{name}.txt", "w") as f:
                    f.write("\n".join(lines))
                n_done += 1
                if n_done % 250 == 0:
                    print(f"  {n_done}/{args.n_train + args.n_val}...", flush=True)

    # --- preview con cajas ---
    import cv2

    sample = out / "images" / "train" / "train_00000.png"
    img = cv2.imread(str(sample))
    with open(out / "labels" / "train" / "train_00000.txt") as f:
        for line in f:
            c, cx, cy, w, h = line.split()
            x0 = int((float(cx) - float(w) / 2) * IMG_W)
            y0 = int((float(cy) - float(h) / 2) * IMG_H)
            x1 = int((float(cx) + float(w) / 2) * IMG_W)
            y1 = int((float(cy) + float(h) / 2) * IMG_H)
            color = (0, 255, 0) if c == "0" else (255, 0, 0)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    cv2.imwrite(str(out / "preview" / "muestra.jpg"), img)
    print(f"OK: {n_done} pares + preview en {out/'preview'/'muestra.jpg'}", flush=True)


if __name__ == "__main__":
    main()

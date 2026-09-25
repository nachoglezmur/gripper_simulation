"""Genera demos expertas: (frame inicial RGB, pick_xyz) para behavior cloning.

Uso:
  python -m control.expert_demos --n 400 --seed 11
Salida: data/demos/demos.npz. Exige >=95% de éxito o falla.
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    import mujoco as mj
    import numpy as np

    from control.model_data import build_model_data
    from control.trajectory import run_expert

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=str, default="data/demos/demos.npz")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    import pathlib

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print("Compilando modelo...", flush=True)
    model, data = build_model_data()

    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.1, 0.1, 0.65]
    cam.distance = 3.2
    cam.elevation = -24
    cam.azimuth = 110
    vopt = mj.MjvOption()
    vopt.flags[mj.mjtVisFlag.mjVIS_TENDON] = True

    images, picks, ok = [], [], 0
    with mj.Renderer(model, height=480, width=640) as renderer:
        for i in range(args.n):
            sx, sy = rng.uniform(-0.9, 0.9, size=2)
            # Frame inicial: home + calcetín colocado (lo que verá la CNN).
            from control.model_data import reset_to_home

            reset_to_home(model, data)
            sj = model.joint("sock_joint").id
            qa = model.jnt_qposadr[sj]
            data.qpos[qa : qa + 3] = [float(sx), float(sy), 0.026]
            data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
            mj.mj_forward(model, data)
            renderer.update_scene(data, camera=cam, scene_option=vopt)
            images.append(renderer.render().copy())
            # Experto headless (sin render) + validación de éxito.
            success, pick = run_expert(model, data, (float(sx), float(sy)))
            picks.append(pick)
            ok += success
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{args.n} éxito={ok}/{i + 1}", flush=True)

    images = np.stack(images)
    picks = np.stack(picks)
    np.savez_compressed(args.out, images=images, pick_xyz=picks)
    rate = ok / args.n
    print(f"OK: {args.out} {images.shape} éxito={rate:.3f}", flush=True)
    assert rate >= 0.95, f"éxito experto {rate:.3f} < 0.95"


if __name__ == "__main__":
    main()

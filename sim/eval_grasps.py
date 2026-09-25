"""Evaluación headless del bucle aprendido: 20 placements aleatorios.

frame -> YOLO + CNN -> fusión -> waypoints desde el target APRENDIDO
(calcetín en su pose real) -> éxito si termina en la cesta.
Exige >=17/20 y guarda models/cable_schedule_example.csv
(t, x, y, z, L0..L3: la interfaz del mundo real).

Uso: python -m sim.eval_grasps --n 20 --seed 99
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")


def main():
    import mujoco as mj
    import numpy as np

    from control.cable_ik import cable_lengths
    from control.decide import fuse_pick
    from control.model_data import build_model_data, reset_to_home
    from control.trajectory import Z_PICK, build_waypoints, run_expert
    from perception.detector import Detector
    from control.policy_bc import GraspPolicy

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("Cargando YOLO + CNN...", flush=True)
    det = Detector()
    pol = GraspPolicy()
    model, data = build_model_data()

    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.1, 0.1, 0.65]
    cam.distance = 3.2
    cam.elevation = -24
    cam.azimuth = 110
    vopt = mj.MjvOption()
    vopt.flags[mj.mjtVisFlag.mjVIS_TENDON] = True

    ok, src_count = 0, {"cnn": 0, "yolo": 0}
    sched_rows = []
    with mj.Renderer(model, height=480, width=640) as renderer:
        for i in range(args.n):
            sx, sy = rng.uniform(-0.9, 0.9, size=2)
            reset_to_home(model, data)
            sj = model.joint("sock_joint").id
            qa = model.jnt_qposadr[sj]
            data.qpos[qa : qa + 3] = [float(sx), float(sy), 0.026]
            data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
            mj.mj_forward(model, data)
            renderer.update_scene(data, camera=cam, scene_option=vopt)
            frame = renderer.render().copy()

            out = det.detect(frame, robot_z=float(data.joint("drive_z").qpos[0]))
            if out["sock_xy"] is None or out["conf"]["sock"] < 0.5:
                print(f"[{i}] sin detección (conf={out['conf']['sock']:.2f}): fallo", flush=True)
                continue
            cnn_xy = pol.predict(frame)
            pick_xy, src = fuse_pick(cnn_xy, out["sock_xy"], out["conf"]["sock"])
            src_count[src] += 1
            success, _ = run_expert(model, data, (float(sx), float(sy)),
                                    target_xy=(float(pick_xy[0]), float(pick_xy[1])))
            ok += success
            print(f"[{i}] sock=({sx:+.2f},{sy:+.2f}) pick=({pick_xy[0]:+.2f},{pick_xy[1]:+.2f})"
                  f"[{src}] -> {'OK' if success else 'FALLO'}", flush=True)
            if i == 0:
                for t, pos, _g, _label in build_waypoints(pick_xy, (0.55, 0.45)):
                    L = cable_lengths(pos)
                    sched_rows.append([t, pos[0], pos[1], pos[2], *L])

    import pathlib
    csv = pathlib.Path("models/cable_schedule_example.csv")
    np.savetxt(csv, np.array(sched_rows), delimiter=",",
               header="t,x,y,z,L0,L1,L2,L3", comments="")
    print(f"Éxito: {ok}/{args.n} (fuentes: {src_count}). CSV: {csv}", flush=True)
    assert ok >= 17, f"solo {ok}/20 recogidas"


if __name__ == "__main__":
    main()

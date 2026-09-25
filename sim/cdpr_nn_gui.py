"""App interactiva: click coloca el calcetín, ESPACIO recoge solo con la imagen.

Bucle: frame cámara -> YOLO + CNN -> fusión -> waypoints -> IK logueada.
Ventanas: visor 3D (MuJoCo) + imagen de cámara (OpenCV, click aquí).
Teclas en la ventana de cámara: click=colocar, ESPACIO=recoger, R=reset, Q=salir.
"""

import os

os.environ["MUJOCO_GL"] = "glfw"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import time

import mujoco as mj
import mujoco.viewer
import numpy as np

from control.cable_ik import cable_lengths
from control.decide import fuse_pick
from control.model_data import (
    build_model_data,
    gravity_feedforward,
    reset_to_home,
)
from control.policy_bc import GraspPolicy
from control.trajectory import Z_PICK, build_waypoints, make_get_target
from perception.detector import Detector
from sim.click_place import ClickPlacer


def main():
    print("Cargando YOLO + CNN...", flush=True)
    det = Detector()
    pol = GraspPolicy()
    model, data = build_model_data()
    reset_to_home(model, data)
    ff = gravity_feedforward(model)
    ids = {
        "x": model.actuator("act_x").id,
        "y": model.actuator("act_y").id,
        "z": model.actuator("act_z").id,
        "ff": model.actuator("act_z_ff").id,
        "gl": model.actuator("grip_l").id,
        "gr": model.actuator("grip_r").id,
    }

    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.1, 0.1, 0.65]
    cam.distance = 3.2
    cam.elevation = -24
    cam.azimuth = 110
    vopt = mj.MjvOption()
    vopt.flags[mj.mjtVisFlag.mjVIS_TENDON] = True

    placer = ClickPlacer(model, data)
    log = open("models/cable_schedule_live.csv", "w")
    log.write("t,x,y,z,L0,L1,L2,L3\n")

    renderer = mj.Renderer(model, height=480, width=640)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.1, 0.1, 0.65]
        viewer.cam.distance = 3.2
        viewer.cam.elevation = -24
        viewer.cam.azimuth = 110

        waypoints, get_target, sim_time = None, None, 0.0
        running = False
        print("Listo: click para colocar, ESPACIO para recoger.", flush=True)
        while viewer.is_running():
            renderer.update_scene(data, camera=cam, scene_option=vopt)
            frame = renderer.render().copy()

            if placer.reset_requested:
                placer.reset_requested = False
                running = False
                reset_to_home(model, data)
                print("Reset a home.", flush=True)

            if placer.pending_pick and not running:
                placer.pending_pick = False
                out = det.detect(frame, robot_z=float(data.joint("drive_z").qpos[0]))
                if out["sock_xy"] is None or out["conf"]["sock"] < 0.5:
                    print(f"Sin detección fiable (conf={out['conf']['sock']:.2f}): me quedo en hover.",
                          flush=True)
                else:
                    cnn_xy = pol.predict(frame)
                    pick, src = fuse_pick(cnn_xy, out["sock_xy"], out["conf"]["sock"])
                    waypoints = build_waypoints(pick, (0.55, 0.45))
                    get_target = make_get_target(waypoints)
                    sim_time, running = 0.0, True
                    print(f"Recogiendo en ({pick[0]:+.2f},{pick[1]:+.2f})[{src}] "
                          f"(YOLO {out['sock_xy']}, CNN {cnn_xy})", flush=True)

            if running:
                for _ in range(10):
                    pos, grip = get_target(sim_time)
                    data.ctrl[ids["x"]] = pos[0]
                    data.ctrl[ids["y"]] = pos[1]
                    data.ctrl[ids["z"]] = pos[2]
                    data.ctrl[ids["ff"]] = ff
                    data.ctrl[ids["gl"]] = grip
                    data.ctrl[ids["gr"]] = grip
                    mj.mj_step(model, data)
                    sim_time += model.opt.timestep
                    L = cable_lengths(pos)
                    log.write(f"{sim_time:.3f},{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f},"
                              f"{L[0]:.4f},{L[1]:.4f},{L[2]:.4f},{L[3]:.4f}\n")
                if sim_time >= waypoints[-1][0]:
                    running = False
                    print("Recogida completada. Click para otro objeto o Q para salir.",
                          flush=True)
            else:
                # IDLE: mantener home con feedforward.
                data.ctrl[ids["x"]] = 0.0
                data.ctrl[ids["y"]] = 0.0
                data.ctrl[ids["z"]] = 1.35
                data.ctrl[ids["ff"]] = ff
                for _ in range(10):
                    mj.mj_step(model, data)

            # Overlay de detecciones en la ventana de cámara.
            out = det.detect(frame, robot_z=float(data.joint("drive_z").qpos[0]))
            import cv2

            vis = frame.copy()
            for b in out["boxes"]:
                x0, y0, x1, y1 = map(int, b["xyxy"])
                color = (0, 255, 0) if b["cls"] == "robot" else (255, 0, 0)
                cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
                cv2.putText(vis, f"{b['cls']} {b['conf']:.2f}", (x0, y0 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if placer.show(vis):
                break
            viewer.sync()
            time.sleep(0.005)

    log.close()
    print("Cerrado. Log de cables: models/cable_schedule_live.csv", flush=True)


if __name__ == "__main__":
    main()

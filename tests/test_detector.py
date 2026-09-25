"""Test end-to-end del detector: render con pose conocida -> <5 cm."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")


def test_detector_known_pose():
    import mujoco as mj
    import numpy as np

    from control.model_data import build_model_data
    from perception.detector import Detector

    model, data = build_model_data()
    sj = model.joint("sock_joint").id
    qa = model.jnt_qposadr[sj]
    data.qpos[qa : qa + 3] = [-0.30, 0.15, 0.026]
    data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
    mj.mj_forward(model, data)

    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.1, 0.1, 0.65]
    cam.distance = 3.2
    cam.elevation = -24
    cam.azimuth = 110
    vopt = mj.MjvOption()
    with mj.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=cam, scene_option=vopt)
        frame = renderer.render().copy()

    det = Detector()
    out = det.detect(frame, robot_z=1.35)
    assert out["sock_xy"] is not None, "no detectó el calcetín"
    assert out["robot_xy"] is not None, "no detectó el cubo"
    assert out["conf"]["sock"] > 0.5
    err = np.linalg.norm(out["sock_xy"] - np.array([-0.30, 0.15]))
    assert err < 0.05, f"error calcetín {err:.3f} m"
    err_r = np.linalg.norm(out["robot_xy"] - np.array([0.0, 0.0]))
    assert err_r < 0.10, f"error cubo {err_r:.3f} m"

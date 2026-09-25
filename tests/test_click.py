"""place_sock_at_pixel: válido coloca, cielo/fuera de alcance rechaza."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")


def test_place_valid_pixel():
    import mujoco as mj
    import numpy as np

    from control.model_data import build_model_data
    from perception.camera import project
    from sim.click_place import place_sock_at_pixel

    model, data = build_model_data()
    # Píxel que mira al punto (-0.5, 0.3) del suelo.
    u, v = project(np.array([-0.5, 0.3, 0.0]))
    placed = place_sock_at_pixel(model, data, u, v)
    assert placed is not None
    assert np.linalg.norm(placed - np.array([-0.5, 0.3])) < 0.02
    sock = data.xpos[model.body("sock_target").id]
    assert abs(sock[0] + 0.5) < 0.02 and abs(sock[1] - 0.3) < 0.02


def test_place_sky_rejected():
    from control.model_data import build_model_data
    from sim.click_place import place_sock_at_pixel

    model, data = build_model_data()
    assert place_sock_at_pixel(model, data, 320, 0) is None

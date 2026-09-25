"""Test IK analítica del CDPR (falla hasta crear control/cable_ik.py)."""

import numpy as np


def test_ik_roundtrip():
    from control.cable_ik import XYZ_HOME, cable_lengths

    L = cable_lengths(XYZ_HOME)
    assert L.shape == (4,)
    assert np.all(L > 0.5) and np.all(L < 4.0)


def test_ik_matches_simulation():
    """Las longitudes analíticas deben coincidir con las distancias reales
    ancla<->gancho medidas en la simulación (<1 mm)."""
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco as mj

    from control.cable_ik import ANCHORS, HOOKS, cable_lengths
    from control.model_data import build_model_data

    model, data = build_model_data()
    mj.mj_forward(model, data)
    carrier_xyz = data.xpos[model.body("carrier").id].copy()
    ana = cable_lengths(carrier_xyz)
    for i in range(4):
        anchor = data.site(f"anchor{i}").xpos
        hook = data.site(f"hook{i}").xpos
        real = np.linalg.norm(anchor - hook)
        assert abs(ana[i] - real) < 1e-3, f"cable{i}: {ana[i]} vs {real}"

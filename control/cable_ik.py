"""IK analítica del CDPR: longitudes de cable <-> posición cartesiana.

Hipótesis (documentada para el mundo real): cables rectos e inextensibles.
L_i = |ancla_i - (xyz_carrier + offset_gancho_i)|, con anclas y ganchos del
XML original (sites anchor0..3, hook0..3; offsets en el frame del carrier).
"""

import numpy as np

ANCHORS = np.array(
    [
        [-1.35, -1.35, 2.3],
        [1.35, -1.35, 2.3],
        [1.35, 1.35, 2.3],
        [-1.35, 1.35, 2.3],
    ]
)
HOOKS = np.array(
    [
        [-0.065, -0.065, 0.065],
        [0.065, -0.065, 0.065],
        [0.065, 0.065, 0.065],
        [-0.065, 0.065, 0.065],
    ]
)
XYZ_HOME = np.array([0.0, 0.0, 1.35])
REACH_XY = 1.1  # rango drive_x/drive_y del XML


def cable_lengths(xyz) -> np.ndarray:
    """Longitudes de los 4 cables para el carrier en xyz (m)."""
    xyz = np.asarray(xyz, dtype=float)
    return np.linalg.norm(ANCHORS - (xyz + HOOKS), axis=1)


def clamp_reachable(xyz):
    """Proyecta un target al volumen alcanzable del robot."""
    xyz = np.asarray(xyz, dtype=float).copy()
    xyz[0] = float(np.clip(xyz[0], -REACH_XY, REACH_XY))
    xyz[1] = float(np.clip(xyz[1], -REACH_XY, REACH_XY))
    xyz[2] = float(np.clip(xyz[2], 0.5, 2.1))
    return xyz


def cable_slack(lengths, xyz, tol: float = 1e-3) -> np.ndarray:
    """True por cable si la longitud ordenada excede la distancia recta
    (cable flojo: la hipótesis del modelo no aplica)."""
    straight = cable_lengths(xyz)
    return np.asarray(lengths, dtype=float) > straight + tol

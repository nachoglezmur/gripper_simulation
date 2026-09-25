"""Fusión CNN + YOLO para el punto de agarre.

La CNN propone desde la imagen; YOLO+deproyección verifica con geometría.
Si discrepan >15 cm se confía en YOLO (medición geométrica directa).
"""

import numpy as np

from control.cable_ik import clamp_reachable

FUSE_TOL_M = 0.15


def fuse_pick(cnn_xy, yolo_xy, yolo_conf, min_conf=0.5):
    """Devuelve (pick_xyz[X,Y], fuente). z la pone el llamador (Z_PICK)."""
    cnn_xy = np.asarray(cnn_xy, dtype=float)[:2]
    if yolo_xy is not None and yolo_conf >= min_conf:
        yolo_xy = np.asarray(yolo_xy, dtype=float)[:2]
        if np.linalg.norm(cnn_xy - yolo_xy) > FUSE_TOL_M:
            return clamp_reachable(np.append(yolo_xy, 0.0))[:2], "yolo"
    return clamp_reachable(np.append(cnn_xy, 0.0))[:2], "cnn"

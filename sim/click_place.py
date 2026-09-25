"""Click-to-place: click en la imagen 2D -> teleport del calcetín al suelo.

La ventana es la imagen de la cámara (OpenCV), no el visor 3D: así el robot
solo usa información de imagen, como una cámara real. La parte testeable
(place_sock_at_pixel) no necesita GUI.
"""

import numpy as np

from control.cable_ik import REACH_XY
from perception.camera import ray_to_floor

SOCK_Z = 0.026


def place_sock_at_pixel(model, data, u, v):
    """Teleporta el calcetín al punto de suelo bajo el píxel.

    Devuelve (x, y) si el click es válido y alcanzable, None si se rechaza.
    """
    import mujoco as mj

    xy = ray_to_floor((float(u), float(v)))
    if xy is None:
        return None
    if abs(xy[0]) > REACH_XY or abs(xy[1]) > REACH_XY:
        return None
    sj = model.joint("sock_joint").id
    qa = model.jnt_qposadr[sj]
    data.qpos[qa : qa + 3] = [float(xy[0]), float(xy[1]), SOCK_Z]
    data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
    data.qvel[model.jnt_dofadr[sj] : model.jnt_dofadr[sj] + 6] = 0
    mj.mj_forward(model, data)
    return np.array([xy[0], xy[1]])


class ClickPlacer:
    """Ventana OpenCV con la imagen de cámara + callback de ratón."""

    def __init__(self, model, data, window="camara (click: colocar | ESPACIO: recoger | R: reset | Q: salir)"):
        import cv2

        self.model = model
        self.data = data
        self.window = window
        self.last_frame = None
        self.pending_pick = False
        self.reset_requested = False
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        import cv2

        if event == cv2.EVENT_LBUTTONDOWN:
            placed = place_sock_at_pixel(self.model, self.data, x, y)
            if placed is None:
                print("click fuera de alcance: ignorado", flush=True)
            else:
                print(f"calcetín colocado en ({placed[0]:.2f}, {placed[1]:.2f})", flush=True)

    def show(self, frame_rgb):
        import cv2

        self.last_frame = frame_rgb
        cv2.imshow(self.window, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            self.pending_pick = True
        elif key in (ord("r"), ord("R")):
            self.reset_requested = True
        return key in (ord("q"), ord("Q"), 27)

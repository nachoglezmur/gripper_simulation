"""Geometría exacta píxel <-> mundo para la cámara fija del vídeo original.

Usa el frustum real que MuJoCo calcula (scn.camera[0]: pos/forward/up +
frustum_center/width/bottom/top/near), así que project() coincide con lo
que mj.Renderer dibuja, sin suponer ningún FOV.

Cámara fija (idéntica al vídeo):
  lookat=[0.1,0.1,0.65], distance=3.2, elevation=-24, azimuth=110, 640x480.
"""

import numpy as np

WIDTH = 640
HEIGHT = 480
LOOKAT = (0.1, 0.1, 0.65)
DISTANCE = 3.2
ELEVATION = -24.0
AZIMUTH = 110.0
# Región válida de suelo (un poco más que el alcance del robot).
FLOOR_BOUNDS = 1.3

_rig_cache = {}


class CameraRig:
    """Frustum + pose de una cámara libre MuJoCo concreta."""

    def __init__(self, model, data, lookat=LOOKAT, distance=DISTANCE,
                 elevation=ELEVATION, azimuth=AZIMUTH,
                 width=WIDTH, height=HEIGHT):
        import mujoco as mj

        self.width = width
        self.height = height
        cam = mj.MjvCamera()
        mj.mjv_defaultCamera(cam)
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = list(lookat)
        cam.distance = distance
        cam.elevation = elevation
        cam.azimuth = azimuth
        vopt = mj.MjvOption()
        pert = mj.MjvPerturb()
        scn = mj.MjvScene(model, maxgeom=10000)
        mj.mjv_updateScene(model, data, vopt, pert, cam,
                           mj.mjtCatBit.mjCAT_ALL, scn)
        glcam = scn.camera[0]
        self.pos = np.array(glcam.pos, dtype=float)
        self.fwd = np.array(glcam.forward, dtype=float)
        self.up = np.array(glcam.up, dtype=float)
        self.right = np.cross(self.fwd, self.up)
        self.right /= np.linalg.norm(self.right)
        # mjv_updateScene deja frustum_width=0: el renderer lo deduce del
        # aspect del viewport (altura * W/H). Replicamos exactamente eso.
        self.fc = float(glcam.frustum_center)
        self.fb = float(glcam.frustum_bottom)
        self.ft = float(glcam.frustum_top)
        self.fw = (self.ft - self.fb) * (width / height)
        self.near = float(glcam.frustum_near)

    def project(self, world):
        """Mundo (x,y,z) -> píxel (u,v) con origen arriba-izquierda."""
        d = np.asarray(world, dtype=float) - self.pos
        zc = float(np.dot(d, self.fwd))
        xc = float(np.dot(d, self.right))
        yc = float(np.dot(d, self.up))
        xn = xc / zc * self.near
        yn = yc / zc * self.near
        u = (xn - (self.fc - self.fw / 2.0)) / self.fw * self.width
        v_gl = (yn - self.fb) / (self.ft - self.fb) * self.height
        return float(u), float(self.height - v_gl)

    def ray_to_floor(self, pixel):
        """Píxel (u,v) -> (x,y) de intersección con z=0, o None."""
        u, v = pixel
        xn = (u / self.width) * self.fw + (self.fc - self.fw / 2.0)
        yn = ((self.height - v) / self.height) * (self.ft - self.fb) + self.fb
        direction = xn * self.right + yn * self.up + self.near * self.fwd
        direction = direction / np.linalg.norm(direction)
        if direction[2] >= -1e-9:
            return None
        t = -self.pos[2] / direction[2]
        hit = self.pos + t * direction
        if abs(hit[0]) > FLOOR_BOUNDS or abs(hit[1]) > FLOOR_BOUNDS:
            return None
        return np.array([hit[0], hit[1]])

    def ray(self, pixel):
        """Píxel (u,v) -> (origen, dirección unitaria) en mundo."""
        u, v = pixel
        xn = (u / self.width) * self.fw + (self.fc - self.fw / 2.0)
        yn = ((self.height - v) / self.height) * (self.ft - self.fb) + self.fb
        direction = xn * self.right + yn * self.up + self.near * self.fwd
        return self.pos.copy(), direction / np.linalg.norm(direction)

    def unproject(self, pixel, depth_fwd):
        """Píxel + profundidad (componente forward) -> punto 3D mundo."""
        pos, direction = self.ray(pixel)
        s = depth_fwd / float(np.dot(direction, self.fwd))
        return pos + s * direction

    def depth_from_apparent_height(self, h_px, real_h):
        """Profundidad forward por tamaño aparente (objeto de alto conocido)."""
        return real_h * self.height * self.near / (h_px * (self.ft - self.fb))


def _default_rig():
    key = "default"
    if key not in _rig_cache:
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from control.model_data import build_model_data

        model, data = build_model_data()
        _rig_cache[key] = CameraRig(model, data)
    return _rig_cache[key]


def default_rig():
    """Rig de la cámara fija (para profundidad por tamaño, etc.)."""
    return _default_rig()


def project(world):
    """Mundo (x,y,z) -> píxel (u,v). Cámara fija del vídeo."""
    return _default_rig().project(world)


def ray_to_floor(pixel):
    """Píxel (u,v) -> (x,y) en z=0, o None si no hay suelo alcanzable."""
    return _default_rig().ray_to_floor(pixel)

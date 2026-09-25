"""Trayectoria experta del vídeo original, parametrizada y reutilizable.

Mismos waypoints/ganancias que cable_robot_gui.py, pero como funciones
puras sobre (model, data): la usan las demos, la app neuronal y la eval.
"""

import numpy as np

Z_HOVER = 1.35
Z_PICK = 0.665
Z_BASKET = 0.98
GRIP_OPEN = 0.0
GRIP_CLOSED = 0.42
GRIP_DROP = -0.20


def build_waypoints(sock_xy, basket_xy):
    sx, sy = float(sock_xy[0]), float(sock_xy[1])
    bx, by = float(basket_xy[0]), float(basket_xy[1])
    return [
        (0.0, np.array([0.0, 0.0, Z_HOVER]), 0.0, "1. Inicio en reposo"),
        (2.0, np.array([sx, sy, Z_HOVER]), 0.0, "2. Volando sobre el calcetín"),
        (4.0, np.array([sx, sy, Z_PICK]), 0.0, "3. Descendiendo con pinza abierta"),
        (5.4, np.array([sx, sy, Z_PICK]), 0.42, "4. Cerrando pinza (atrapando calcetín)"),
        (7.0, np.array([sx, sy, Z_HOVER]), 0.42, "5. Elevando el calcetín"),
        (10.0, np.array([bx, by, Z_HOVER]), 0.42, "6. Transportando hacia la cesta"),
        (11.6, np.array([bx, by, Z_BASKET]), 0.42, "7. Descendiendo sobre la cesta"),
        (12.6, np.array([bx, by, Z_BASKET]), -0.20, "8. Apertura amplia: soltando"),
        (14.2, np.array([bx, by, Z_BASKET]), -0.20, "8b. El calcetín cae en la cesta"),
        (16.5, np.array([0.0, 0.0, Z_HOVER]), 0.0, "9. Retorno a reposo"),
    ]


def make_get_target(waypoints):
    def get_target(t):
        if t <= waypoints[0][0]:
            return waypoints[0][1], waypoints[0][2]
        if t >= waypoints[-1][0]:
            return waypoints[-1][1], waypoints[-1][2]
        for i in range(len(waypoints) - 1):
            t0, p0, g0 = waypoints[i][0], waypoints[i][1], waypoints[i][2]
            t1, p1, g1 = waypoints[i + 1][0], waypoints[i + 1][1], waypoints[i + 1][2]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0)
                s = 0.5 * (1.0 - np.cos(np.pi * alpha))
                return p0 + s * (p1 - p0), g0 + s * (g1 - g0)

    return get_target


def run_expert(model, data, sock_xy, basket_xy=(0.55, 0.45), log=False,
               target_xy=None):
    """Ejecuta la trayectoria completa. Devuelve (success, pick_xyz).

    success: el calcetín termina a <0.25 m del centro de la cesta.
    target_xy: a dónde apuntan los waypoints (por defecto, la pose real del
      calcetín). Para evaluar la red se pasa el target APRENDIDO mientras el
      calcetín sigue en su pose real.
    pick_xyz: punto de agarre usado (target + Z_PICK) para behavior cloning.
    """
    import mujoco as mj

    from control.model_data import gravity_feedforward, reset_to_home

    reset_to_home(model, data)
    sj = model.joint("sock_joint").id
    qa = model.jnt_qposadr[sj]
    data.qpos[qa : qa + 3] = [float(sock_xy[0]), float(sock_xy[1]), 0.026]
    data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
    mj.mj_forward(model, data)

    ids = {
        "x": model.actuator("act_x").id,
        "y": model.actuator("act_y").id,
        "z": model.actuator("act_z").id,
        "ff": model.actuator("act_z_ff").id,
        "gl": model.actuator("grip_l").id,
        "gr": model.actuator("grip_r").id,
    }
    ff = gravity_feedforward(model)
    tgt = target_xy if target_xy is not None else sock_xy
    waypoints = build_waypoints(tgt, basket_xy)
    get_target = make_get_target(waypoints)
    dt = model.opt.timestep
    total = waypoints[-1][0]
    steps = int(total / dt)
    active = -1
    for step in range(steps):
        t = step * dt
        pos, grip = get_target(t)
        data.ctrl[ids["x"]] = pos[0]
        data.ctrl[ids["y"]] = pos[1]
        data.ctrl[ids["z"]] = pos[2]
        data.ctrl[ids["ff"]] = ff
        data.ctrl[ids["gl"]] = grip
        data.ctrl[ids["gr"]] = grip
        mj.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            if log:
                print(f"NaN en t={t:.2f}", flush=True)
            return False, np.array([sock_xy[0], sock_xy[1], Z_PICK])
        if log:
            idx = sum(1 for wp in waypoints if t >= wp[0]) - 1
            if idx != active:
                active = idx
                print(f"t={t:5.2f} -> {waypoints[active][3]}", flush=True)

    sock = data.xpos[model.body("sock_target").id][:2]
    basket = np.array(basket_xy)
    success = bool(np.linalg.norm(sock - basket) < 0.25)
    return success, np.array([tgt[0], tgt[1], Z_PICK])

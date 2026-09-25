# ==============================================================================
# ROBOT DE PINZA SUSPENDIDA POR 4 CABLES (CDPR TIDYING ROBOT)
# Versión LOCAL WSL + GPU (adaptado desde Colab)
# - Render EGL headless con GPU (WSLg / Mesa)
# - Guarda vídeo en disco en lugar de media.show_video()
# - Verificación JAX GPU + MJX
# ==============================================================================
import os
# MUJOCO_GL debe fijarse antes de importar mujoco
os.environ.setdefault('MUJOCO_GL', 'egl')
# imageio-ffmpeg trae su propio binario ffmpeg (sin necesidad de apt)
try:
    import imageio_ffmpeg
    os.environ.setdefault('IMAGEIO_FFMPEG_EXE', imageio_ffmpeg.get_ffmpeg_exe())
except Exception:
    pass

import shutil
import subprocess
import io
import time
import numpy as np
from PIL import Image
import mediapy as media
import mujoco as mj

print("⚙️ Verificando GPU y entorno local...")
# 1. nvidia-smi (si existe)
if shutil.which('nvidia-smi'):
    try:
        subprocess.run(['nvidia-smi', '-L'], check=False)
    except Exception as e:
        print(f"aviso nvidia-smi: {e}")
else:
    print("nvidia-smi no encontrado en PATH (ok en algunos WSL, seguimos con JAX).")

# 2. JAX + GPU
import jax
print(f"JAX versión: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
gpu_available = any(d.platform == 'gpu' or 'cuda' in str(d).lower() or 'CudaDevice' in str(type(d)) for d in jax.devices())
# Chequeo robusto: nombre del device
dev_str = str(jax.devices())
if 'cuda' in dev_str.lower() or 'gpu' in dev_str.lower():
    gpu_available = True
print(f"GPU JAX detectada: {gpu_available}")
if not gpu_available:
    print("⚠️ JAX no ve GPU, pero seguimos (MuJoCo EGL puede usar Mesa/WSLg).")

# 3. MJX
try:
    from mujoco import mjx
    print(f"✅ MJX disponible: {mjx.__file__}")
    print(f"MuJoCo versión: {mj.__version__}")
except Exception as e:
    print(f"⚠️ MJX no disponible: {e}")

print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}")

# 2. Generación procedural de texturas (Parqué de roble mate y AprilTag)
print("🎨 Generando texturas de alta resolución...")
np.random.seed(0)

# A. Suelo de parqué cálido mate (sin reflejos ni quemados)
def generate_wood_floor_png():
    res = 512
    img = np.zeros((res, res, 3), dtype=np.uint8)
    plank_h = 64
    for y in range(res):
        plank_idx = y // plank_h
        base_color = np.array([170, 138, 102], dtype=np.float32) if plank_idx % 2 == 0 else np.array([185, 150, 112], dtype=np.float32)
        grain = np.sin(y * 0.4) * 5 + np.sin(y * 0.08) * 10
        for x in range(res):
            noise = (np.sin(x * 0.15 + y * 0.05) * 4) + (np.random.rand() - 0.5) * 5
            color = base_color + grain + noise
            if y % plank_h in [0, plank_h - 1] or (x + (plank_idx * 128)) % 256 in [0, 1]:
                color *= 0.5
            img[y, x] = np.clip(color, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='PNG')
    return buf.getvalue()

# B. Marcador AprilTag nítido (36h11)
def generate_apriltag_png():
    grid = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    full_grid = np.ones((10, 10), dtype=np.uint8)
    full_grid[1:9, 1:9] = grid
    img_data = np.repeat(np.repeat(full_grid * 255, 25, axis=0), 25, axis=1)
    img_rgb = np.stack([img_data]*3, axis=-1)
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format='PNG')
    return buf.getvalue()

assets = {
    'wood_floor.png': generate_wood_floor_png(),
    'apriltag.png': generate_apriltag_png()
}

# 3. Modelo MJCF XML
xml_model = """
<mujoco model="cable_robot_perfect">
  <compiler angle="radian" autolimits="true"/>

  <option timestep="0.002" iterations="80" tolerance="1e-8"
          integrator="implicitfast" cone="elliptic"/>

  <visual>
    <quality shadowsize="4096"/>
    <!-- Headlight suave con specular=0: garantiza visibilidad total sin quemar nada -->
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <map force="0.1" zfar="30"/>
  </visual>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.88 0.92 0.96" rgb2="0.65 0.70 0.76" width="512" height="512"/>
    <texture name="wood_tex" type="2d" file="wood_floor.png"/>
    <material name="wood_mat" texture="wood_tex" texrepeat="5 5" reflectance="0.0" specular="0.0" shininess="0.0"/>
    <texture name="tag_tex" type="2d" file="apriltag.png"/>
    <material name="tag_mat" texture="tag_tex" specular="0.2" shininess="0.1"/>
  </asset>

  <worldbody>
    <!-- Iluminación difusa con specular=0 para evitar quemados solares -->
    <light pos="0 0 3.0" dir="0 0 -1" diffuse="0.55 0.55 0.55" specular="0 0 0" castshadow="true"/>
    <light pos="1.5 -1.5 2.5" dir="-1 1 -1.5" diffuse="0.25 0.25 0.25" specular="0 0 0"/>

    <!-- Suelo de parqué mate -->
    <geom name="floor" type="plane" size="2.5 2.5 0.05" material="wood_mat" friction="1.0 0.02 0.001"/>

    <!-- 4 Postes esquineros exteriores delgados y semitransparentes (nunca tapan la cámara) -->
    <geom type="cylinder" fromto="-1.35 -1.35 0 -1.35 -1.35 2.3" size="0.01" rgba="0.4 0.4 0.45 0.35"/>
    <site name="anchor0" pos="-1.35 -1.35 2.3"/>

    <geom type="cylinder" fromto="1.35 -1.35 0 1.35 -1.35 2.3" size="0.01" rgba="0.4 0.4 0.45 0.35"/>
    <site name="anchor1" pos="1.35 -1.35 2.3"/>

    <geom type="cylinder" fromto="1.35 1.35 0 1.35 1.35 2.3" size="0.01" rgba="0.4 0.4 0.45 0.35"/>
    <site name="anchor2" pos="1.35 1.35 2.3"/>

    <geom type="cylinder" fromto="-1.35 1.35 0 -1.35 1.35 2.3" size="0.01" rgba="0.4 0.4 0.45 0.35"/>
    <site name="anchor3" pos="-1.35 1.35 2.3"/>

    <!-- Cesta de ropa (Laundry Hamper) -->
    <body name="basket" pos="0.55 0.45 0">
      <geom type="box" size="0.16 0.16 0.005" pos="0 0 0.005" rgba="0.96 0.96 0.96 1" friction="1.0 0.05 0.001"/>
      <geom type="box" size="0.16 0.006 0.09" pos="0 0.16 0.09" rgba="0.92 0.93 0.95 0.95"/>
      <geom type="box" size="0.16 0.006 0.09" pos="0 -0.16 0.09" rgba="0.92 0.93 0.95 0.95"/>
      <geom type="box" size="0.006 0.16 0.09" pos="0.16 0 0.09" rgba="0.92 0.93 0.95 0.95"/>
      <geom type="box" size="0.006 0.16 0.09" pos="-0.16 0 0.09" rgba="0.92 0.93 0.95 0.95"/>
      <geom type="box" size="0.165 0.009 0.006" pos="0 0.16 0.18" rgba="0.05 0.5 0.85 1" contype="0" conaffinity="0"/>
      <geom type="box" size="0.165 0.009 0.006" pos="0 -0.16 0.18" rgba="0.05 0.5 0.85 1" contype="0" conaffinity="0"/>
      <geom type="box" size="0.009 0.165 0.006" pos="0.16 0 0.18" rgba="0.05 0.5 0.85 1" contype="0" conaffinity="0"/>
      <geom type="box" size="0.009 0.165 0.006" pos="-0.16 0 0.18" rgba="0.05 0.5 0.85 1" contype="0" conaffinity="0"/>
    </body>

    <!-- Calcetín objetivo en el suelo -->
    <body name="sock_target" pos="-0.30 0.15 0.026">
      <freejoint name="sock_joint"/>
      <!-- Fricción equilibrada (condim=3) para evitar adherencias -->
      <geom name="sock_core" type="capsule" fromto="0 -0.042 0 0 0.042 0" size="0.025"
            rgba="0.14 0.16 0.22 1" mass="0.06" condim="3" friction="1.2 0.01 0.001"/>
      <geom type="capsule" fromto="0 -0.025 0.012 0 0.025 0.012" size="0.018"
            rgba="0.2 0.22 0.28 1" mass="0.03" condim="3" friction="1.2 0.01 0.001"/>
      <geom type="sphere" pos="0 0.03 0.008" size="0.024"
            rgba="0.2 0.5 0.85 1" mass="0.02" condim="3" friction="1.2 0.01 0.001"/>
    </body>

    <!-- Prenda granate secundaria en el suelo -->
    <body name="cloth_red" pos="0.05 -0.3 0.025">
      <freejoint name="cloth_joint"/>
      <geom type="ellipsoid" size="0.08 0.06 0.02" rgba="0.45 0.1 0.16 1" mass="0.12" friction="1.0 0.05 0.001"/>
    </body>

    <!-- ================= ROBOT SUSPENDIDO ================= -->
    <body name="carrier_x" pos="0 0 0">
      <inertial pos="0 0 0" mass="0.5" diaginertia="0.01 0.01 0.01"/>
      <joint name="drive_x" type="slide" axis="1 0 0" range="-1.1 1.1" damping="6" armature="0.5"/>
      <body name="carrier_y" pos="0 0 0">
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.01 0.01 0.01"/>
        <joint name="drive_y" type="slide" axis="0 1 0" range="-1.1 1.1" damping="6" armature="0.5"/>
        <body name="carrier" pos="0 0 0">
          <joint name="drive_z" type="slide" axis="0 0 1" range="0.5 2.1" damping="6" armature="0.5"/>

          <!-- Cubo de tracking AprilTag -->
          <geom name="box_body" type="box" size="0.065 0.065 0.065" pos="0 0 0" rgba="0.97 0.97 0.98 1" mass="0.4"/>
          <geom type="box" size="0.001 0.05 0.05" pos="0.066 0 0" material="tag_mat" contype="0" conaffinity="0"/>
          <geom type="box" size="0.001 0.05 0.05" pos="-0.066 0 0" material="tag_mat" contype="0" conaffinity="0"/>
          <geom type="box" size="0.05 0.001 0.05" pos="0 0.066 0" material="tag_mat" contype="0" conaffinity="0"/>
          <geom type="box" size="0.05 0.001 0.05" pos="0 -0.066 0" material="tag_mat" contype="0" conaffinity="0"/>
          <geom type="box" size="0.002 0.045 0.055" pos="0 0 0.11" material="tag_mat" contype="0" conaffinity="0"/>
          <geom type="box" size="0.015 0.008 0.012" pos="0 0 0.17" rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>
          <geom type="cylinder" fromto="0 0 0.165 0.005 0 0.165" size="0.003" rgba="0.9 0.1 0.1 1" contype="0" conaffinity="0"/>

          <!-- 4 Ganchos de los cables -->
          <site name="hook0" pos="-0.065 -0.065 0.065" size="0.005" rgba="0.8 0.1 0.1 1"/>
          <site name="hook1" pos=" 0.065 -0.065 0.065" size="0.005" rgba="0.1 0.8 0.1 1"/>
          <site name="hook2" pos=" 0.065  0.065 0.065" size="0.005" rgba="0.1 0.1 0.8 1"/>
          <site name="hook3" pos="-0.065  0.065 0.065" size="0.005" rgba="0.8 0.8 0.1 1"/>

          <!-- Mástil vertical en fibra de carbono -->
          <geom name="mast" type="cylinder" fromto="0 0 -0.065 0 0 -0.52" size="0.007"
                rgba="0.15 0.15 0.16 1" mass="0.1" contype="0" conaffinity="0"/>

          <!-- Cabezal de la pinza -->
          <body name="gripper_chassis" pos="0 0 -0.54">
            <geom name="chassis" type="box" size="0.045 0.048 0.042" pos="0 0 0" rgba="0.97 0.97 0.98 1" mass="0.35"/>
            <geom name="chassis_top" type="cylinder" size="0.038 0.012" pos="0 0 0.042" rgba="0.9 0.9 0.92 1" mass="0.05"/>
            <!-- Ribete azul estético -->
            <geom name="blue_band" type="box" size="0.046 0.049 0.003" pos="0 0 0.022" rgba="0.0 0.6 0.95 1" contype="0" conaffinity="0"/>
            <geom name="grill" type="box" size="0.004 0.022 0.014" pos="0.046 0 -0.015" rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>

            <!-- DEDO IZQUIERDO: Rango [-0.25, 0.55] con punta redondeada y biselada -->
            <body name="finger_left" pos="-0.045 0 -0.042">
              <joint name="finger_left_joint" type="hinge" axis="0 -1 0" range="-0.25 0.55" damping="0.4"/>
              <geom type="box" size="0.008 0.016 0.045" pos="0 0 -0.045" rgba="0.94 0.94 0.96 1" mass="0.04" contype="0" conaffinity="0"/>
              <!-- Almohadilla antideslizante -->
              <geom name="pad_left" type="box" size="0.005 0.022 0.036" pos="0.008 0 -0.045"
                    rgba="0.12 0.12 0.12 1" mass="0.01" condim="3" friction="1.2 0.01 0.001"/>
              <!-- Punta biselada suave que NUNCA retiene el calcetín -->
              <geom type="capsule" fromto="0.004 0 -0.068 0.008 0 -0.082" size="0.006" rgba="0.12 0.12 0.12 1"/>
            </body>

            <!-- DEDO DERECHO SIMÉTRICO -->
            <body name="finger_right" pos="0.045 0 -0.042">
              <joint name="finger_right_joint" type="hinge" axis="0 1 0" range="-0.25 0.55" damping="0.4"/>
              <geom type="box" size="0.008 0.016 0.045" pos="0 0 -0.045" rgba="0.94 0.94 0.96 1" mass="0.04" contype="0" conaffinity="0"/>
              <geom name="pad_right" type="box" size="0.005 0.022 0.036" pos="-0.008 0 -0.045"
                    rgba="0.12 0.12 0.12 1" mass="0.01" condim="3" friction="1.2 0.01 0.001"/>
              <geom type="capsule" fromto="-0.004 0 -0.068 -0.008 0 -0.082" size="0.006" rgba="0.12 0.12 0.12 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- Cables tensores dinámicos -->
  <tendon>
    <spatial name="cable0" width="0.0025" rgba="0.25 0.25 0.3 1"><site site="anchor0"/><site site="hook0"/></spatial>
    <spatial name="cable1" width="0.0025" rgba="0.25 0.25 0.3 1"><site site="anchor1"/><site site="hook1"/></spatial>
    <spatial name="cable2" width="0.0025" rgba="0.25 0.25 0.3 1"><site site="anchor2"/><site site="hook2"/></spatial>
    <spatial name="cable3" width="0.0025" rgba="0.25 0.25 0.3 1"><site site="anchor3"/><site site="hook3"/></spatial>
  </tendon>

  <!-- Actuadores: Positivo CIERRA firmemente; Negativo ABRE ampliamente -->
  <actuator>
    <position name="act_x" joint="drive_x" kp="400" kv="70" ctrlrange="-1.1 1.1" forcerange="-70 70"/>
    <position name="act_y" joint="drive_y" kp="400" kv="70" ctrlrange="-1.1 1.1" forcerange="-70 70"/>
    <position name="act_z" joint="drive_z" kp="400" kv="70" ctrlrange="0.5 2.1" forcerange="-90 90"/>
    <motor name="act_z_ff" joint="drive_z" gear="1" forcerange="-30 30"/>
    <position name="grip_l" joint="finger_left_joint"  kp="25" kv="3" ctrlrange="-0.25 0.5" forcerange="-10 10"/>
    <position name="grip_r" joint="finger_right_joint" kp="25" kv="3" ctrlrange="-0.25 0.5" forcerange="-10 10"/>
  </actuator>
</mujoco>
"""

# 4. Compilación e inicialización
print("🚀 Compilando modelo en MuJoCo...")
model = mj.MjModel.from_xml_string(xml_model, assets)
data = mj.MjData(model)

act_x_id  = model.actuator('act_x').id
act_y_id  = model.actuator('act_y').id
act_z_id  = model.actuator('act_z').id
ff_id     = model.actuator('act_z_ff').id
grip_l_id = model.actuator('grip_l').id
grip_r_id = model.actuator('grip_r').id

carrier_id = model.body('carrier').id
gravity_ff = model.body_subtreemass[carrier_id] * (-model.opt.gravity[2])

p_home = np.array([0.0, 0.0, 1.35])
data.joint('drive_x').qpos[0] = p_home[0]
data.joint('drive_y').qpos[0] = p_home[1]
data.joint('drive_z').qpos[0] = p_home[2]
data.ctrl[act_x_id] = p_home[0]
data.ctrl[act_y_id] = p_home[1]
data.ctrl[act_z_id] = p_home[2]
data.ctrl[ff_id]    = gravity_ff
data.ctrl[grip_l_id] = 0.0
data.ctrl[grip_r_id] = 0.0
mj.mj_forward(model, data)

# Demo MJX en GPU (JAX): coloca el modelo en el acelerador
try:
    from mujoco import mjx as _mjx
    mjx_model = _mjx.put_model(model)
    mjx_data = _mjx.put_data(model, data)
    print(f"✅ MJX put_model/put_data OK. Devices: {mjx_data.qpos.devices() if hasattr(mjx_data.qpos, 'devices') else jax.devices()}")
    # Un paso MJX jit-compilado como prueba GPU
    jit_step = jax.jit(_mjx.step)
    mjx_data = jit_step(mjx_model, mjx_data)
    mjx_data.qpos.block_until_ready()
    print("✅ MJX step en GPU OK.")
except Exception as e:
    print(f"⚠️ MJX demo falló (no crítico para el vídeo): {e}")

# 5. Coordenadas reales exactas del calcetín y la cesta
sock_body_id   = model.body('sock_target').id
basket_body_id = model.body('basket').id

real_sock_pos   = data.xpos[sock_body_id][:2].copy()
real_basket_pos = data.xpos[basket_body_id][:2].copy()
print(f"🎯 Calcetín ubicado en: X={real_sock_pos[0]:.2f}, Y={real_sock_pos[1]:.2f}")
print(f"🧺 Cesta ubicada en:    X={real_basket_pos[0]:.2f}, Y={real_basket_pos[1]:.2f}")

z_hover  = 1.35
z_pick   = 0.665   # A ras de suelo envolviendo el calcetín
z_basket = 0.98    # Altura sobre la cesta

# Trayectoria:
waypoints = [
    (0.0,  np.array([0.0, 0.0, z_hover]),                     0.0,   "1. Inicio en reposo"),
    (2.0,  np.array([real_sock_pos[0], real_sock_pos[1], z_hover]), 0.0,   "2. Volando sobre el calcetín"),
    (4.0,  np.array([real_sock_pos[0], real_sock_pos[1], z_pick]),  0.0,   "3. Descendiendo con pinza abierta"),
    (5.4,  np.array([real_sock_pos[0], real_sock_pos[1], z_pick]),  0.42,  "4. Cerrando pinza (atrapando calcetín)"),
    (7.0,  np.array([real_sock_pos[0], real_sock_pos[1], z_hover]), 0.42,  "5. Elevando el calcetín"),
    (10.0, np.array([real_basket_pos[0], real_basket_pos[1], z_hover]), 0.42,  "6. Transportando hacia la cesta"),
    (11.6, np.array([real_basket_pos[0], real_basket_pos[1], z_basket]), 0.42,  "7. Descendiendo sobre la cesta"),
    (12.6, np.array([real_basket_pos[0], real_basket_pos[1], z_basket]), -0.20, "8. Apertura amplia en abanico: soltando"),
    (14.2, np.array([real_basket_pos[0], real_basket_pos[1], z_basket]), -0.20, "8b. El calcetín cae dentro de la cesta"),
    (16.5, np.array([0.0, 0.0, z_hover]),                     0.0,   "9. Retorno a posición de reposo"),
]

def get_target(t):
    if t <= waypoints[0][0]:
        return waypoints[0][1], waypoints[0][2]
    if t >= waypoints[-1][0]:
        return waypoints[-1][1], waypoints[-1][2]
    for i in range(len(waypoints) - 1):
        t0, p0, g0 = waypoints[i][0], waypoints[i][1], waypoints[i][2]
        t1, p1, g1 = waypoints[i+1][0], waypoints[i+1][1], waypoints[i+1][2]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)
            s = 0.5 * (1.0 - np.cos(np.pi * alpha))
            return p0 + s * (p1 - p0), g0 + s * (g1 - g0)

# 6. Cámara con encuadre despejado y nítido
fps = 30
dt = model.opt.timestep
total_duration = waypoints[-1][0]
total_steps = int(total_duration / dt)
steps_per_frame = int(round(1.0 / (fps * dt)))

cam = mj.MjvCamera()
mj.mjv_defaultCamera(cam)
cam.lookat = [0.1, 0.1, 0.65]
cam.distance = 3.2
cam.elevation = -24
cam.azimuth = 110

vopt = mj.MjvOption()
vopt.flags[mj.mjtVisFlag.mjVIS_TENDON] = True

frames = []
print(f"🎬 Iniciando simulación física ({total_duration:.1f}s, {total_steps} steps)...")
start_time = time.time()
active_wp = 0

with mj.Renderer(model, height=480, width=640) as renderer:
    for step in range(total_steps):
        t = step * dt

        pos_target, grip_target = get_target(t)
        data.ctrl[act_x_id]  = pos_target[0]
        data.ctrl[act_y_id]  = pos_target[1]
        data.ctrl[act_z_id]  = pos_target[2]
        data.ctrl[ff_id]     = gravity_ff
        data.ctrl[grip_l_id] = grip_target
        data.ctrl[grip_r_id] = grip_target

        mj.mj_step(model, data)

        if not np.isfinite(data.qpos).all():
            print("⚠️ NaN detectado, deteniendo en step", step)
            break

        if step % steps_per_frame == 0:
            current_idx = 0
            for idx, wp in enumerate(waypoints):
                if t >= wp[0]: current_idx = idx
            if active_wp != current_idx:
                active_wp = current_idx
                print(f"⏱️ t = {t:5.2f}s -> {waypoints[active_wp][3]}")

            renderer.update_scene(data, camera=cam, scene_option=vopt)
            frames.append(renderer.render().copy())

print(f"✅ Simulación completada en {time.time() - start_time:.1f}s. Frames: {len(frames)}")

import pathlib
out_path = pathlib.Path(__file__).parent / "cable_robot.mp4"
print(f"🎥 Guardando vídeo en {out_path} ...")
# mediapy necesita ffmpeg del sistema; en WSL sin sudo usamos imageio-ffmpeg (binario pip)
try:
    media.write_video(str(out_path), frames, fps=fps)
except RuntimeError as e:
    print(f"mediapy sin ffmpeg del sistema ({e}), usando imageio-ffmpeg...")
    import imageio.v2 as imageio
    imageio.mimsave(str(out_path), frames, fps=fps, codec='libx264', quality=8)
print(f"✅ Vídeo guardado: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
print("Para verlo en Windows, abre el mp4 desde el Explorador. También puedes copiarlo fuera de WSL.")

# Test MJX en GPU (JAX CUDA) - modelo simple compatible con MJX
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import jax
import mujoco
from mujoco import mjx
import numpy as np

print(f"JAX: {jax.__version__}, devices: {jax.devices()}")

# Modelo simple: dos esferas (MJX soporta sphere-sphere, plane, etc.)
xml = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)
print(f"mjx qpos device: {mjx_data.qpos.devices()}")

jit_step = jax.jit(mjx.step)
mjx_data = jit_step(mjx_model, mjx_data)
# Simula 100 pasos
for _ in range(100):
    mjx_data = jit_step(mjx_model, mjx_data)
mjx_data.qpos.block_until_ready()
print(f"✅ MJX 100 steps en GPU OK. qpos final: {np.asarray(mjx_data.qpos)}")
print(f"GPU usada: {jax.devices()[0]}")

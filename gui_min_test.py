# Test minimo del visor GLFW en WSLg
import os
os.environ['MUJOCO_GL'] = 'glfw'
print("import mujoco...", flush=True)
import mujoco as mj
import mujoco.viewer
print("compilando modelo minimo...", flush=True)
xml = """<mujoco><worldbody><geom name="floor" type="plane" size="5 5 0.1"/><body name="ball" pos="0 0 1"><freejoint/><geom name="b" type="sphere" size="0.1"/></body></worldbody></mujoco>"""
model = mj.MjModel.from_xml_string(xml)
data = mj.MjData(model)
print("launch viewer...", flush=True)
mujoco.viewer.launch(model, data)
print("viewer cerrado.", flush=True)

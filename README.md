# gripper_simulation

Robot CDPR (pinza suspendida por 4 cables) en MuJoCo que **detecta por imagen con red neuronal, acepta click para colocar el objeto y lo recoge solo**, registrando cuánto debe tirar de cada cable. Listo para puente sim-to-real (ONNX + IK de cables).

## Cómo funciona

```
imagen RGB (cámara fija) → YOLOv8n (cubo + calcetín) → deproyección a suelo
   → CNN de grasp (x, y) → fusión (|cnn−yolo|>15 cm ⇒ yolo)
   → waypoints cartesianos → IK analítica a L0..L3 (longitudes de cable)
```

- **YOLOv8n** fine-tuneado con 3500 imágenes sintéticas auto-etiquetadas del propio simulador (mAP50 0.99): estima la posición del robot y del objeto **solo con la imagen**.
- **CNN ResNet18** (behavior cloning de 800 demos del controlador experto): propone el punto de agarre (error 1.5 cm en test).
- **Click**: clickas en la ventana de la cámara y el calcetín se coloca en ese punto del suelo (ray-casting); con ESPACIO el robot va a por él.
- **Mundo real**: modelos en `models/*.onnx`, tabla posición→cables en CSV, calibración documentada en `REAL_WORLD_INTERFACE.md`.

## Uso (WSL + GPU NVIDIA)

```bash
python -m venv --without-pip .venv && .venv/bin/python /tmp/get-pip.py
.venv/bin/pip install mujoco mediapy Pillow imageio imageio-ffmpeg mujoco-mjx "jax[cuda13]"
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install ultralytics opencv-python onnx onnxscript pytest

python -m pytest tests/                                   # 11 tests
python -m perception.dataset_gen --n-train 3000 --n-val 500
python -m perception.train_yolo --epochs 100 --batch 8    # mAP50 > 0.90
python -m control.expert_demos --n 400                    # x2 semillas
python -m control.policy_bc --epochs 200 --demos data/demos/demos.npz,data/demos/demos2.npz
python -m sim.eval_grasps --n 20                          # exige >=17/20
bash launch_nn_gui.sh                                     # app: click + ESPACIO
```

## Estructura

| Carpeta | Contenido |
|---|---|
| `control/` | `model_data.py` (modelo original), `cable_ik.py` (XYZ↔longitudes), `trajectory.py` (experto), `policy_bc.py` (CNN+ONNX), `decide.py` (fusión) |
| `perception/` | `camera.py` (píxel↔mundo exacto), `dataset_gen.py`, `train_yolo.py`, `detector.py` |
| `sim/` | `click_place.py`, `cdpr_nn_gui.py` (app), `eval_grasps.py` |
| `tests/` | 11 tests (entorno GPU, IK, cámara, detector, click, artefactos) |
| `cable_robot_gui.py` | Visor del modelo original fotorrealista (sin redes) |

## Limitación conocida

El bucle es **open-loop**: detecta una vez al pulsar ESPACIO y ejecuta la trayectoria de 16.5 s a ciegas. Si el objetivo **se mueve** (rueda), falla. Siguiente paso: re-detección periódica + replanificación (ver `REAL_WORLD_INTERFACE.md`).

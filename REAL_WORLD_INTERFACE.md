# Interfaz mundo real del CDPR neuronal

Cómo llevar lo aprendido en simulación al robot físico de 4 cables.

## 1. Qué exporta este proyecto

| Artefacto | Fichero | Uso en el robot real |
|---|---|---|
| Detector YOLOv8n (cubo + calcetín) | `models/yolo_cdpr.onnx` (12 MB) | Entrada: imagen RGB 640×480 de la cámara cenital. Salida: cajas + confianzas. Runtime: `onnxruntime-gpu` (el `onnxruntime` de pip es solo CPU). |
| CNN punto de agarre | `models/grasp_cnn.onnx` + `models/grasp_cnn.onnx.data` (45 MB, viajan juntos) + `models/grasp_norm.npz` (media/desv para desnormalizar) | Entrada: imagen RGB 480×640 normalizada ImageNet. Salida: `(x, y)` en metros (desnormalizar con `grasp_norm.npz`, `z = 0.665` fijo). |
| Tabla posición → cables | `models/cable_schedule_example.csv`, `models/cable_schedule_live.csv` | Columnas `t,x,y,z,L0..L3`: longitudes (m) que cada cable debe tener en cada instante. |

## 2. Convención de cables (la "respuesta" a cuánto tirar)

- Anclas (esquinas, mundo): `A0=(-1.35,-1.35,2.3)`, `A1=(+1.35,-1.35,2.3)`, `A2=(+1.35,+1.35,2.3)`, `A3=(-1.35,+1.35,2.3)` m.
- Ganchos (offsets en el frame del carrier): `H0=(-.065,-.065,+.065)`, `H1=(+.065,-.065,+.065)`, `H2=(+.065,+.065,+.065)`, `H3=(-.065,+.065,+.065)` m.
- Longitud: `L_i = |A_i − (XYZ + H_i)|` (cables rectos e inextensibles; ver `control/cable_ik.py`).
- Calibración en el robot real: medir las 4 `A_i` con cinta/láser desde el origen suelo e introducirlas en `ANCHORS`; medir el cero de cada enrollador con el carrier en `(0,0,1.35)` y comparar con `XYZ_HOME` (`control/cable_ik.py`).
- Si un cable ordenado supera la distancia recta (`cable_slack()`), hay flojedad: el modelo no aplica; tensar antes de continuar.

## 3. Calibración de cámara (imprescindible)

La red se entrenó con UNA cámara fija (`lookat [0.1,0.1,0.65]`, dist 3.2 m,
elev −24°, azim 110°, 640×480). En el mundo real:

1. Colocar la cámara cenital en la misma pose aproximada.
2. Imprimir un patrón o usar el cubo AprilTag para estimar la homografía
   imagen↔suelo y ajustar `perception/camera.py` si cambia la óptica/pose.
3. Revalidar `mAP50` con ~50 fotos reales etiquetadas; si < 0.8, hacer
   fine-tune mixto (fotos reales + sintéticas).

## 4. Bucle de control en el robot físico

```
imagen RGB → YOLO (cajas) → deproyección a suelo (cámara calibrada)
        → CNN grasp (x, y) → fusión (|cnn−yolo|>15 cm ⇒ yolo)
        → waypoints cartesianos → IK a L0..L3 → enrolladores (posición)
```

## 5. Limitaciones conocidas

- `onnxruntime` de pip es CPU; para GPU instalar `onnxruntime-gpu`.
- Hipótesis de cables rectos: sin modelo de catenaria ni elasticidad.
- La CNN solo propone `(x, y)`; `z` de picado es fijo (0.665 m).
- Sin detección fiable (conf < 0.5) el sistema NO actúa (seguridad).

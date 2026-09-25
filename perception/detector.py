"""Detector runtime: YOLOv8n fine-tuneado -> posiciones en el suelo.

Entrada: imagen RGB 640x480 (convención del renderer MuJoCo).
Salida: dict con robot_xy / sock_xy (np.ndarray[2] en z=0, o None si
confianza < umbral) + confianzas y cajas por clase.
"""

import numpy as np

from perception.camera import default_rig, ray_to_floor

CLASS_NAMES = {0: "cubo_robot", 1: "calcetin"}
CUBE_HEIGHT_M = 0.13  # box_body del XML original (solo fallback)


class Detector:
    def __init__(self, weights="models/yolo_cdpr.pt", device=0, conf=0.35):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.conf = conf

    def detect(self, image_rgb, robot_z=None):
        """robot_z: altura actual del carrier (propiocepción). El rayo del
        centro de la caja se intersecta con ese plano: exacto salvo el
        descentrado caja<->cubo (mm). Sin robot_z, fallback por tamaño."""
        import cv2

        bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
        res = self.model.predict(bgr, device=self.device, conf=self.conf,
                                 verbose=False)[0]
        out = {"robot_xy": None, "sock_xy": None,
               "conf": {"robot": 0.0, "sock": 0.0}, "boxes": []}
        if res.boxes is None or len(res.boxes) == 0:
            return out
        boxes = res.boxes.xyxy.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        best = {}
        for box, c, cf in zip(boxes, clss, confs):
            key = "robot" if c == 0 else "sock"
            if key not in best or cf > best[key][1]:
                best[key] = (box, float(cf))
        rig = default_rig()
        for key, (box, cf) in best.items():
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            out["boxes"].append({"cls": key, "xyxy": box.tolist(), "conf": cf})
            if key == "sock":
                xy = ray_to_floor((float(cx), float(cy)))
            elif robot_z is not None:
                pos, direction = rig.ray((float(cx), float(cy)))
                t = (robot_z - pos[2]) / direction[2]
                hit = pos + t * direction
                xy = np.array([hit[0], hit[1]])
            else:
                h_px = float(box[3] - box[1])
                zc = rig.depth_from_apparent_height(h_px, CUBE_HEIGHT_M)
                p3 = rig.unproject((float(cx), float(cy)), zc)
                xy = np.array([p3[0], p3[1]])
            if xy is not None:
                out[f"{key}_xy"] = xy
                out["conf"][key] = cf
        return out

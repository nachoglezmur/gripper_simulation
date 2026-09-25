"""Fine-tune YOLOv8n (COCO -> cubo_robot/calcetin) en GPU + export ONNX.

Uso:
  python -m perception.train_yolo --epochs 100 --batch 8
Criterio: mAP50 > 0.90 en val. Salidas: models/yolo_cdpr.pt/.onnx
"""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--data", type=str, default="data/yolo/cdpr.yaml")
    ap.add_argument("--out", type=str, default="models")
    args = ap.parse_args()

    import pathlib

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")  # ~6 MB, se descarga solo la primera vez
    model.train(
        data=args.data, epochs=args.epochs, imgsz=640, batch=args.batch,
        device=0, project=args.out, name="yolo_cdpr_train", exist_ok=True,
        seed=7, verbose=True,
    )
    metrics = model.val(data=args.data, imgsz=640, device=0)
    map50 = float(metrics.box.map50)
    print(f"mAP50(val) = {map50:.4f}", flush=True)
    assert map50 > 0.90, f"mAP50 {map50:.4f} bajo el umbral 0.90"
    model.save(f"{args.out}/yolo_cdpr.pt")
    model.export(format="onnx", imgsz=640)
    # ultralytics guarda el onnx junto a los pesos; lo copiamos a models/
    import glob
    import shutil

    cands = glob.glob("runs/detect/models/yolo_cdpr_train/weights/*.onnx")
    assert cands, "no se generó el .onnx"
    shutil.copy(cands[0], f"{args.out}/yolo_cdpr.onnx")
    print(f"OK: {args.out}/yolo_cdpr.pt + {args.out}/yolo_cdpr.onnx", flush=True)


if __name__ == "__main__":
    main()

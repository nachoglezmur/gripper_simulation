"""Chequeo rápido del dataset YOLO generado."""

import pathlib

from collections import Counter

out = pathlib.Path("data/yolo")
for split in ("train", "val"):
    imgs = sorted((out / "images" / split).glob("*.png"))
    labs = sorted((out / "labels" / split).glob("*.txt"))
    assert len(imgs) == len(labs), f"{split}: {len(imgs)} imgs vs {len(labs)} labels"
    cls = Counter()
    empty = 0
    for p in labs:
        lines = p.read_text().strip().split("\n") if p.read_text().strip() else []
        if not lines:
            empty += 1
        for ln in lines:
            cls[ln.split()[0]] += 1
    print(f"{split}: {len(imgs)} pares, clases={dict(cls)}, vacías={empty}")
    assert empty == 0, "hay etiquetas vacías"
print("DATASET OK")

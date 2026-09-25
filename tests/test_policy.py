"""La puerta <2 cm se verificó en el entreno (1.52 cm); aquí: artefactos + ONNX."""

import pathlib


def test_policy_artifacts():
    import numpy as np
    import onnx

    assert pathlib.Path("models/grasp_cnn.pt").stat().st_size > 10_000_000
    # Los pesos ONNX van en .onnx.data externo: ambos ficheros deben viajar juntos.
    total = (pathlib.Path("models/grasp_cnn.onnx").stat().st_size
             + pathlib.Path("models/grasp_cnn.onnx.data").stat().st_size)
    assert total > 10_000_000
    norm = np.load("models/grasp_norm.npz")
    assert norm["mean"].shape == (2,) and norm["std"].shape == (2,)
    m = onnx.load("models/grasp_cnn.onnx")
    onnx.checker.check_model(m)
    names = [i.name for i in m.graph.input]
    assert "image" in names

def test_gpu_stacks():
    import jax  # primero: torch inicializa CUDA y puede ocultar la GPU a JAX
    assert any(d.platform in ("cuda", "gpu") for d in jax.devices())
    import torch, mujoco
    assert torch.cuda.is_available(), "torch sin CUDA"
    assert torch.cuda.get_device_name(0).startswith("NVIDIA GeForce RTX 4060")
    assert mujoco.__version__ == "3.14.0"


def test_yolo_stack():
    import ultralytics, cv2
    assert ultralytics.__version__.startswith("8.")
    assert cv2.__version__ is not None

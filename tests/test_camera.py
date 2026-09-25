"""Test de geometría píxel<->mundo (falla hasta crear perception/camera.py)."""


def test_project_ray_roundtrip():
    import numpy as np

    from perception.camera import ray_to_floor

    from perception.camera import project

    w = np.array([-0.30, 0.15, 0.0])
    u, v = project(w)
    back = ray_to_floor((u, v))
    assert back is not None
    assert np.linalg.norm(back - w[:2]) < 0.01  # <1 cm


def test_floor_points_inside_image():
    import numpy as np

    from perception.camera import project

    for pt in ([0, 0, 0], [0.55, 0.45, 0], [-0.3, 0.15, 0]):
        u, v = project(np.array(pt, dtype=float))
        assert 0 <= u < 640 and 0 <= v < 480, f"{pt} -> {(u, v)}"
    # La esquina (0.9,-0.9) queda fuera de encuadre por abajo: documenta el límite.
    u, v = project(np.array([0.9, -0.9, 0.0]))
    assert not (0 <= u < 640 and 0 <= v < 480)


def test_ray_sky_is_none():
    from perception.camera import ray_to_floor

    # Fila superior central: mira al cielo/fondo, no al suelo alcanzable.
    assert ray_to_floor((320, 0)) is None

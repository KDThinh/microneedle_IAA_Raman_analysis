"""Plant timelapse image analysis (Sobel edge segmentation, growth metrics).

Heavy dependencies (OpenCV) load when you import ``plant_timelapse.plant_contour_v3``.
"""


def __getattr__(name: str):  # PEP 562
    if name == "main":
        from plant_timelapse.plant_contour_v3 import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["main"]

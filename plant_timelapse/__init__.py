"""Nb timelapse helpers (segmentation, etc.).

Heavy dependencies (OpenCV) are loaded by ``plant_timelapse.segment_nb`` only when
you import or run that submodule.
"""


def __getattr__(name: str):  # PEP 562
    if name == "run_segmentation_pipeline":
        from plant_timelapse.segment_nb import run_segmentation_pipeline

        return run_segmentation_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["run_segmentation_pipeline"]

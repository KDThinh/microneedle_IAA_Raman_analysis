"""Command-line driver for the DeepLabCut keypoint pipeline.

Run with the DLC env:
    .venv-dlc\\Scripts\\python.exe -m plant_timelapse.keypoints <command>

Typical order:
    gpu-check -> build-videos -> create-project -> extract -> label
             -> train -> evaluate -> analyze -> postprocess

Heavy DeepLabCut imports are done lazily inside each command so the lightweight
commands (gpu-check, build-videos) also work outside the DLC env.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import config as C
from . import frame_export, frame_select, postprocess

app = typer.Typer(add_completion=False, help="Plant base+meristem keypoint pipeline (DeepLabCut).")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _resolve_datasets(dataset: Optional[str]):
    """Return [(name, Path), ...] for one dataset or all."""
    items = list(C.dataset_items())
    if dataset:
        items = [(n, p) for n, p in items if n == dataset]
        if not items:
            raise typer.BadParameter(f"Unknown dataset '{dataset}'. Known: {list(C.DATASETS)}")
    return items


def _save_config_pointer(cfg_path: Path) -> None:
    C.POINTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    C.POINTER_FILE.write_text(str(cfg_path), encoding="utf-8")


def _label_columns():
    """The (scorer, bodypart, coord) MultiIndex columns DLC uses for labels."""
    import pandas as pd

    cols = []
    for bp in C.KEYPOINTS:
        cols.append((C.EXPERIMENTER, bp, "x"))
        cols.append((C.EXPERIMENTER, bp, "y"))
    return pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"])


def ensure_label_rows(labeled_dir: Path, dataset: str, filenames: list[str]) -> int:
    """Ensure CollectedData_<scorer>.h5/.csv has a (NaN) row for each filename.

    napari-deeplabcut hangs when a labeled-data folder holds image files that have
    no matching row in CollectedData (image count != label-row count). Seeding
    empty rows keeps the folder internally consistent so the GUI opens cleanly;
    the rows show up as unlabeled frames for you to fill in. Returns rows added.
    """
    import numpy as np
    import pandas as pd

    h5 = labeled_dir / f"CollectedData_{C.EXPERIMENTER}.h5"
    if h5.exists():
        df = pd.read_hdf(h5)
    else:
        df = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=[None, None, None]),
            columns=_label_columns(),
        )

    to_add = [
        ("labeled-data", dataset, fn)
        for fn in filenames
        if ("labeled-data", dataset, fn) not in set(df.index)
    ]
    if not to_add:
        return 0

    add_df = pd.DataFrame(
        np.nan,
        index=pd.MultiIndex.from_tuples(to_add, names=df.index.names),
        columns=df.columns,
    )
    df = pd.concat([df, add_df])
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.to_hdf(h5, key="df_with_missing", mode="w")
    df.to_csv(h5.with_suffix(".csv"))
    return len(to_add)


def _get_config(config: Optional[str]) -> Path:
    """Resolve the active DLC config.yaml (explicit > pointer file > newest on disk)."""
    if config:
        p = Path(config)
        if not p.exists():
            raise typer.BadParameter(f"config not found: {p}")
        return p
    if C.POINTER_FILE.exists():
        p = Path(C.POINTER_FILE.read_text(encoding="utf-8").strip())
        if p.exists():
            return p
    candidates = sorted(C.DLC_DIR.glob(f"{C.PROJECT_NAME}-*/config.yaml"))
    if candidates:
        return candidates[-1]
    raise typer.Exit(
        code=typer.echo("No DLC project found. Run 'create-project' first.", err=True) or 1
    )


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
@app.command("gpu-check")
def gpu_check():
    """Report PyTorch/CUDA availability and the detected GPU."""
    import torch

    avail = torch.cuda.is_available()
    typer.echo(f"torch: {torch.__version__}")
    typer.echo(f"cuda available: {avail}")
    if avail:
        typer.echo(f"device: {torch.cuda.get_device_name(0)}")
        typer.echo(f"cuda runtime: {torch.version.cuda}")
    else:
        typer.echo("WARNING: training will run on CPU (slow). Check the CUDA wheel install.", err=True)


@app.command("build-videos")
def build_videos(
    dataset: Optional[str] = typer.Option(None, help="Only this dataset (default: all)."),
    step: int = typer.Option(1, help="Use every Nth frame (1 = all frames)."),
    fps: int = typer.Option(30, help="Output video frame rate."),
    limit: Optional[int] = typer.Option(None, help="Cap number of frames (debug)."),
):
    """Encode each dataset into a cropped 8-bit mp4 (absolute 0-4095 scaling + ROI)."""
    for name, path in _resolve_datasets(dataset):
        if not path.exists():
            typer.echo(f"[skip] {name}: path not found: {path}", err=True)
            continue
        out = C.video_path(name)
        typer.echo(f"[{name}] {path}")
        out_path, n, (w, h) = frame_export.build_video(path, out, fps=fps, step=step, limit=limit)
        dn = frame_select.summarize_day_night(frame_export.list_tiffs(path)[::step][:limit] if limit else frame_export.list_tiffs(path)[::step])
        typer.echo(f"  -> {out_path}  ({n} frames @ {w}x{h}; day={dn['day']} night={dn['night']})")


@app.command("create-project")
def create_project(
    dataset: Optional[str] = typer.Option(None, help="Only this dataset's video (default: all)."),
):
    """Create the DeepLabCut project and set bodyparts to base + meristem."""
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions

    videos = []
    for name, _ in _resolve_datasets(dataset):
        v = C.video_path(name)
        if not v.exists():
            raise typer.BadParameter(f"Missing video {v}. Run 'build-videos' first.")
        videos.append(str(v))

    C.DLC_DIR.mkdir(parents=True, exist_ok=True)
    cfg = deeplabcut.create_new_project(
        C.PROJECT_NAME,
        C.EXPERIMENTER,
        videos,
        working_directory=str(C.DLC_DIR),
        copy_videos=True,
        multianimal=False,
    )
    if cfg is None:
        # project already exists; fall back to newest on disk
        cfg = str(sorted(C.DLC_DIR.glob(f"{C.PROJECT_NAME}-*/config.yaml"))[-1])

    auxiliaryfunctions.edit_config(
        cfg,
        {
            "bodyparts": C.KEYPOINTS,
            "numframes2pick": C.NUMFRAMES2PICK,
            "dotsize": 6,
        },
    )
    _save_config_pointer(Path(cfg))
    typer.echo(f"Project config: {cfg}")
    typer.echo(f"bodyparts set to {C.KEYPOINTS}; numframes2pick={C.NUMFRAMES2PICK}")


@app.command("extract")
def extract(
    config: Optional[str] = typer.Option(None, help="Path to config.yaml (default: active project)."),
    numframes: Optional[int] = typer.Option(None, help="Override frames to pick per video."),
):
    """Auto-select diverse frames to label (kmeans on each cropped video)."""
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions

    cfg = _get_config(config)
    if numframes is not None:
        auxiliaryfunctions.edit_config(str(cfg), {"numframes2pick": numframes})
    deeplabcut.extract_frames(str(cfg), mode="automatic", algo="kmeans", userfeedback=False, crop=False)
    typer.echo("Extraction done. Next: 'label'.")


@app.command("label")
def label(
    dataset: Optional[str] = typer.Option(
        None,
        help="Which labeled-data folder to open (e.g. run4, run5). Default: first folder.",
    ),
    config: Optional[str] = typer.Option(None, help="Path to config.yaml (default: active project)."),
):
    """Launch the DeepLabCut labeling GUI (click base + meristem on each frame).

    Each dataset gets its OWN labeled-data folder, so run this once per dataset
    (e.g. --dataset run4, then --dataset run5). Save inside napari with Ctrl+S.
    """
    import napari
    from qtpy.QtCore import QTimer

    cfg = _get_config(config)
    data_dir = cfg.parent / "labeled-data"
    if dataset is not None:
        image_dir = data_dir / dataset
        if not image_dir.is_dir():
            avail = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
            raise typer.BadParameter(f"No labeled-data folder '{dataset}'. Available: {avail}")
    else:
        subdirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
        if not subdirs:
            raise typer.BadParameter(f"No labeled-data folders under {data_dir}. Run 'extract' first.")
        image_dir = subdirs[0]

    typer.echo(f"Launching DLC labeling GUI for '{image_dir.name}'.")
    typer.echo("Label 'base' (stem at soil) and 'meristem' (growth tip).")
    typer.echo("When done: press Ctrl+S to save, then close the window.")

    # --- Work around a napari hang when opening the folder ---------------------
    # viewer.open() spawns a napari `progress` object; the Qt activity dialog reacts
    # by building a progress-bar widget and calling QApplication.processEvents()
    # re-entrantly (qt_activity_dialog.add_progress_bar, line ~214). In the
    # DeepLabCut + napari 0.6.6 + PySide6 combo this re-entrant processEvents() can
    # spin at 100% CPU forever (confirmed via py-spy: MainThread stuck in
    # add_progress_bar). It's a race, which is why some folders open and others hang.
    # We don't need GUI progress bars for labeling, so neuter the activity dialog's
    # progress handler BEFORE the viewer (and thus the dialog) is created.
    try:
        from napari._qt.dialogs import qt_activity_dialog as _qad

        _qad.QtActivityDialog.handle_progress_change = lambda self, event: None
    except Exception as exc:  # pragma: no cover - defensive against napari internals
        typer.echo(f"(warning: could not disable napari progress bars: {exc})", err=True)

    # We drive napari ourselves (instead of deeplabcut.label_frames) so we control
    # the launch. Defer add_plugin_dock_widget + open onto a 0 ms timer so they run
    # inside the event loop. napari.run() blocks until the window is closed.
    viewer = napari.Viewer()

    def _startup():
        viewer.window.add_plugin_dock_widget("napari-deeplabcut", "Keypoint controls")
        # Open the image folder first. Passing [folder, config.yaml] in a single
        # viewer.open() hangs on folders that do not yet have CollectedData_*.h5
        # (run5 fresh labeling). run4 worked only because saved labels were already
        # present and the folder reader picked them up without needing config in
        # the same open() call.
        viewer.open([str(image_dir)], plugin="napari-deeplabcut", stack=False)
        has_points = any(getattr(layer, "name", "").startswith("CollectedData") for layer in viewer.layers)
        if not has_points:
            viewer.open([str(cfg)], plugin="napari-deeplabcut", stack=False)

    QTimer.singleShot(0, _startup)
    napari.run()


@app.command("check-labels")
def check_labels(
    config: Optional[str] = typer.Option(None, help="Path to config.yaml (default: active project)."),
):
    """Render labeled frames with markers to verify annotation quality."""
    import deeplabcut

    cfg = _get_config(config)
    deeplabcut.check_labels(str(cfg))
    typer.echo("Wrote *_labeled folders under labeled-data/. Inspect for misplaced points.")


@app.command("train")
def train(
    config: Optional[str] = typer.Option(None, help="Path to config.yaml (default: active project)."),
    net: str = typer.Option(C.DEFAULT_NET_TYPE, help="Backbone (e.g. resnet_50, hrnet_w32)."),
    epochs: Optional[int] = typer.Option(None, help="Training epochs (PyTorch engine)."),
    save_epochs: Optional[int] = typer.Option(None, help="Checkpoint interval."),
    batch_size: Optional[int] = typer.Option(None, help="Batch size (lower if OOM on 8GB)."),
):
    """Create the training dataset and train the keypoint model."""
    import deeplabcut

    cfg = _get_config(config)
    typer.echo(f"Creating training dataset (net={net}) ...")
    deeplabcut.create_training_dataset(str(cfg), net_type=net)

    kwargs = {}
    if epochs is not None:
        kwargs["epochs"] = epochs
    if save_epochs is not None:
        kwargs["save_epochs"] = save_epochs
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    typer.echo(f"Training ... {kwargs or '(defaults)'}")
    deeplabcut.train_network(str(cfg), **kwargs)
    typer.echo("Training done. Next: 'evaluate'.")


@app.command("evaluate")
def evaluate(
    config: Optional[str] = typer.Option(None, help="Path to config.yaml (default: active project)."),
):
    """Evaluate the trained network (writes error metrics + plots)."""
    import deeplabcut

    cfg = _get_config(config)
    deeplabcut.evaluate_network(str(cfg), plotting=True)
    typer.echo(
        "Evaluation done. See evaluation-results-pytorch/iteration-0/ "
        "for metrics CSV, per-frame H5, and LabeledImages overlays."
    )


@app.command("analyze")
def analyze(
    dataset: Optional[str] = typer.Option(None, help="Only this dataset (default: all)."),
    config: Optional[str] = typer.Option(None, help="Path to config.yaml (default: active project)."),
    filter: bool = typer.Option(True, help="Also run temporal median filtering."),
    make_video: bool = typer.Option(False, help="Render a labeled overlay video."),
):
    """Run the trained model on the cropped videos (writes predictions .h5/.csv)."""
    import deeplabcut

    cfg = _get_config(config)
    videos = [str(C.video_path(name)) for name, _ in _resolve_datasets(dataset)]
    deeplabcut.analyze_videos(str(cfg), videos, save_as_csv=True)
    if filter:
        deeplabcut.filterpredictions(str(cfg), videos)
    if make_video:
        deeplabcut.create_labeled_video(str(cfg), videos)
    typer.echo("Analysis done. Next: 'postprocess'.")


@app.command("add-label-frames")
def add_label_frames(
    dataset: str = typer.Argument(..., help="Dataset name (e.g. run4)."),
    frames: str = typer.Option(..., "--frames", help="Comma-separated video frame indices, e.g. 177,185,188"),
    step: int = typer.Option(1, help="Same --step used in build-videos."),
    config: Optional[str] = typer.Option(None, help="Path to config.yaml (default: active project)."),
):
    """Export specific video frames into labeled-data/ for manual correction.

    Use after spot-checking bad predictions in the QC notebook. Then run
    ``label --dataset <name>``, ``check-labels``, ``train``, and ``analyze``.
    """
    import cv2

    cfg = _get_config(config)
    data_dir = cfg.parent / "labeled-data" / dataset
    if not data_dir.is_dir():
        avail = sorted(p.name for p in (cfg.parent / "labeled-data").iterdir() if p.is_dir())
        raise typer.BadParameter(f"No labeled-data folder '{dataset}'. Available: {avail}")

    dataset_path = dict(C.DATASETS).get(dataset)
    if dataset_path is None:
        raise typer.BadParameter(f"Unknown dataset '{dataset}'.")

    indices = [int(x.strip()) for x in frames.split(",") if x.strip()]
    tiffs = frame_export.list_tiffs(dataset_path)[::step]
    written_names = []
    for i in indices:
        if not 0 <= i < len(tiffs):
            typer.echo(f"[skip] frame {i}: out of range (0..{len(tiffs) - 1})", err=True)
            continue
        fr = frame_export.prepare_frame(tiffs[i])
        name = f"frame{i:03d}.png"
        dst = data_dir / name
        cv2.imwrite(str(dst), fr)
        typer.echo(f"  frame {i} -> {dst.name}")
        written_names.append(name)

    # Seed empty label rows so the folder stays consistent (image count ==
    # label-row count); otherwise napari-deeplabcut hangs on open.
    seeded = ensure_label_rows(data_dir, dataset, written_names)
    typer.echo(
        f"Wrote {len(written_names)} frame(s) to {data_dir}; "
        f"seeded {seeded} empty label row(s). Next: label --dataset {dataset}"
    )


@app.command("postprocess")
def postprocess_cmd(
    dataset: Optional[str] = typer.Option(None, help="Only this dataset (default: all)."),
    config: Optional[str] = typer.Option(None, help="Path to config.yaml (default: active project)."),
    step: int = typer.Option(1, help="Same --step used in build-videos (for timestamp alignment)."),
    smooth_window: int = typer.Option(11, help="Savitzky-Golay window (odd)."),
    likelihood_floor: float = typer.Option(0.0, help="Default confidence floor for both keypoints."),
    base_likelihood_floor: Optional[float] = typer.Option(
        None, help="Override floor for base (defaults to --likelihood-floor)."
    ),
    meristem_likelihood_floor: Optional[float] = typer.Option(
        None, help="Override floor for meristem (defaults to --likelihood-floor)."
    ),
    interpolate_limit: Optional[int] = typer.Option(
        None, help="Max consecutive low-confidence frames to bridge (e.g. 3)."
    ),
    destep_jump_px: Optional[float] = typer.Option(
        None,
        help="Correct step jumps in interpolated y coords (px). E.g. 15 shifts "
        "subsequent frames after sudden jumps (background-style correction).",
    ),
    monotonic: bool = typer.Option(False, help="Apply non-decreasing growth prior."),
):
    """Convert DLC predictions into a stem-height CSV (raw + smoothed)."""
    cfg = _get_config(config)
    proj_dir = cfg.parent
    for name, path in _resolve_datasets(dataset):
        pred = _find_prediction(proj_dir, name)
        if pred is None:
            typer.echo(f"[skip] {name}: no prediction file found (run 'analyze').", err=True)
            continue
        table = postprocess.build_height_table(
            pred,
            dataset_dir=path,
            step=step,
            smooth_window=smooth_window,
            likelihood_floor=likelihood_floor,
            base_likelihood_floor=base_likelihood_floor,
            meristem_likelihood_floor=meristem_likelihood_floor,
            interpolate_limit=interpolate_limit,
            destep_jump_px=destep_jump_px,
            monotonic=monotonic,
        )
        out_csv = C.DLC_DIR / f"stem_height_{name}.csv"
        table.to_csv(out_csv, index=False)
        typer.echo(f"[{name}] {pred.name} -> {out_csv}  ({len(table)} frames)")


def _find_prediction(proj_dir: Path, dataset_name: str) -> Optional[Path]:
    """Locate the DLC prediction for a dataset (prefer filtered .h5).

    analyze_videos writes next to the source mp4 (videos_prepared/). Older DLC
    layouts may also place outputs under the project videos/ copy.
    """
    search_dirs = [C.VIDEO_DIR, proj_dir / "videos"]
    h5s: list[Path] = []
    for d in search_dirs:
        if d.is_dir():
            h5s.extend(sorted(d.glob(f"{dataset_name}*.h5")))
    if not h5s:
        return None
    filtered = [p for p in h5s if "filtered" in p.name]
    return filtered[-1] if filtered else h5s[-1]


if __name__ == "__main__":
    app()

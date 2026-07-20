"""
Training driver. One function, three models, identical settings.

The paper's entire claim is "identical conditions across three models", so
per-model tuning is FORBIDDEN. Everything comes from src/config.py. If you find
yourself passing a different lr for one model, the controlled comparison is
dead and the paper's contribution goes with it.

Resume-safe: Kaggle kills sessions at 12h. Checkpoints save every epoch and
`train_one(..., resume=True)` picks up where it stopped.
"""

from __future__ import annotations

import os
from pathlib import Path

from .. import config as C
from ..utils.run_logger import RunLogger


def setup_wandb(project: str = "vindr-iccit2026", enabled: bool = True) -> bool:
    """Enable W&B via Ultralytics' native integration.

    Why bother: it streams metrics to cloud as they are produced, so a run
    killed at hour 11 of 12 still has every curve. Local logs die with the
    session. Overhead is ~1-3% (async background process); image logging is the
    expensive part, not scalars.

    Measure it yourself on the D3 smoke test: run 5 epochs with and without,
    compare `sec_per_epoch` from RunLogger. If >5%, disable image logging first.
    """
    if not enabled:
        os.environ["WANDB_MODE"] = "disabled"
        print("[wandb] disabled")
        return False
    try:
        import wandb  # noqa: F401
        from ultralytics import settings
        settings.update({"wandb": True})
        os.environ["WANDB_PROJECT"] = project
        # Do NOT use WANDB_MODE=offline -- offline writes to local disk, which
        # the 12h session kill takes with it. That defeats the whole point.
        os.environ["WANDB_MODE"] = "online"
        print(f"[wandb] enabled, project={project}")
        return True
    except ImportError:
        print("[wandb] not installed, continuing without")
        return False


def train_one(
    model_key: str,
    data_yaml: Path,
    seed: int | None = None,
    epochs: int | None = None,
    resume: bool = False,
    use_wandb: bool = True,
    runs_dir: Path | None = None,
    **overrides,
):
    """Train one model under the frozen protocol.

    Parameters
    ----------
    model_key : "yolov8s" | "yolo11s" | "yolo26s"
    overrides : escape hatch for the D3 smoke test ONLY (e.g. epochs=5).
                Anything overridden is logged, so a smoke run can never be
                silently mistaken for a real one.
    """
    from ultralytics import YOLO

    if model_key not in C.MODELS:
        raise ValueError(f"unknown model {model_key!r}, expected one of {list(C.MODELS)}")

    seed = C.SEED if seed is None else seed
    epochs = C.EPOCHS if epochs is None else epochs
    runs_dir = Path(runs_dir or C.RUNS_DIR)
    run_name = f"{model_key}_{C.IMGSZ}_e{epochs}_seed{seed}"

    setup_wandb(enabled=use_wandb)

    log = RunLogger(run_name=run_name, out_dir=str(runs_dir))
    params = {**C.as_dict(), "model": model_key, "epochs": epochs, "seed": seed,
              "weights": C.MODELS[model_key], "resumed": resume}
    if overrides:
        params["OVERRIDES"] = str(overrides)
        print(f"[train] ⚠ overrides active: {overrides} -- not a protocol run")
    log.set_params(**params)

    model = YOLO(C.MODELS[model_key])

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=C.IMGSZ,
        batch=C.BATCH,
        device=C.DEVICE,
        workers=C.WORKERS,
        amp=C.AMP,
        seed=seed,
        project=str(runs_dir / "ultralytics"),
        name=run_name,
        exist_ok=True,
        save_period=1,        # every epoch -- the 12h kill is not negotiable
        resume=resume,
        val=True,
        plots=True,
        verbose=True,
    )
    train_kwargs.update(overrides)

    try:
        results = model.train(**train_kwargs)
        log.from_ultralytics(results)
        # True per-epoch time, excluding weight downloads / AMP check / label
        # scan. Wall-clock/epochs overstates it by ~25% on a short run.
        log.epoch_times_from_ultralytics(runs_dir / "ultralytics" / run_name)
        status = "ok"
    except Exception as e:
        print(f"[train] FAILED: {e}")
        log.set_params(error=str(e)[:500])
        status = "failed"
        raise
    finally:
        log.finish(status=status)

    # Reminder, because it is the single easiest way to ruin the results table.
    print("\n[train] NOTE: mAP@0.4 (the competition metric) is NOT in these "
          "results. Ultralytics reports @0.5 and @0.5:0.95 only.\n"
          "        Run src/evaluation/metrics.py to compute @0.4 before "
          "building any table.\n")
    return model, results


def train_all(
    data_yaml: Path,
    models: list[str] | None = None,
    use_wandb: bool = True,
    **overrides,
) -> dict:
    """Sequential training of all three. Single GPU, one after another.

    Sequential not parallel: 16 GB holds one s-scale run comfortably at 512px,
    and two concurrent runs would contend for bandwidth -- which is exactly the
    resource P100 is chosen for.
    """
    models = models or list(C.MODELS)
    out = {}
    for i, key in enumerate(models, 1):
        print(f"\n{'='*70}\n[{i}/{len(models)}] {key}\n{'='*70}")
        try:
            model, results = train_one(
                key, data_yaml, use_wandb=use_wandb, **overrides
            )
            out[key] = {"model": model, "results": results, "status": "ok"}
        except Exception as e:
            # One model failing must not lose the other two.
            print(f"[train_all] {key} failed, continuing: {e}")
            out[key] = {"status": "failed", "error": str(e)}
    return out


def best_weights(run_name: str, runs_dir: Path | None = None) -> Path:
    """Path to best.pt for a finished run."""
    runs_dir = Path(runs_dir or C.RUNS_DIR)
    p = runs_dir / "ultralytics" / run_name / "weights" / "best.pt"
    if not p.exists():
        raise FileNotFoundError(f"no weights at {p}")
    return p


def all_best_weights(runs_dir: Path | None = None) -> dict[str, str]:
    """{model_key: path} for benchmark_inference() and the XAI stage."""
    runs_dir = Path(runs_dir or C.RUNS_DIR)
    out = {}
    for key in C.MODELS:
        run_name = f"{key}_{C.IMGSZ}_e{C.EPOCHS}_seed{C.SEED}"
        p = runs_dir / "ultralytics" / run_name / "weights" / "best.pt"
        if p.exists():
            out[key] = str(p)
        else:
            print(f"[weights] missing: {key} ({p})")
    return out

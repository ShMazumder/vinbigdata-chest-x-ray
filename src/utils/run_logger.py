"""
Run provenance logger for VinDr-CXR detection experiments.

Captures hardware, software, hyperparameters and results for every training run
so that results tables can be reconstructed on D10 without archaeology.

Design goals (12-day deadline):
  - Zero config. Import, wrap the run, done.
  - Append-only. Never overwrites a previous run.
  - Two outputs: one JSON per run (full detail) + one master CSV (table-ready).
  - Fails soft. Logging must never crash training.

Usage
-----
    from run_logger import RunLogger, benchmark_inference

    log = RunLogger(run_name="yolo26s_512_e40_seed0", out_dir="/kaggle/working/runs")
    log.set_params(
        model="yolo26s", imgsz=512, epochs=40, batch=16, seed=0,
        subset="positive_only", fusion="wbf@0.5", split="80/10/10",
    )
    log.set_data(n_train=3520, n_val=440, n_test=440, n_classes=14)

    # ... train ...

    log.set_results(map40=0.271, map50=0.243, map5095=0.118, per_class={...})
    log.finish()

Then on D10:
    RunLogger.master_table("/kaggle/working/runs")   # -> pandas DataFrame
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Environment capture
# --------------------------------------------------------------------------


def _safe(fn, default=None):
    """Run fn, swallow anything. Logging must never kill a training run."""
    try:
        return fn()
    except Exception:
        return default


# PyTorch dropped Pascal (sm_60) support. Current builds require sm_70+.
# Kaggle's P100 is sm_60 and WILL NOT TRAIN -- it fails or falls back to CPU.
# Verified against a live Kaggle session 2026-07-20.
MIN_COMPUTE_CAPABILITY = (7, 0)


def capture_gpu(strict: bool = False) -> dict[str, Any]:
    """Hardware fingerprint, with a supported-architecture check.

    T4   (Turing TU104) -> sm_75, 16 GB GDDR6, 320 GB/s, tensor cores.  USE THIS.
    P100 (Pascal GP100) -> sm_60, 16 GB HBM2, 732 GB/s, no tensor cores.
                           Faster on paper. Unsupported by PyTorch. Unusable.

    strict=True raises rather than warns -- use it at the top of a training
    notebook so an unusable accelerator fails in seconds, not after a session
    of confusing CPU-speed epochs.
    """
    import torch

    if not torch.cuda.is_available():
        info = {"gpu_available": False, "gpu_supported": False}
        if strict:
            raise RuntimeError("no CUDA device -- check the notebook accelerator setting")
        return info

    cc = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)

    family = {(6, 0): "P100", (7, 5): "T4", (8, 0): "A100"}.get(cc, props.name)
    supported = cc >= MIN_COMPUTE_CAPABILITY

    info = {
        "gpu_available": True,
        "gpu_supported": supported,
        "gpu_name": props.name,
        "gpu_family": family,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "has_tensor_cores": cc[0] >= 7,
        "vram_gb": round(props.total_memory / 1e9, 2),
        "n_gpus_visible": torch.cuda.device_count(),
        "multi_processor_count": props.multi_processor_count,
    }

    if not supported:
        msg = (
            f"\n{'='*70}\n"
            f"UNUSABLE GPU: {props.name} is sm_{cc[0]}{cc[1]}, below PyTorch's "
            f"sm_{MIN_COMPUTE_CAPABILITY[0]}{MIN_COMPUTE_CAPABILITY[1]} floor.\n"
            f"Training will fail or silently run on CPU.\n\n"
            f"FIX: notebook settings -> Accelerator -> 'GPU T4 x2' (sm_75).\n"
            f"Keep device=0; a single T4 is correct. Do not attempt DDP.\n"
            f"{'='*70}"
        )
        if strict:
            raise RuntimeError(msg)
        print(msg)

    return info


def capture_env() -> dict[str, Any]:
    """Software fingerprint + git commit, for the reproducibility statement."""
    import torch

    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": _safe(lambda: torch.version.cuda),
        "cudnn": _safe(lambda: torch.backends.cudnn.version()),
    }

    env["ultralytics"] = _safe(
        lambda: __import__("ultralytics").__version__, "not_installed"
    )
    env["git_commit"] = _safe(
        lambda: subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        .decode()
        .strip(),
        "no_git",
    )
    # Kaggle exposes these; useful for telling sessions apart after the fact.
    env["kaggle_kernel"] = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "not_kaggle")
    env["session_id"] = os.environ.get("KAGGLE_DATA_PROXY_TOKEN", "")[:8] or "local"
    return env


# --------------------------------------------------------------------------
# Logger
# --------------------------------------------------------------------------

# Column order for the master CSV. Anything not here still lands in the JSON.
CSV_FIELDS = [
    "run_name", "timestamp_utc", "status",
    # hardware -- keep adjacent to results, this is the FPS confound guard
    "gpu_family", "gpu_name", "compute_capability", "vram_gb", "has_tensor_cores",
    # params
    "model", "imgsz", "epochs", "batch", "optimizer", "lr0", "seed", "amp",
    "device", "subset", "fusion", "split",
    # data
    "n_train", "n_val", "n_test", "n_classes",
    # results
    "map40", "map50", "map5095", "precision", "recall",
    # cost -- sec_per_epoch includes setup; sec_per_epoch_true does not.
    "wall_clock_min", "sec_per_epoch", "sec_per_epoch_true", "train_only_min", "fps",
    # software
    "torch", "ultralytics", "cuda", "git_commit",
]


class RunLogger:
    def __init__(self, run_name: str, out_dir: str = "runs"):
        self.run_name = run_name
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

        self.record: dict[str, Any] = {"run_name": run_name, "status": "running"}
        self.record.update(capture_env())
        self.record.update(capture_gpu())

        print(f"[run_logger] {run_name}")
        print(f"[run_logger] GPU: {self.record.get('gpu_name')} "
              f"(CC {self.record.get('compute_capability')}, "
              f"{self.record.get('vram_gb')} GB, "
              f"tensor_cores={self.record.get('has_tensor_cores')})")

    # -- inputs ------------------------------------------------------------

    def set_params(self, **kwargs) -> "RunLogger":
        """Hyperparameters. Log these BEFORE training, not after."""
        self.record.update(kwargs)
        return self

    def set_data(self, **kwargs) -> "RunLogger":
        """Dataset shape: n_train, n_val, n_test, n_classes, class_counts..."""
        self.record.update(kwargs)
        return self

    def set_results(self, per_class: dict | None = None, **kwargs) -> "RunLogger":
        """Metrics. per_class goes to JSON only, not the flat CSV."""
        self.record.update(kwargs)
        if per_class:
            self.record["per_class"] = per_class
        return self

    def from_ultralytics(self, results) -> "RunLogger":
        """Pull metrics straight off an Ultralytics results object.

        Note: Ultralytics reports mAP@0.5 and mAP@0.5:0.95 natively. mAP@0.4 --
        the VinBigData competition metric -- is NOT produced by default and must
        be computed separately. Do not silently report @0.5 as if it were @0.4.
        """
        box = _safe(lambda: results.box)
        if box is None:
            print("[run_logger] WARN: no .box on results, skipping auto-extract")
            return self
        self.record.update({
            "map50": _safe(lambda: float(box.map50)),
            "map5095": _safe(lambda: float(box.map)),
            "precision": _safe(lambda: float(box.mp)),
            "recall": _safe(lambda: float(box.mr)),
        })
        return self

    # -- output ------------------------------------------------------------

    def epoch_times_from_ultralytics(self, save_dir) -> "RunLogger":
        """Read true per-epoch times from Ultralytics' results.csv.

        `sec_per_epoch` derived from wall-clock divides by epochs and therefore
        folds in setup: weight downloads, the AMP check, label scanning. On a
        5-epoch smoke run that inflated the figure by ~25% (54.3 vs a real
        ~42 s). Harmless for a smoke test, wrong in a paper.
        """
        import csv as _csv

        path = Path(save_dir) / "results.csv"
        if not path.exists():
            return self
        try:
            with path.open() as f:
                rows = list(_csv.DictReader(f))
            key = next((k for k in rows[0] if k.strip() == "time"), None)
            if not key or len(rows) < 2:
                return self
            cum = [float(r[key]) for r in rows]
            per = [b - a for a, b in zip(cum, cum[1:])]
            self.record["sec_per_epoch_true"] = round(sum(per) / len(per), 1)
            self.record["train_only_min"] = round(cum[-1] / 60, 2)
        except Exception as e:
            print(f"[run_logger] could not parse results.csv: {e}")
        return self

    def finish(self, status: str = "ok") -> dict[str, Any]:
        elapsed_min = (time.time() - self._t0) / 60
        self.record["status"] = status
        self.record["wall_clock_min"] = round(elapsed_min, 2)
        epochs = self.record.get("epochs")
        if epochs:
            # NOTE: includes setup (downloads, AMP check, label scan).
            # sec_per_epoch_true, when present, is the figure to report.
            self.record["sec_per_epoch"] = round(elapsed_min * 60 / epochs, 1)

        json_path = self.out_dir / f"{self.run_name}.json"
        json_path.write_text(json.dumps(self.record, indent=2, default=str))

        self._append_csv()
        print(f"[run_logger] wrote {json_path}")
        print(f"[run_logger] {elapsed_min:.1f} min "
              f"({self.record.get('sec_per_epoch', '?')} s/epoch)")
        return self.record

    def _append_csv(self) -> None:
        csv_path = self.out_dir / "master_results.csv"
        row = {k: self.record.get(k, "") for k in CSV_FIELDS}
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)

    # -- read back ---------------------------------------------------------

    @staticmethod
    def run_is_complete(run_dir, expected_epochs: int) -> bool:
        """True only if training actually ran to completion.

        `best.pt` existing is NOT sufficient -- Ultralytics rewrites it every
        epoch, so a run killed at epoch 12 leaves a perfectly loadable
        checkpoint that a naive existence check treats as finished. The
        authoritative signal is the row count in results.csv.
        """
        run_dir = Path(run_dir)
        if not (run_dir / "weights" / "best.pt").exists():
            return False
        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            return False
        try:
            with csv_path.open() as f:
                n = sum(1 for _ in csv.reader(f)) - 1     # minus header
            return n >= expected_epochs
        except Exception:
            return False

    @staticmethod
    def cleanup_incomplete(out_dir="runs", expected_epochs=None, dry_run=True):
        """Delete partial run directories and their JSON records.

        Partial runs are actively dangerous here: they carry a loadable but
        under-trained best.pt, which silently poisons both the skip logic and
        any evaluation that globs for weights.
        """
        import shutil

        out_dir = Path(out_dir)
        ult = out_dir / "ultralytics"
        if expected_epochs is None:
            from .. import config as C
            expected_epochs = C.EPOCHS

        removed, kept = [], []
        for run_dir in sorted(p for p in ult.glob("*") if p.is_dir()):
            if RunLogger.run_is_complete(run_dir, expected_epochs):
                kept.append(run_dir.name)
                continue
            removed.append(run_dir.name)
            if not dry_run:
                shutil.rmtree(run_dir, ignore_errors=True)
                (out_dir / f"{run_dir.name}.json").unlink(missing_ok=True)

        # JSONs whose run directory is gone entirely (e.g. failed at setup).
        orphans = [j for j in out_dir.glob("*.json")
                   if j.stem != "eval_summary" and not (ult / j.stem).is_dir()]
        for j in orphans:
            removed.append(f"{j.name} (orphan record)")
            if not dry_run:
                j.unlink(missing_ok=True)

        verb = "would remove" if dry_run else "removed"
        print(f"complete ({len(kept)}): {kept}")
        print(f"{verb} ({len(removed)}): {removed}")
        if dry_run and removed:
            print("\nre-run with dry_run=False to delete")
        return {"kept": kept, "removed": removed}

    @staticmethod
    def rebuild_master(out_dir="runs", expected_epochs=None):
        """Regenerate master_results.csv from the per-run JSONs.

        Use after cleanup_incomplete(), or any time the CSV has accumulated
        rows from failed attempts. The JSONs are the source of truth; the CSV
        is a derived convenience.
        """
        out_dir = Path(out_dir)
        if expected_epochs is None:
            from .. import config as C
            expected_epochs = C.EPOCHS

        rows = []
        for j in sorted(out_dir.glob("*.json")):
            if j.stem == "eval_summary":
                continue
            try:
                rec = json.loads(j.read_text())
            except Exception as e:
                print(f"  skip unreadable {j.name}: {e}")
                continue
            if rec.get("status") != "ok":
                print(f"  skip {j.stem}: status={rec.get('status')}")
                continue
            run_dir = out_dir / "ultralytics" / j.stem
            if not RunLogger.run_is_complete(run_dir, expected_epochs):
                print(f"  skip {j.stem}: incomplete on disk")
                continue
            rows.append({k: rec.get(k, "") for k in CSV_FIELDS})

        csv_path = out_dir / "master_results.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {csv_path} with {len(rows)} valid runs")
        return csv_path

    @staticmethod
    def master_table(out_dir: str = "runs"):
        """Load every run into a DataFrame. Use this to build the paper tables."""
        import pandas as pd

        return pd.read_csv(Path(out_dir) / "master_results.csv")

    @staticmethod
    def check_hardware_consistency(out_dir: str = "runs") -> bool:
        """Guard the FPS confound.

        mAP is hardware-independent. Latency is NOT. If any speed number is
        reported in the paper, every model must have been benchmarked on the
        same card. Call this before writing the results section.
        """
        import pandas as pd

        df = pd.read_csv(Path(out_dir) / "master_results.csv")
        fams = df["gpu_family"].dropna().unique()
        if len(fams) <= 1:
            print(f"[run_logger] hardware consistent: {fams}")
            return True
        print(f"[run_logger] ⚠️  MIXED HARDWARE across runs: {fams}")
        print("[run_logger] mAP comparisons remain valid.")
        print("[run_logger] Any FPS / latency / throughput claim is INVALID.")
        print("[run_logger] Re-run benchmark_inference() for all checkpoints "
              "in ONE session on ONE card before reporting speed.")
        return False


# --------------------------------------------------------------------------
# Inference benchmark -- run all checkpoints back to back, one session
# --------------------------------------------------------------------------


def benchmark_inference(
    model_paths: dict[str, str],
    imgsz: int = 512,
    n_warmup: int = 10,
    n_iter: int = 100,
    out_dir: str = "runs",
) -> dict[str, dict]:
    """Time every model on the SAME card in ONE session.

    YOLO26 is marketed on edge/CPU inference speed, so a latency table is
    expected in this paper. It is only valid if produced here, not stitched
    together from training runs that may have landed on different GPUs.
    """
    import numpy as np
    import torch
    from ultralytics import YOLO

    gpu = capture_gpu()
    print(f"[benchmark] card: {gpu.get('gpu_name')} ({gpu.get('gpu_family')})")

    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    out: dict[str, dict] = {}

    for name, path in model_paths.items():
        model = YOLO(path)
        for _ in range(n_warmup):
            model.predict(dummy, imgsz=imgsz, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iter):
            model.predict(dummy, imgsz=imgsz, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        ms = dt / n_iter * 1000
        out[name] = {
            "ms_per_image": round(ms, 2),
            "fps": round(1000 / ms, 1),
            "n_params_M": _safe(
                lambda: round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
            ),
            **gpu,
        }
        print(f"[benchmark] {name}: {ms:.2f} ms  ({1000/ms:.1f} FPS)")

    path = Path(out_dir) / "inference_benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"[benchmark] wrote {path}")
    return out


if __name__ == "__main__":
    print(json.dumps({**capture_env(), **capture_gpu()}, indent=2, default=str))

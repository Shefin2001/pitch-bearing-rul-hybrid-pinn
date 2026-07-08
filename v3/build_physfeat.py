"""build_physfeat.py -- Stage 10: per-run physics features -> physfeat.parquet.

Gathers each (speed, condition, file_idx) recording from the raw parquet,
band-passes it, and extracts the PHYS_DIM-vector via signal_physics.

Parallel backends (pick one):
    --gpu            CuPy on the H200, serial over runs (default when CUDA)
    --workers N      CPU ProcessPoolExecutor (default: all vCPUs)
    --mpi            mpi4py rank-sharding:  mpirun -np 16 python -m ...v3.build_physfeat --mpi

Checkpointing: every finished run is appended to physfeat_partial.jsonl
(crash-safe append). On restart, done runs are skipped. Final output is
physfeat.parquet + the .physfeat_complete sentinel.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from common.config import Config  # noqa: E402
from Hybrid_PINN_ParisRUL.common.dataset_v2 import (  # noqa: E402
    _read_parquet_rg,
    apply_bandpass,
    design_bandpass,
    discover_runs,
)
from Hybrid_PINN_ParisRUL.v3.signal_physics import (  # noqa: E402
    PHYS_DIM,
    PHYS_NAMES,
    PhysFeatureExtractor,
)

RESULTS_DIR = ROOT / "Hybrid_PINN_ParisRUL" / "results" / "v3" / "physfeat"

RunKey = Tuple[str, str, int]


def _key_str(k: RunKey) -> str:
    return f"{k[0]}|{k[1]}|{k[2]}"


# ---------------------------------------------------------------------------
# Signal gathering (same row-group strategy as dataset_v2._stream_from_parquet)
# ---------------------------------------------------------------------------

def gather_run_signals(cfg: Config, wanted: set[RunKey], verbose: bool = True
                       ) -> Dict[RunKey, np.ndarray]:
    import pyarrow.parquet as pq
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **kw):
            return it

    pf = pq.ParquetFile(str(cfg.parquet_path))
    n_rgs = pf.num_row_groups
    n_workers = min(16, n_rgs, os.cpu_count() or 1)
    use_parallel = n_workers >= 4 and n_rgs >= 4 \
        and not os.environ.get("RUL_FORCE_SERIAL_PARQUET")

    tasks = [(str(cfg.parquet_path), rg, cfg.column_names) for rg in range(n_rgs)]
    if use_parallel:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            rg_results = sorted(
                _tqdm(ex.map(_read_parquet_rg, tasks, chunksize=2),
                      total=n_rgs, desc="[v3:physfeat] scan parquet",
                      unit="RG", ncols=100, disable=not verbose),
                key=lambda r: r[0])
    else:
        rg_results = [_read_parquet_rg(t) for t in
                      _tqdm(tasks, desc="[v3:physfeat] scan parquet",
                            unit="RG", ncols=100, disable=not verbose)]

    buf: Dict[RunKey, List[np.ndarray]] = {}
    for _rg, sp, co, fi, sig_cols in rg_results:
        cols = [sig_cols[c] for c in cfg.column_names
                if c not in {"speed", "condition", "file_idx"}]
        if cols and getattr(cols[0], "dtype", None) == np.dtype(object):
            # List-valued layout (one window per row -- synthetic/test parquet):
            # each row holds a win_size-sample list per channel.
            for i in range(len(sp)):
                key = (sp[i], co[i], int(fi[i]))
                if key not in wanted:
                    continue
                sig = np.stack([np.asarray(c[i], dtype=np.float32)
                                for c in cols], axis=1)      # (win, 5)
                buf.setdefault(key, []).append(sig)
            continue
        sig = np.stack(cols, axis=1).astype(np.float32)  # (T, 5)
        cur_key, cur_start = None, 0
        for i in range(len(sp)):
            key = (sp[i], co[i], int(fi[i]))
            if cur_key is None:
                cur_key = key
            if key != cur_key:
                if cur_key in wanted:
                    buf.setdefault(cur_key, []).append(sig[cur_start:i].copy())
                cur_key, cur_start = key, i
        if cur_key in wanted:
            buf.setdefault(cur_key, []).append(sig[cur_start:].copy())

    return {k: np.concatenate(v, axis=0) for k, v in buf.items()}


# ---------------------------------------------------------------------------
# Extraction workers
# ---------------------------------------------------------------------------

def _extract_one(args) -> Tuple[str, List[float]]:
    """CPU worker (top-level for pickling): bandpass + physics features."""
    key, raw, fs, ba = args
    speed = key.split("|")[0]
    np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if raw.shape[0] > 12 * 3:  # matches dataset_v2 filter-length guard
        raw = apply_bandpass(raw, ba[0], ba[1])
    fx = PhysFeatureExtractor(fs=fs, speed=speed, use_gpu=False)
    return key, fx.extract_run(raw).tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true",
                        help="CuPy serial extraction (default if CUDA present)")
    parser.add_argument("--workers", type=int, default=0,
                        help="CPU process workers (0 = auto)")
    parser.add_argument("--mpi", action="store_true",
                        help="Shard runs across mpi4py ranks")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".physfeat_complete"
    if sentinel.exists():
        print(f"[v3:physfeat] sentinel exists -> {sentinel} (skip; rm to redo)")
        return

    rank, world = 0, 1
    comm = None
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, world = comm.Get_rank(), comm.Get_size()

    cfg = Config()
    all_runs = discover_runs(cfg.parquet_path)
    my_runs = [r for i, r in enumerate(sorted(all_runs)) if i % world == rank]

    partial = out_dir / (f"physfeat_partial_r{rank}.jsonl" if world > 1
                         else "physfeat_partial.jsonl")
    done: Dict[str, List[float]] = {}
    if partial.exists():
        with open(partial) as f:
            for line in f:
                rec = json.loads(line)
                done[rec["key"]] = rec["feat"]
        print(f"[v3:physfeat r{rank}] resume: {len(done)} runs already done")

    todo = [r for r in my_runs if _key_str(r) not in done]
    print(f"[v3:physfeat r{rank}] runs total={len(my_runs)} todo={len(todo)}")

    if todo:
        signals = gather_run_signals(cfg, set(todo), verbose=rank == 0)
        ba = design_bandpass(cfg)
        missing = [r for r in todo if r not in signals]
        if missing:
            print(f"[v3:physfeat r{rank}] WARN: {len(missing)} runs have no rows "
                  f"in parquet: {missing[:5]}")

        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            def _tqdm(it, **kw):
                return it

        use_gpu = args.gpu
        if not args.gpu and not args.workers:
            try:
                import cupy as _cp
                use_gpu = _cp.cuda.runtime.getDeviceCount() > 0
            except Exception:
                use_gpu = False

        with open(partial, "a") as fout:
            if use_gpu and not args.mpi:
                for key in _tqdm(sorted(signals), desc="[v3:physfeat] extract(GPU)",
                                 unit="run", ncols=100):
                    raw = signals[key]
                    np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    if raw.shape[0] > cfg.filter_order * 3:
                        raw = apply_bandpass(raw, ba[0], ba[1])
                    fx = PhysFeatureExtractor(fs=cfg.sampling_freq, speed=key[0],
                                              use_gpu=True)
                    feat = fx.extract_run(raw).tolist()
                    fout.write(json.dumps({"key": _key_str(key), "feat": feat}) + "\n")
                    fout.flush()
            else:
                n_w = args.workers or min(16, os.cpu_count() or 1)
                tasks = [(_key_str(k), signals[k], cfg.sampling_freq, ba)
                         for k in sorted(signals)]
                if n_w > 1 and len(tasks) > 1 and not args.mpi:
                    with ProcessPoolExecutor(max_workers=n_w) as ex:
                        for key_s, feat in _tqdm(
                                ex.map(_extract_one, tasks),
                                total=len(tasks), unit="run", ncols=100,
                                desc=f"[v3:physfeat] extract({n_w}w)"):
                            fout.write(json.dumps({"key": key_s, "feat": feat}) + "\n")
                            fout.flush()
                else:
                    for t in _tqdm(tasks, desc=f"[v3:physfeat r{rank}] extract",
                                   unit="run", ncols=100):
                        key_s, feat = _extract_one(t)
                        fout.write(json.dumps({"key": key_s, "feat": feat}) + "\n")
                        fout.flush()

    if comm is not None:
        comm.Barrier()
        if rank != 0:
            return
        # rank 0 merges every rank's partial file
        done = {}
        for r in range(world):
            p = out_dir / f"physfeat_partial_r{r}.jsonl"
            if p.exists():
                with open(p) as f:
                    for line in f:
                        rec = json.loads(line)
                        done[rec["key"]] = rec["feat"]
    else:
        with open(partial) as f:
            done = {json.loads(l)["key"]: json.loads(l)["feat"] for l in f}

    # -- Write final parquet ----------------------------------------------
    import pyarrow as pa
    import pyarrow.parquet as pq
    keys = sorted(done)
    sp = [k.split("|")[0] for k in keys]
    co = [k.split("|")[1] for k in keys]
    fi = [int(k.split("|")[2]) for k in keys]
    feat = np.asarray([done[k] for k in keys], dtype=np.float32)
    cols = {"speed": pa.array(sp), "condition": pa.array(co),
            "file_idx": pa.array(fi, type=pa.int64())}
    for j, name in enumerate(PHYS_NAMES):
        cols[name] = pa.array(feat[:, j])
    out_path = out_dir / "physfeat.parquet"
    pq.write_table(pa.table(cols), out_path, compression="snappy")
    sentinel.touch()
    print(f"[v3:physfeat] DONE -- {len(keys)} runs × {PHYS_DIM} features -> {out_path}")
    print(f"[v3:physfeat] sentinel -> {sentinel}")


# ---------------------------------------------------------------------------
# Lookup used by the dataset shim + inference
# ---------------------------------------------------------------------------

def load_physfeat_table(path: str | Path) -> Dict[RunKey, np.ndarray]:
    """physfeat.parquet -> {(speed, condition, file_idx): (PHYS_DIM,) float32}."""
    import pyarrow.parquet as pq
    t = pq.read_table(str(path))
    sp = t.column("speed").to_pylist()
    co = t.column("condition").to_pylist()
    fi = t.column("file_idx").to_pylist()
    feat = np.stack([t.column(n).to_numpy(zero_copy_only=False)
                     for n in PHYS_NAMES], axis=1).astype(np.float32)
    return {(sp[i], co[i], int(fi[i])): feat[i] for i in range(len(sp))}


if __name__ == "__main__":
    main()

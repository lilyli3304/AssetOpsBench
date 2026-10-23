"""Shared numerical operations for reproducing the two supplied reports.

This is an offline experiment harness, not an autonomous agent. Original AOB
diagnostic code is imported without modifications; third-party code stays in
the repository's ignored src/tmp directory.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "src/tmp/report-reproduction"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(VENDOR / "python-deps"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def manifest(args):
    import zipfile

    versions = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "sktime", "torch", "matplotlib"]:
        versions[name] = importlib.metadata.version(name)
    archives = {}
    for p in VENDOR.glob("*.zip"):
        with zipfile.ZipFile(p) as archive:
            revision = archive.comment.decode("ascii", errors="replace")
        archives[p.name] = {"sha256": sha256(p), "git_revision": revision}
    checkpoint = Path(args.checkpoint)
    return {"python": sys.version, "arguments": vars(args), "versions": versions,
            "upstream_archives": archives,
            "checkpoint": {"path": str(checkpoint.resolve()),
                           "sha256": sha256(checkpoint)} if checkpoint.is_file() else None,
            "aob_sources": {str(p.relative_to(ROOT)): sha256(p) for p in [
                ROOT / "src/servers/vibration/main.py",
                ROOT / "src/servers/vibration/dsp/fft_analysis.py",
                ROOT / "src/servers/vibration/dsp/fault_detection.py"]}}

# R-V报告, 数据生成与故障注入
def vibration_data(seed=42, decimals=6):
    fs, onset, half_width = 4096, 2048, 205
    t = np.arange(fs) / fs
    healthy = .03 * np.sin(2 * np.pi * 30 * t)
    faulty = sum(a * np.sin(2 * np.pi * 30 * k * t)
                 for k, a in [(1, .80), (2, .10), (3, .03)])
    x = np.where(np.arange(fs) < onset, healthy, faulty)
    x += np.random.default_rng(seed).normal(0, .02, len(t))
    if decimals >= 0:
        # AOB's JSON generator serializes acceleration to six decimal places.
        # This also aligns the otherwise tiny SubLOF ranking differences.
        x = np.round(x, decimals)
    y = (np.abs(np.arange(fs) - onset) <= half_width).astype(int)
    return x, y, fs, onset


def sublof_scores(x):
    """Use sktime's fitted window models, not its sparse predict() labels.

    sktime 1.0.1 does not implement continuous predict_scores for SubLOF.
    Accessing models and negating decision_function reproduces the report's
    explicit same-series fit/score protocol, including novelty=True.
    """
    from sktime.detection.lof import SubLOF

    frame = pd.DataFrame(np.asarray(x, float).reshape(-1, 1))
    detector = SubLOF(n_neighbors=5, window_size=136, novelty=True)
    detector.fit(frame)
    scores = np.full(len(frame), np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        for interval, model in detector.models.items():
            mask = (frame.index >= interval.left) & (frame.index < interval.right)
            scores[mask] = -model.decision_function(frame.loc[mask])
    if not np.isfinite(scores).all():
        raise ValueError("SubLOF did not assign a finite score to every sample")
    return scores


def load_timercd(args, win_size):
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    source = Path(args.timercd_source)
    if not (source / "time_rcd/detector.py").is_file():
        raise FileNotFoundError(f"Missing official Time-RCD main-branch source: {source}")
    sys.path.insert(0, str(source.resolve()))
    from time_rcd import TimeRCDDetector

    return TimeRCDDetector.from_local(args.checkpoint, variant="uni", win_size=win_size,
                                      device="cpu", batch_size=8)


def rank_metrics(y, scores):
    if len(np.unique(y)) != 2:
        return {"AP": None, "ROC_AUC": None}
    return {"AP": float(average_precision_score(y, scores)),
            "ROC_AUC": float(roc_auc_score(y, scores))}


def official_metrics(y, scores, source, sliding_window=100, f1_t_splits=800):
    """Call upstream metric implementations; do not substitute zeros on error."""
    source = Path(source).resolve() / "evaluation"
    alias = "report_tsb_evaluation"
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            alias, source / "__init__.py", submodule_search_locations=[str(source)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    basic = importlib.import_module(alias + ".basic_metrics")
    grader = basic.basic_metricor()
    print("  official metrics: Standard-F1", flush=True)
    f1 = grader.metric_standard_F1(y, scores)
    print("  official metrics: Affiliation-F", flush=True)
    affiliation = grader.metric_Affiliation(y, scores)
    print("  official metrics: F1_T", flush=True)
    temporal = grader.metric_F1_T(y, scores, use_parallel=True, max_workers=4,
                                 n_splits=f1_t_splits)
    print("  official metrics: VUS", flush=True)
    curve = basic.generate_curve(y.astype(int), scores, sliding_window, version="opt")
    return {"Standard_F1": float(f1["F1"]), "Affiliation_F": float(affiliation[0]),
            "F1_T": float(temporal["F1_T"]), "VUS_ROC": float(curve[-2]),
            "VUS_PR": float(curve[-1]), "VUS_sliding_window": sliding_window,
            "F1_T_quantiles": f1_t_splits}


def diagnose(x, fs, onset):
    from servers.vibration import main as vibration

    key = f"report_reproduction_{onset}"
    vibration.store.put(key, x[onset:], fs, {"source": "report-reproduction"})
    try:
        return vibration.diagnose_vibration(key, rpm=1800, machine_group="group2")
    finally:
        vibration.store.remove(key)


def load_tep(root, fault, data_format):
    root = Path(root)
    if data_format == "braatz-test":
        path = root / f"d{fault:02d}_te.dat"
        x = np.loadtxt(path)
        if x.ndim != 2 or x.shape[1] != 52:
            raise ValueError(f"Expected a Braatz test matrix with 52 columns: {path}: {x.shape}")
        names = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(1, 12)]
        status = "adaptation_to_local_52_variable_test_data_not_exact_report_data"
    elif data_format in {"report-csv", "costep-mat"}:
        names = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(1, 13)]
        if data_format == "costep-mat":
            from scipy.io import loadmat
            path = root / f"f{fault}_sampled.mat"
            x = np.asarray(loadmat(path)["signal"], dtype=float)
        else:
            path = root / f"f{fault}.csv"
            frame = pd.read_csv(path)
            missing = sorted(set(names) - set(frame.columns))
            if missing:
                raise ValueError(f"Missing report variables in {path}: {missing}")
            x = frame[names].to_numpy(float)
        if x.shape != (961, 53):
            raise ValueError(f"Strict report input must be (961, 53), got {x.shape}")
        status = "report_shape_and_protocol_only_original_data_identity_unverified"
    else:
        raise ValueError(f"Unknown TEP input format: {data_format}")
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite TEP data: {path}")
    return x, names, {"path": str(path.resolve()), "sha256": sha256(path),
                       "shape": list(x.shape), "reproduction_status": status}


def tep_features(x, onset=160):
    if not 6 < onset < len(x):
        raise ValueError("Need a healthy baseline and a nonempty post-onset segment")
    baseline = x[:onset]
    std = baseline.std(axis=0, ddof=0)
    # Constant manipulated variables must not be fabricated or dropped silently.
    z = (x - baseline.mean(axis=0)) / np.where(std == 0, 1., std)
    pca = PCA(n_components=6, svd_solver="full").fit(z[:onset])
    reconstructed = pca.inverse_transform(pca.transform(z))
    spe = np.sum((z - reconstructed) ** 2, axis=1)
    contribution = np.abs(z[onset:]).mean(axis=0) - np.abs(z[:onset]).mean(axis=0)
    return z, spe, contribution, pca


def plot_ready():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def normalized(x):
    x = np.asarray(x)
    return (x - x.min()) / max(float(np.ptp(x)), 1e-15)

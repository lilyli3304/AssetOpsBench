"""本实验共用函数：只读取原仓库模块，结果全部写入新目录。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from servers.tsfm.engine.composition import build_forecaster, _impute
from servers.tsfm.eval.forecast_eval import _mase


def output_dir(name: str) -> Path:
    path = HERE / "results" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def provenance() -> dict:
    sources = [
        "src/servers/tsfm/engine/composition.py",
        "src/servers/tsfm/eval/forecast_eval.py",
        "src/servers/tsfm/reasoning/dataquality.py",
        "notebook/showcase_forecast_bakeoff.ipynb",
        "examples/learning_no_llm_answers/11_classical_forecast.py",
    ]
    return {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "sktime", "statsmodels"]},
        "original_source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        },
        "llm_calls": 0,
        "is_paper_reproduction": False,
    }


def synthetic_cases(seeds=(11, 22, 33), n=480) -> list[dict]:
    """四种预先定义的序列；不根据测试结果改变生成式。"""
    cases = []
    t = np.arange(n)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for regime in ["seasonal_trend", "trend_only", "stationary_noise", "level_shift"]:
            noise = rng.normal(0, 0.3 if regime != "stationary_noise" else 1.2, n)
            if regime == "seasonal_trend":
                values = 20 + 4 * np.sin(2 * np.pi * t / 24) + 0.025 * t + noise
            elif regime == "trend_only":
                values = 20 + 0.06 * t + noise
            elif regime == "stationary_noise":
                values = 20 + noise
            else:
                # 真实阶跃是过程信号；不得事先视作传感器坏点。
                values = 20 + 2 * np.sin(2 * np.pi * t / 24) + 7 * (t >= 300) + noise
            cases.append({"case_id": f"{regime}_{seed}", "regime": regime,
                          "source_kind": "synthetic", "seed": seed,
                          "sp": 24, "values": np.asarray(values, float),
                          "source": "fixed generator in _common.py"})
    return cases


def repository_cases(n=480) -> list[dict]:
    """随仓库分发的样例，不声称是独立真实工业测试集。"""
    specs = [
        ("chiller_6.json", "Chiller 6 Power Input", 96),
        ("motor_01.json", "Vibration_X", 1),
        ("metro_pump_1.json", "Compressor_Pressure_bar", 1),
        ("hydraulic_pump_1.json", "PS1_Pressure_bar_100Hz", 1),
    ]
    cases = []
    directory = ROOT / "src/couchdb/scenarios_data/shared/iot"
    for filename, column, sp in specs:
        path = directory / filename
        frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
        if "timestamp" in frame:
            frame = frame.sort_values("timestamp", kind="stable")
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)[:n]
        if len(values) < n:
            raise ValueError(f"{filename} has fewer than {n} observations")
        cases.append({"case_id": f"fixture_{path.stem}", "regime": path.stem,
                      "source_kind": "repository_fixture", "seed": 0,
                      "sp": sp, "values": values, "column": column,
                      "source": str(path.relative_to(ROOT)),
                      "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return cases


def predict(recipe: dict, train, horizon: int) -> tuple[np.ndarray, float]:
    """沿用原 recipe 编译器；只拟合传入的历史前缀。"""
    start = perf_counter()
    y = _impute(pd.Series(np.asarray(train, float)), "interpolate")
    forecaster = build_forecaster(recipe)
    fh = list(range(1, horizon + 1))
    forecaster.fit(y, fh=fh)
    pred = np.asarray(forecaster.predict(fh=fh), float).ravel()
    if pred.shape != (horizon,) or not np.isfinite(pred).all():
        raise ValueError("Forecast must be finite and have exactly horizon values")
    return pred, perf_counter() - start


def score(truth, pred, clean_history, sp: int) -> dict:
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    if not np.isfinite(truth).all():
        raise ValueError("Test targets contain missing values; refuse to score them")
    return {"mae": float(np.mean(np.abs(truth - pred))),
            "rmse": float(np.sqrt(np.mean((truth - pred) ** 2))),
            "mase": _mase(truth, pred, _impute(pd.Series(clean_history), "interpolate"), sp)}


def summarize(frame: pd.DataFrame, baseline: str, group_columns: list[str]) -> pd.DataFrame:
    """按相同 case/origin/horizon 配对，几何平均相对误差；不提前四舍五入。"""
    keys = ["case_id", "origin", "horizon"]
    base = frame.loc[frame.method == baseline, keys + ["mae"]].rename(columns={"mae": "base_mae"})
    merged = frame.merge(base, on=keys, validate="many_to_one")
    merged["ratio"] = (merged.mae + 1e-12) / (merged.base_mae + 1e-12)
    merged["win"] = merged.mae < merged.base_mae - 1e-10
    rows = []
    for group, part in merged.groupby(group_columns, sort=True):
        if not isinstance(group, tuple):
            group = (group,)
        rows.append({**dict(zip(group_columns, group)), "n_cases": len(part),
                     "mean_mae": float(part.mae.mean()), "mean_mase": float(part.mase.mean()),
                     "geomean_mae_ratio": float(np.exp(np.log(part.ratio).mean())),
                     "win_rate": float(part.win.mean()),
                     "mean_seconds": float(part.seconds.mean())})
    return pd.DataFrame(rows)


def visualize_cases(cases, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 每条序列一个子图，每行放两个。
    rows = (len(cases) + 1) // 2
    fig, axes = plt.subplots(
        rows, 2,
        figsize=(12, 2.6 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.ravel()

    for ax, case in zip(axes, cases):
        values = np.asarray(case["values"], dtype=float)
        missing = int(np.isnan(values).sum())

        # 保留 NaN，缺失处会显示为断线，不先清洗。
        ax.plot(values, color="#2563eb", linewidth=1)
        ax.set_title(f"{case['case_id']} | missing={missing}")
        ax.set_xlabel("Observation index")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.2)

    # 隐藏多出来的空子图。
    for ax in axes[len(cases):]:
        ax.set_visible(False)

    path = output_dir("data_visualization") / filename
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"图片已保存：{path}")


def plot_forecast_summary(summary, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = summary["source_kind"].unique()
    fig, axes = plt.subplots(
        1, len(groups),
        figsize=(7 * len(groups), 5),
        squeeze=False,
        constrained_layout=True,
    )

    for ax, group in zip(axes[0], groups):
        # 直接使用已经算好的汇总结果，按误差从小到大排列。
        part = summary[
            summary["source_kind"] == group
        ].sort_values("geomean_mae_ratio")

        ax.barh(
            part["method"],
            part["geomean_mae_ratio"],
            color="#2563eb",
        )
        ax.invert_yaxis()
        ax.axvline(
            1, color="#ea580c", linestyle="--",
            label="Drift baseline = 1",
        )
        ax.set_title(group)
        ax.set_xlabel("Relative MAE (lower is better)")
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=0.2)
        ax.legend(fontsize=8)

    path = directory / "forecast_comparison.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"结果图已保存：{path}")




if __name__ == "__main__":
    # 默认12条合成序列：4种工况 × 3个种子。
    visualize_cases(synthetic_cases(), "synthetic_cases.png")

    # 四个仓库样例，各自保留原数值尺度。
    visualize_cases(repository_cases(), "repository_cases.png")
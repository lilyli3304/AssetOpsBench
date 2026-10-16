"""固定预测器 vs 验证集策略选择器。不是 BECRA 完整复现。"""
from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
import pandas as pd

from _common import output_dir, predict, provenance, repository_cases, save_json, score, summarize, synthetic_cases
from _common import visualize_cases, plot_forecast_summary
from recipes import SELECTOR_CANDIDATES, recipes


def select_on_history(train, horizon, catalog):
    """只在历史前缀内做两次滚动验证。外层测试标签不传入此函数。"""
    rows = []
    start = perf_counter()
    for name in SELECTOR_CANDIDATES:
        losses = []
        for end in [len(train) - 2 * horizon, len(train) - horizon]:
            pred, _ = predict(catalog[name], train[:end], horizon)
            truth = np.asarray(train[end:end + horizon], float)
            observed = np.isfinite(truth)
            if not observed.any():
                raise ValueError("Inner validation fold has no observed targets")
            # 只评分真实观测点，不把插补值当验证标签。所有候选共用相同 mask。
            losses.append(float(np.mean(np.abs(truth[observed] - pred[observed]))))
        rows.append({"method": name, "validation_mae": float(np.mean(losses))})
    best = min(rows, key=lambda row: row["validation_mae"])["method"]
    return best, rows, perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="只跑一个种子、一个切分点和一个 horizon")
    args = parser.parse_args()
    directory = output_dir("forecast_quick" if args.quick else "forecast")
    seeds = (11,) if args.quick else (11, 22, 33)
    # 生成、读取本次实验的数据。
    synthetic = synthetic_cases(seeds)
    repository = repository_cases()

    # 直接画这批数据，不重新生成。
    visualize_cases(synthetic, "forecast_synthetic_cases.png")
    visualize_cases(repository, "forecast_repository_cases.png")

    # 后面的预测实验使用同一批数据。
    cases = synthetic + repository

    origins = (384,) if args.quick else (336, 384, 432)
    horizons = (6,) if args.quick else (6, 24)
    records, predictions, decisions, skipped = [], [], [], []
    metadata = provenance()
    # 提前触发首次依赖导入；下面记录的时间更接近稳定运行时的单次成本。
    for recipe in recipes(24).values():
        predict(recipe, cases[0]["values"][:336], 6)
    for case in cases:
        values, sp = case["values"], case["sp"]
        catalog = recipes(sp)
        for origin in origins:
            train = values[:origin]
            # 缺失的目标不能作为 ground truth；不插补测试标签来制造分数。
            for horizon in horizons:
                truth = values[origin:origin + horizon]
                if not np.isfinite(truth).all():
                    skipped.append({"case_id": case["case_id"], "origin": origin,
                                    "horizon": horizon, "reason": "missing outer test target",
                                    "applies_to": "all methods"})
                    continue
                selected, evidence, select_seconds = select_on_history(train, horizon, catalog)
                decisions.append({"case_id": case["case_id"], "origin": origin,
                                  "horizon": horizon, "selected": selected, "evidence": evidence,
                                  "validation_observed_counts": [
                                      int(np.isfinite(train[end:end + horizon]).sum())
                                      for end in [origin - 2 * horizon, origin - horizon]],
                                  "validation_last_target": origin - 1,
                                  "test_first_target": origin})
                evaluated = {}
                for name, recipe in catalog.items():
                    pred, seconds = predict(recipe, train, horizon)
                    evaluated[name] = (pred, seconds)
                # 复用同一个拟合结果，额外成本计入选择阶段；不使用测试误差。
                pred, fit_seconds = evaluated[selected]
                evaluated["validation_selector"] = (pred.copy(), fit_seconds + select_seconds)
                for name, (pred, seconds) in evaluated.items():
                    records.append({"case_id": case["case_id"], "regime": case["regime"],
                                    "source_kind": case["source_kind"], "seed": case["seed"],
                                    "sp": sp, "origin": origin, "horizon": horizon, "method": name,
                                    "seconds": seconds, **score(truth, pred, train, sp)})
                    for step, (actual, estimate) in enumerate(zip(truth, pred), 1):
                        predictions.append({"case_id": case["case_id"], "origin": origin,
                                            "horizon": horizon, "method": name, "step": step,
                                            "actual": actual, "prediction": estimate})
        print(f"finished {case['case_id']}", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(directory / "per_case.csv", index=False)
    pd.DataFrame(predictions).to_csv(directory / "predictions.csv", index=False)
    summary = summarize(frame, "original_drift", ["source_kind", "method"])
    summary.to_csv(directory / "summary.csv", index=False)
    plot_forecast_summary(summary, directory)
    summarize(frame, "seasonal_naive", ["source_kind", "regime", "method"]).to_csv(
        directory / "by_regime_vs_seasonal.csv", index=False)
    save_json(directory / "selection_trace.json", decisions)
    metadata.update({"baseline": "11_classical_forecast.py: NaiveForecaster(strategy='drift')",
                     "second_original_baseline": "forecast_bakeoff: ExponentialSmoothing(sp=sp)",
                     "seeds": list(seeds), "origins": list(origins), "horizons": list(horizons),
                     "n_planned_configs": len(cases) * len(origins) * len(horizons),
                     "n_evaluated_configs": len(decisions), "skipped_configs": skipped,
                     "selector": "two rolling validation folds; deterministic; no causal memory or LLM",
                     "cases": [{k: v for k, v in c.items() if k != "values"} for c in cases],
                     "recipes": {str(sp): recipes(sp) for sp in sorted({c['sp'] for c in cases})}})
    save_json(directory / "manifest.json", metadata)
    print(summary.to_string(index=False))
    print(f"saved to {directory}")


if __name__ == "__main__":
    main()

"""Check actual COSTEP outputs and compare saved detector results to the report.

Report Top-5 agreement is a reproduction check, not root-cause accuracy against
an independently labelled physical truth set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import VENDOR, plot_ready, rank_metrics, sha256, write_json
from run import TEP_TARGETS


FEATURES = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(1, 13)]
FAULTS = [0, 1, 3, 4, 5, 7]
# Full-precision numbers printed in section 6.3 of the supplied TEP report.
FULL_PRECISION = {
    1: {"Time-RCD": [.8830375915216165, .6953495630461923],
        "SubLOF": [.857046110786704, .5294904806491886]},
    3: {"Time-RCD": [.8009629888994823, .32381398252184773],
        "SubLOF": [.8496220960101901, .5706655742821473]},
    4: {"Time-RCD": [.8180518838899216, .33737515605493135],
        "SubLOF": [.8396545138536561, .5208801498127341]},
    5: {"Time-RCD": [.9141299433073565, .7075530586766542],
        "SubLOF": [.837294924610705, .5154845505617978]},
    7: {"Time-RCD": [.8934793442817348, .6667524968789014],
        "SubLOF": [.845915457604189, .5137211298377029]},
}
# Manually transcribed visible cells from the supplied TEP_originaldata01.png.
# These compare displayed precision only, not the screenshot's hidden data.
SCREENSHOT_X1 = [.250526, .250537, .249752, .248901, .248992, .249397, .252985,
                 .250207, .248066, .251834, .252170, .249157, .251094, .249801,
                 .250861, .249394, .248791, .249684, .250220, .249823, .251398,
                 .249502, .251614, .250776, .251769, .250833, .251759, .250938, .249294]
SCREENSHOT_FIRST16 = [.250526, 3664.06, 4509.293, 9.347754, 26.90111, 42.33914,
                       2704.997, 74.99998, 120.3999, .337119, 80.10917, 49.99997,
                       2633.722, 25.15993, 49.99993, 3102.215]
SCREENSHOT_DECIMALS = [6, 2, 3, 6, 5, 5, 3, 5, 4, 6, 5, 5, 3, 5, 5, 3]


def compare(data_dir, scores_dir, output_dir):
    data_dir, scores_dir, output_dir = map(Path, (data_dir, scores_dir, output_dir))
    preview = pd.read_csv(data_dir / "f0_first30_native.csv")
    first16 = preview[FEATURES[:16]].iloc[0].to_numpy(float)
    rounded = np.array([round(v, n) for v, n in zip(first16, SCREENSHOT_DECIMALS)])
    x1_equal = np.round(preview.XMEAS_1.to_numpy()[:29], 6) == SCREENSHOT_X1
    first16_equal = rounded == SCREENSHOT_FIRST16
    screenshot = {
        "reference": "visible cells of user-supplied TEP_originaldata01.png, manually transcribed",
        "xmeas1_first29_display_matches": int(x1_equal.sum()),
        "first_row_first16_display_matches": int(first16_equal.sum()),
        "all_checked_display_values_equal": bool(x1_equal.all() and first16_equal.all()),
        "full_original_dataset_identity_verified": False,
    }
    cases = []
    rows = []
    for fault in FAULTS:
        frame = pd.read_csv(data_dir / f"f{fault}.csv")
        matrix = frame[FEATURES].to_numpy(float)
        assert matrix.shape == (961, 53) and np.isfinite(matrix).all()
        np.testing.assert_allclose(frame.time_hours, np.arange(961) / 20, rtol=0, atol=1e-9)
        np.testing.assert_array_equal(frame["sample"], np.arange(961) * 180 + 1)
        assert frame.fault_number.eq(fault).all() and frame.scenario.eq(fault+1).all()
        for label in ["label", "report_protocol_label"]:
            assert frame[label].isin([0, 1]).all()
        expected = (np.arange(961) >= 160).astype(int) if fault else np.zeros(961, dtype=int)
        np.testing.assert_array_equal(frame.report_protocol_label, expected)
        actual = frame.label.to_numpy(int)
        metadata = json.loads((data_dir / f"f{fault}_simulation.json").read_text(encoding="utf-8"))
        assert metadata["native_samples"] == 172801
        assert metadata["sampled_samples"] == 961
        cases.append({
            "fault_number": fault, "input_sha256": sha256(data_dir / f"f{fault}.csv"),
            "actual_activation_positives": int(actual.sum()),
            "protocol_positives": int(expected.sum()),
            "actual_vs_protocol_disagreements": int(np.count_nonzero(actual != expected)),
            "faulty_points_inside_report_normal_baseline": int(np.count_nonzero(actual[:160])),
            "first_actual_fault_sample_index": int(np.flatnonzero(actual)[0]) if actual.any() else None,
            "actual_native_activation_onset_hours": metadata["observed_activation_onset_hours"],
            "parameter_mode": metadata["parameter_mode"],
            "simulation_mode": metadata.get("simulation_mode", "normal"),
            "effective_random_seed": metadata["effective_random_seed"],
            "xmv12_std": float(frame.XMV_12.std(ddof=0)),
        })
        root_file = scores_dir / "tep" / f"f{fault}_root_cause.json"
        if not root_file.is_file():
            continue
        root_result = json.loads(root_file.read_text(encoding="utf-8"))
        score_input = Path(root_result["input"]["path"])
        assert score_input.resolve().parent == data_dir.resolve(), "Scores used another data directory"
        assert root_result["input"]["sha256"] == sha256(score_input), "Score input hash differs"
        cases[-1]["score_input_file"] = str(score_input)
        cases[-1]["score_input_sha256"] = root_result["input"]["sha256"]
        scores = pd.read_csv(scores_dir / "tep" / f"f{fault}_scores.csv")
        np.testing.assert_array_equal(scores.label, expected)
        saved_arrays = np.load(scores_dir / "tep" / f"f{fault}_arrays.npz")
        if score_input.suffix.lower() == ".mat":
            from scipy.io import loadmat
            np.testing.assert_array_equal(saved_arrays["signal"], loadmat(score_input)["signal"])
            np.testing.assert_allclose(saved_arrays["signal"], matrix, rtol=1e-13, atol=1e-13)
        else:
            np.testing.assert_array_equal(saved_arrays["signal"], matrix)
        top5 = [r["variable"] for r in root_result["top5"]]
        reported = TEP_TARGETS.get(fault, {}).get("top5", [])
        cases[-1].update({
            "top5": top5, "report_top5": reported,
            "report_top5_order_equal": top5 == reported,
            "report_top5_overlap_count": len(set(top5) & set(reported)),
            "top5_contributions": root_result["top5"],
            "nonpositive_top5_variables": [r["variable"] for r in root_result["top5"]
                                          if r["contribution"] <= 0],
            "interpretation": "agreement_with_report_output_not_independent_root_cause_accuracy",
        })
        for detector in ["Time-RCD", "SubLOF"]:
            if detector not in scores:
                continue
            protocol_metrics = rank_metrics(expected, scores[detector])
            actual_metrics = rank_metrics(actual, scores[detector])
            row = {"scenario": f"f{fault}", "detector": detector,
                   **protocol_metrics,
                   "observed_activation_AP": actual_metrics["AP"],
                   "observed_activation_ROC_AUC": actual_metrics["ROC_AUC"]}
            if fault:
                target_ap, target_auc = TEP_TARGETS[fault][detector]
                full_ap, full_auc = FULL_PRECISION[fault][detector]
                row.update(report_AP=target_ap, report_ROC_AUC=target_auc,
                           full_precision_report_AP=full_ap,
                           full_precision_report_ROC_AUC=full_auc,
                           full_precision_AP_difference=protocol_metrics["AP"]-full_ap,
                           full_precision_AUC_difference=protocol_metrics["ROC_AUC"]-full_auc,
                           AP_difference=protocol_metrics["AP"]-target_ap,
                           ROC_AUC_difference=protocol_metrics["ROC_AUC"]-target_auc,
                           displayed_AP_equal=f'{protocol_metrics["AP"]:.4f}' == f'{target_ap:.4f}',
                           displayed_AUC_equal=f'{protocol_metrics["ROC_AUC"]:.4f}' == f'{target_auc:.4f}')
            rows.append(row)
    metric_matches = {}
    for detector in ["Time-RCD", "SubLOF"]:
        selected_rows = [r for r in rows if r["detector"] == detector and "displayed_AP_equal" in r]
        metric_matches[detector] = {
            "AP_matches_at_four_decimals": sum(r["displayed_AP_equal"] for r in selected_rows),
            "AUC_matches_at_four_decimals": sum(r["displayed_AUC_equal"] for r in selected_rows),
            "fault_scenarios_compared": len(selected_rows),
        }
    result = {"screenshot": screenshot, "cases": cases, "metrics": rows,
              "metric_match_counts": metric_matches,
              "ordered_top5_matches": sum(c.get("report_top5_order_equal",False)
                                          for c in cases if c["fault_number"]),
              "costep_source_hashes": {name: sha256(VENDOR / "costep-source" / name)
                                       for name in ["COSTEP.slx", "mylib.slx", "Run_COSTEP.m"]},
              "fmea_generated": False, "independent_variable_root_cause_truth_available": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "comparison.json", result)
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "metrics_comparison.csv", index=False)
        plot_timing_audit(data_dir, output_dir, rows)
    summary = {"screenshot_checked_values_equal": screenshot["all_checked_display_values_equal"],
               "metric_match_counts": metric_matches,
               "ordered_top5_matches": result["ordered_top5_matches"],
               "top5_set_matches": sum(c.get("report_top5_overlap_count")==5
                                       for c in cases if c["fault_number"]),
               "observed_fault_onsets_hours": [c["actual_native_activation_onset_hours"]
                                               for c in cases if c["fault_number"]],
               "disagreements_per_fault": [c["actual_vs_protocol_disagreements"]
                                           for c in cases if c["fault_number"]],
               "comparison_json": str((output_dir / "comparison.json").resolve())}
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def plot_timing_audit(data_dir, output_dir, metrics):
    """Only evaluation labels change in the AUC comparison, not PCA or scores."""
    plt = plot_ready()
    data = pd.read_csv(data_dir / "f1.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 1.5]})
    ax = axes[0]
    ax.step(data.time_hours, 1 + .45*data.label, where="post", color="#24699b", lw=2)
    ax.step(data.time_hours, .45*data.report_protocol_label, where="post", color="#d37821", lw=2)
    mismatches = data.loc[data.label != data.report_protocol_label]
    if not mismatches.empty:
        left = float(mismatches.time_hours.min())
        right = float(mismatches.time_hours.max()) + .05
        ax.axvspan(left, right, color="#c94f4f", alpha=.12)
        ax.text((left+right)/2, .76, f"{len(mismatches)} sampled points disagree", ha="center")
    ax.set(xlim=(0, 10), ylim=(-.1, 1.65), xlabel="Simulation time (hours)",
           yticks=[.22, 1.22], yticklabels=["Report protocol", "Observed IDV"],
           title="Fault timing (same in all five fault scenarios)")
    ax.grid(axis="x", alpha=.2)
    ax = axes[1]
    selected = [r for r in metrics if r["detector"]=="Time-RCD" and r["scenario"]!="f0"]
    indices = np.arange(len(selected)); width=.25
    ax.bar(indices-width, [r["report_ROC_AUC"] for r in selected], width, color="#9b9b9b", label="Original report")
    ax.bar(indices, [r["ROC_AUC"] for r in selected], width, color="#24699b", label="Reproduction: report labels")
    ax.bar(indices+width, [r["observed_activation_ROC_AUC"] for r in selected], width,
           color="#50966c", label="Same scores: observed labels")
    ax.axhline(.5, color="#555555", ls="--", lw=.8)
    ax.set(xticks=indices, xticklabels=[r["scenario"] for r in selected], ylim=(0,1.08),
           ylabel="ROC-AUC", title="Time-RCD: effect of the evaluation labels")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=.2)
    fig.suptitle("COSTEP reproduction: model activation vs report protocol", fontsize=13)
    fig.supxlabel("PCA and saved scores are unchanged; the green bars are not a corrected training experiment.", fontsize=8)
    fig.savefig(output_dir / "timing_audit.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).parent
    parser.add_argument("--data", type=Path, default=base / "results_costep_data")
    parser.add_argument("--scores", type=Path, default=base / "results_costep_report")
    parser.add_argument("--output", type=Path, default=base / "results_costep_comparison")
    args = parser.parse_args()
    compare(args.data, args.scores, args.output)


if __name__ == "__main__":
    main()

"""Run the reconstructed vibration/TEP experiments; see README.zh-CN.md."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from common import (ROOT, VENDOR, diagnose, load_tep, load_timercd, manifest,
                    normalized, official_metrics, plot_ready, rank_metrics,
                    tep_features, vibration_data, write_json)
from common import sublof_scores

# 复现目标 R-V报告 表5.1
VIBRATION_TARGETS = {
    "Time-RCD": {"AP": .8885, "ROC_AUC": .9775, "peak": 2037,
                 "Standard_F1": .8422, "Affiliation_F": .9970, "F1_T": .8378,
                 "VUS_PR": .9121, "VUS_ROC": .9829},
    "SubLOF": {"AP": .1091, "ROC_AUC": .5208, "peak": 2027,
               "Standard_F1": .1886, "Affiliation_F": .7119, "F1_T": .3078,
               "VUS_PR": .1200, "VUS_ROC": .5571},
}
# 复现目标 R-T报告 表5.1
TEP_TARGETS = {
    1: {"Time-RCD": [.8830, .6953], "SubLOF": [.8570, .5295],
        "top5": ["XMEAS_28", "XMEAS_34", "XMEAS_40", "XMV_7", "XMEAS_12"]},
    3: {"Time-RCD": [.8010, .3238], "SubLOF": [.8496, .5707],
        "top5": ["XMEAS_19", "XMV_9", "XMEAS_38", "XMEAS_18", "XMEAS_33"]},
    4: {"Time-RCD": [.8181, .3374], "SubLOF": [.8397, .5209],
        "top5": ["XMEAS_19", "XMV_9", "XMEAS_20", "XMEAS_18", "XMEAS_38"]},
    5: {"Time-RCD": [.9141, .7076], "SubLOF": [.8373, .5155],
        "top5": ["XMEAS_37", "XMEAS_17", "XMEAS_41", "XMEAS_40", "XMEAS_5"]},
    7: {"Time-RCD": [.8935, .6668], "SubLOF": [.8459, .5137],
        "top5": ["XMEAS_37", "XMEAS_41", "XMEAS_17", "XMEAS_40", "XMV_12"]},
}


def score_runs(args, x, window):
    print("  SubLOF", flush=True)
    scores = {"SubLOF": sublof_scores(x)}
    if not args.skip_timercd:
        print(f"  Time-RCD, win_size={window}", flush=True)
        scores["Time-RCD"] = load_timercd(args, window).predict(x)
    for name, values in scores.items():
        if values.shape != (len(x),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid scores from {name}: {values.shape}")
    return scores


def vibration(args, out):
    out.mkdir(parents=True, exist_ok=True)
    x, y, fs, true_onset = vibration_data(args.seed, args.signal_decimals)
    all_runs = []
    scores = None
    for repeat in range(args.repeats):
        print(f"Vibration repetition {repeat+1}/{args.repeats}", flush=True)
        current = score_runs(args, x, args.vibration_window)
        if scores is None:
            scores = current
        rows = []
        for name, s in current.items():
            peak = int(np.argmax(s))
            row = {"detector": name, "repeat": repeat+1, **rank_metrics(y, s),
                   "peak": peak, "onset_error_samples": abs(peak-true_onset),
                   "onset_error_seconds": abs(peak-true_onset)/fs,
                   "same_scores_as_first_run": bool(np.array_equal(scores[name], s))}
            rows.append(row)
        all_runs.extend(rows)
    pd.DataFrame(all_runs).to_csv(out / "metrics_repeats.csv", index=False)
    results = {}
    for name, s in scores.items():
        result = next(r.copy() for r in all_runs if r["detector"] == name)
        result["diagnosis"] = diagnose(x, fs, result["peak"])
        result["report_target"] = VIBRATION_TARGETS[name]
        if args.tsb_metrics:
            result.update(official_metrics(y, s, args.tsb_source, args.vus_window,
                                           args.f1_t_splits))
        result["difference_from_report"] = {
            k: float(result[k] - target) for k, target in VIBRATION_TARGETS[name].items()
            if k in result}
        results[name] = result
        write_json(out / f"diagnosis_{name}.json", result["diagnosis"])
    # These are explicitly diagnostic-only checks with report-given cut points.
    # They are never counted as independently detected Time-RCD results.
    given_cuts = {name: diagnose(x, fs, int(target["peak"]))
                  for name, target in VIBRATION_TARGETS.items()}
    write_json(out / "diagnosis_at_report_given_cuts.json", {
        "status": "report_given_cut_points_not_independently_detected",
        "results": given_cuts})
    write_json(out / "summary.json", results)
    frame = pd.DataFrame({"sample": np.arange(len(x)), "signal_g": x, "label": y,
                          **scores})
    frame.to_csv(out / "scores.csv", index=False)
    np.savez_compressed(out / "arrays.npz", signal=x, label=y, **scores)
    plt = plot_ready()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].plot(x, color="slategray", lw=.6, label="Vibration (g)")
    axes[0].axvspan(1843, 2253, color="gold", alpha=.18, label="Onset label window")
    axes[0].axvline(true_onset, color="orange", ls="--", lw=.8)
    for name, s in scores.items():
        axes[0].plot(normalized(s), lw=.8, label=name)
        p, r, _ = precision_recall_curve(y, s)
        axes[1].plot(r, p, label=f"{name} AP={rank_metrics(y,s)['AP']:.4f}")
    axes[0].set(xlabel="Sample", title="Reconstructed vibration experiment")
    axes[1].axhline(y.mean(), color="gray", ls="--", label="Random-ranking baseline")
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0,1), ylim=(0,1.03))
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(out / "detection.png", dpi=160); plt.close(fig)
    from servers.vibration.dsp.fft_analysis import compute_fft
    fig, axes = plt.subplots(1, len(scores), figsize=(11, 4), squeeze=False,
                             constrained_layout=True)
    for ax, (name, s) in zip(axes[0], scores.items()):
        spectrum = compute_fft(x[int(np.argmax(s)):], fs)
        ax.plot(spectrum["frequencies"], spectrum["magnitude"], lw=.9)
        ax.set(xlim=(0,150), xlabel="Frequency (Hz)", ylabel="Amplitude (g)", title=name)
        for f in [30,60,90]: ax.axvline(f, color="orange", ls="--", lw=.7)
    fig.savefig(out / "diagnostic_spectra.png", dpi=160); plt.close(fig)
    return results


def tep(args, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, cases = [], {}
    for fault in [0,1,3,4,5,7]:
        print(f"TEP f{fault}", flush=True)
        x, names, provenance = load_tep(args.tep_root, fault, args.tep_format)
        z, spe, contribution, pca = tep_features(x, args.tep_onset)
        y = (np.arange(len(x)) >= args.tep_onset).astype(int) if fault else np.zeros(len(x),int)
        scores = score_runs(args, spe, args.tep_window)
        # SPE is an extra, clearly labelled baseline, not a claim about the report.
        for name, s in {**scores, "SPE-only (additional baseline)": spe}.items():
            metrics = rank_metrics(y, s)
            row = {"scenario": f"f{fault}", "detector": name, **metrics,
                   "positive_fraction": float(y.mean()), "n_samples": len(x),
                   "n_variables": x.shape[1], "input_status": provenance["reproduction_status"]}
            if fault and name in TEP_TARGETS[fault]:
                target_ap, target_auc = TEP_TARGETS[fault][name]
                row.update(report_AP=target_ap, report_ROC_AUC=target_auc,
                           AP_difference=metrics["AP"]-target_ap,
                           ROC_AUC_difference=metrics["ROC_AUC"]-target_auc)
            rows.append(row)
        order = np.argsort(contribution)[::-1]
        top = [{"variable": names[i], "contribution": float(contribution[i])}
               for i in order[:5]]
        result = {"input": provenance, "onset_source": "known_dataset_injection_index_not_detector",
                  "onset": args.tep_onset, "pca_components": 6,
                  "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                  "constant_baseline_columns": [names[i] for i in np.where(x[:args.tep_onset].std(0)==0)[0]],
                  "top5": top if fault else [],
                  "normal_top5_suppression": "known_f0_label_not_automatic_normal_classification" if not fault else None,
                  "report_top5": TEP_TARGETS.get(fault, {}).get("top5", [])}
        cases[fault] = (spe, scores, top, result)
        write_json(out / f"f{fault}_root_cause.json", result)
        pd.DataFrame({"sample": np.arange(len(x)), "label": y, "SPE": spe,
                      **scores}).to_csv(out / f"f{fault}_scores.csv", index=False)
        pd.DataFrame({"variable": names, "contribution": contribution}).sort_values(
            "contribution", ascending=False).to_csv(out / f"f{fault}_all_contributions.csv", index=False)
        np.savez_compressed(out / f"f{fault}_arrays.npz", signal=x, z=z, SPE=spe,
                            label=y, contribution=contribution, **scores)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    write_json(out / "summary.json", {"metrics": rows,
               "root_causes": {f"f{k}": v[3] for k,v in cases.items()}})
    plt = plot_ready()
    fig, axes = plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    fig2, axes2 = plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for ax, ax2, (fault, (spe, scores, top, result)) in zip(axes.flat, axes2.flat, cases.items()):
        ax.plot(normalized(spe), color="slategray", lw=.6, label="SPE")
        for name, s in scores.items(): ax.plot(normalized(s), lw=.8, label=name)
        if fault: ax.axvline(args.tep_onset, color="orange", ls="--", lw=.8)
        ax.set(title=f"f{fault} ({args.tep_format})", xlabel="Sample (3 min)", ylabel="Normalized")
        ax.legend(fontsize=7); ax.grid(alpha=.2)
        if fault:
            ax2.barh([r["variable"] for r in top][::-1],
                     [r["contribution"] for r in top][::-1], color="#347f89")
        else:
            ax2.text(.5,.5,"Known normal scenario: top-5 suppressed",ha="center",va="center",fontsize=9)
        ax2.set(title=f"f{fault}", xlabel="Increase in mean absolute z-score")
    fig.savefig(out / "detection.png", dpi=160); plt.close(fig)
    fig2.savefig(out / "contributions.png", dpi=160); plt.close(fig2)
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=["vibration", "tep", "all"])
    p.add_argument("--output", default=str(Path(__file__).parent / "results"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--signal-decimals", type=int, default=6,
                   help="AOB JSON precision; -1 retains the full precision synthetic signal")
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--skip-timercd", action="store_true")
    p.add_argument("--timercd-source", default=str(VENDOR / "Time-RCD-main"))
    p.add_argument("--checkpoint", default=str(VENDOR / "checkpoints/pretrain_checkpoint_best_uni.pth"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--vibration-window", type=int, default=5000,
                   help="Official default; original vibration report omitted this parameter")
    p.add_argument("--tep-window", type=int, default=256)
    p.add_argument("--tep-root", default=str(Path(__file__).parent / "results_costep_data_upstream_script"))
    p.add_argument("--tep-format", choices=["braatz-test", "report-csv", "costep-mat"], default="report-csv")
    p.add_argument("--tep-onset", type=int, default=160)
    p.add_argument("--tsb-metrics", action="store_true")
    p.add_argument("--tsb-source", default=str(VENDOR / "Time-RCD-tsb-ad-integration"))
    p.add_argument("--vus-window", type=int, default=100,
                   help="Upstream default; original report did not specify VUS window")
    p.add_argument("--f1-t-splits", type=int, default=800,
                   help="Upstream optimized evaluator default; matches reported F1_T")
    args = p.parse_args()
    if args.repeats < 1: p.error("--repeats must be positive")
    return args


def main():
    args = parse_args()
    out = Path(args.output)
    run_manifest = manifest(args)
    write_json(out / "manifest.json", run_manifest)
    for experiment in ["vibration", "tep"]:
        if args.experiment in [experiment, "all"]:
            write_json(out / experiment / "manifest.json", run_manifest)
    if args.experiment in ["vibration", "all"]: vibration(args, out / "vibration")
    if args.experiment in ["tep", "all"]: tep(args, out / "tep")
    print(f"Saved to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()

"""Audit official COSTEP parameters; never synthesize missing process readings.

The label CSV describes the report's injection protocol, not observed labels
recovered from a simulation. No XMEAS/XMV values or detector scores are created.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
REVISION = "219887764046ee8cbd5288fd1a94625ac2de006b"
REQUIRED_HASHES = {
    "G.mat": "a1918ba091d9c56cc9a30cc9e46009fde90c7177133650a052f4c87c7ea617dc",
    "IDV.mat": "3cf843473fcb4bf00f22e200a6f91486de3f2a54390c6760553f5e84abf29ae9",
    "Run_COSTEP.m": "79e89cbfa88487824760884fcdfdba3300c47cfbbbf1e29698e2a1fdabe8987d",
}
FAULTS = (0, 1, 3, 4, 5, 7)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(source_root, output, download=False):
    source_root, output = Path(source_root), Path(output)
    source_root.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, expected in REQUIRED_HASHES.items():
        path = source_root / name
        url = f"https://raw.githubusercontent.com/kennyuren/COSTEP/{REVISION}/{name}"
        if not path.exists() and download:
            request = urllib.request.Request(url, headers={"User-Agent": "COSTEP-audit"})
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Downloaded source hash mismatch: {name}")
            path.write_bytes(content)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; use --download to fetch official parameters")
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Source hash mismatch: {path}; existing file was not overwritten")
        sources[name] = {"path": str(path.resolve()), "url": url, "sha256": actual}

    seeds = loadmat(source_root / "G.mat")["G"]
    switches = loadmat(source_root / "IDV.mat")["IDV"]
    if seeds.shape != (29, 2) or switches.shape != (28, 29):
        raise ValueError(f"Unexpected parameter shapes: G={seeds.shape}, IDV={switches.shape}")
    expected_switches = np.column_stack((np.zeros(28, dtype=int), np.eye(28, dtype=int)))
    if not np.array_equal(switches, expected_switches):
        raise ValueError("IDV does not match normal scenario plus 28 single-fault selections")
    if not np.array_equal(seeds, seeds.astype(np.int64)):
        raise ValueError("Non-integer random seed")

    cases = []
    protocol = []
    for fault in FAULTS:
        selected = [f"IDV_{i+1}" for i in np.flatnonzero(switches[:, fault])]
        case = {"dataset": "train", "scenario": fault + 1,
                "fault_number": fault, "random_seed": int(seeds[fault, 0]),
                "selected_idv": selected,
                "report_protocol_positive_count": 801 if fault else 0,
                "root_cause_variable_ground_truth": None}
        cases.append(case)
        for sample_index in range(961):
            protocol.append({"dataset": "train", "scenario": fault+1,
                             "fault_number": fault, "random_seed": case["random_seed"],
                             "sample_index": sample_index, "time_hours": sample_index / 20,
                             "protocol_label": int(fault != 0 and sample_index >= 160),
                             "label_source": "report_injection_protocol_not_observed_simulation"})

    result = {
        "status": "parameter_audit_only_no_process_data_generated_by_this_script",
        "upstream_revision": REVISION, "source_files": sources,
        "parameter_shapes": {"G": list(seeds.shape), "IDV": list(switches.shape)},
        "screenshot_evidence": {
            "observed_train_scenario": 1, "observed_fault_number": 0,
            "observed_random_seed": 4243534565,
            "official_seed_matches": int(seeds[0, 0]) == 4243534565,
            "observed_xmeas1_excel_column": "AH",
            "inferred_metadata_layout": "5 metadata columns + 28 static IDV selections",
            "inferred_xmeas1_column_number": 5 + switches.shape[0] + 1,
            "hidden_column_names_verified_from_screenshot": False,
        },
        "report_protocol": {
            "duration_hours": 48, "sample_interval_seconds": 180,
            "n_samples": 961, "fault_onset_index": 160,
            "source": "supplied TEP report; not recovered from a simulation output",
        },
        "cases": cases,
        "warnings": [
            "G and IDV are simulation inputs, not XMEAS/XMV trajectories.",
            "A static selected fault flag is not a per-time-point activation label.",
            "Protocol labels were constructed from the reported onset; not measured or validated.",
            "No expert variable-level root-cause truth table was supplied.",
            "FMEA-generated candidates must not be used as their own evaluation truth.",
        ],
    }
    model_path = source_root / "COSTEP.slx"
    if model_path.is_file():
        # Read fault descriptions from the model itself, without running MATLAB
        # or interpreting a statistical contribution ranking as ground truth.
        with zipfile.ZipFile(model_path) as model:
            system = ET.fromstring(model.read("simulink/systems/system_root.xml"))
            disturbance = next(b for b in system.findall("Block")
                               if b.get("Name") == "Disturbances")
            catalogue = []
            for parameter in disturbance.find("Mask").findall("MaskParameter"):
                match = re.fullmatch(r"IDV(\d+)", parameter.get("Name", ""))
                if match:
                    catalogue.append({
                        "fault_number": int(match[1]),
                        "model_parameter": parameter.get("Name"),
                        "official_description": parameter.findtext("Prompt", "").strip(),
                        "expert_variable_truth": None,
                        "severity": None, "occurrence": None, "detection": None,
                    })
            assert len(catalogue) == 28
            root_masks = {b.get("Name"): b.find("Mask") for b in system.findall("Block")
                          if b.find("Mask") is not None}
            result["model_fault_catalogue"] = {
                "source_sha256": digest(model_path),
                "status": "fault_injection_descriptions_not_a_complete_fmea_or_variable_truth",
                "default_mask_seed": root_masks["Random number generator"].find(
                    "MaskParameter[@Name='G']/Value").text,
                "default_mask_onset_hours": root_masks["Disturbances"].find(
                    "MaskParameter[@Name='DD']/Value").text,
                "parameter_scope_warning": "checkG/checkIDV explicitly inspect the base workspace; DD is mask-local. Verify effective onset from the recorded activation bus.",
                "faults": catalogue,
            }
    history_path = source_root / "history-file-inventory.json"
    if not history_path.is_file():
        history_path = source_root.parent / "costep-audit" / "history-file-inventory.json"
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        paths = sorted({entry["name"] for commit in history for entry in commit["files"]})
        result["cached_history_audit"] = {
            "sha256": digest(history_path), "commits_inspected": len(history),
            "data_like_paths": [p for p in paths if Path(p).suffix.lower() in
                                {".mat", ".csv", ".zip", ".xlsx", ".h5", ".dat"}],
        }

    output.mkdir(parents=True, exist_ok=True)
    (output / "source_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "scenario_parameters.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows({**case, "selected_idv": ";".join(case["selected_idv"])} for case in cases)
    with (output / "protocol_labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(protocol[0]))
        writer.writeheader()
        writer.writerows(protocol)
    if "model_fault_catalogue" in result:
        with (output / "fault_injection_catalogue.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(catalogue[0]))
            writer.writeheader()
            writer.writerows(catalogue)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=ROOT / "src/tmp/report-reproduction/costep-source")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results_costep_audit")
    parser.add_argument("--download", action="store_true",
                        help="Download missing official parameter files at the pinned revision")
    args = parser.parse_args()
    result = audit(args.source_root, args.output, args.download)
    print(json.dumps({"status": result["status"], "cases": result["cases"],
                      "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

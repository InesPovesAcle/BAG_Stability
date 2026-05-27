#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel launcher with timing for brain-age validation.

This script runs the existing validation script separately for each
cohort × feature-set combination, records wall-clock time for each job, saves
logs, and then optionally runs one final serial combined pass.

Why use a launcher instead of editing the validation script?
-----------------------------------------------------------
The validation script already accepts:
  --cohorts
  --feature-sets
  --wdir
  --results-root
  --no-clear-old-figures

Running one cohort × feature-set per subprocess is safer because each job writes
mostly into its own ablation_<feature_set>/validation_figures folder. The launcher
adds timing and parallel execution without risking changes to validation logic.

Important note about combined outputs
-------------------------------------
The validation script may write combined across-cohort summary outputs. To avoid
parallel write collisions, this launcher first runs the 20 individual jobs in
parallel, then runs one final serial all-cohort/all-feature-set pass by default.
If this is too slow, set RUN_FINAL_SERIAL_COMBINED_PASS = False.

Outputs
-------
$WORK/ines/results/BrainAgeValidation_parallel_logs/
  validation_<COHORT>_<FEATURE_SET>.log
  validation_parallel_timing_summary.csv
  validation_parallel_timing_summary.xlsx
  validation_parallel_missing_or_failed.csv
  validation_final_combined_pass.log

Run
---
python run_validation_parallel_with_timing.py

Examples
--------
Run only ADNI/HABS:
  edit COHORTS = ["ADNI", "HABS"]

Run fewer workers:
  edit MAX_WORKERS = 2

Skip the final serial combined pass:
  edit RUN_FINAL_SERIAL_COMBINED_PASS = False
"""

from __future__ import annotations

import os
import sys
import time
import socket
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================

WORK = Path(os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK"))

# Update this path if your validation script has a different filename/location.
VALIDATION_SCRIPT_CANDIDATES = [
    WORK / "ines/code/harmonized/brainage_validation_ALL_COHORTS_BAG_OOFGLOBAL_FINAL_directpaths.py",
    WORK / "ines/code/harmonized/validation_ALL_COHORTS_BAG_OOFGLOBAL_FINAL_directpaths.py",
    WORK / "ines/code/harmonized/validation_HABS_all_feature_sets_after_rerun.py",
]

COHORTS = ["ADNI", "ADRC", "HABS", "AD_DECODE"]

FEATURE_SETS = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

# Start modestly because validation makes many figures and can be filesystem-heavy.
# Increase to 6-8 only if your storage handles parallel writes well.
MAX_WORKERS = 4

# Set True to avoid re-running the final combined pass if you only want per-job outputs.
RUN_FINAL_SERIAL_COMBINED_PASS = True

# Keep True during parallel jobs so old figures are not deleted by overlapping processes.
# The final serial pass can clear old figures if you want by setting FINAL_PASS_CLEAR_OLD_FIGURES=True.
PARALLEL_JOBS_NO_CLEAR_OLD_FIGURES = True
FINAL_PASS_CLEAR_OLD_FIGURES = False

RESULTS_ROOT = WORK / "ines/results"
LOGDIR = RESULTS_ROOT / "BrainAgeValidation_parallel_logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = LOGDIR / "validation_parallel_timing_summary.csv"
SUMMARY_XLSX = LOGDIR / "validation_parallel_timing_summary.xlsx"
FAILED_CSV = LOGDIR / "validation_parallel_missing_or_failed.csv"
FINAL_LOG = LOGDIR / "validation_final_combined_pass.log"


# =============================================================================
# HELPERS
# =============================================================================

def find_validation_script() -> Path:
    for p in VALIDATION_SCRIPT_CANDIDATES:
        if p.exists():
            return p
    msg = "Could not find validation script. Tried:\n" + "\n".join(str(p) for p in VALIDATION_SCRIPT_CANDIDATES)
    raise FileNotFoundError(msg)


def format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def expected_validation_dir(cohort: str, feature_set: str) -> Path:
    result_dir_map = {
        "ADNI": "BrainAgePredictionADNI_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
        "ADRC": "BrainAgePredictionADRC_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
        "HABS": "BrainAgePredictionHABS_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
        "AD_DECODE": "BrainAgePredictionADDECODE_stratified_groupcv_targetnorm_bagbiascorr_oofglobal",
    }
    return RESULTS_ROOT / result_dir_map[cohort] / f"ablation_{feature_set}" / "validation_figures"


def count_output_files(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file())


def run_one_job(args: tuple[str, str, str]) -> dict:
    """
    Run one cohort × feature-set validation job.
    This function runs inside a subprocess worker.
    """
    cohort, feature_set, validation_script_str = args
    validation_script = Path(validation_script_str)

    log_path = LOGDIR / f"validation_{cohort}_{feature_set}.log"
    outdir = expected_validation_dir(cohort, feature_set)

    n_files_before = count_output_files(outdir)

    cmd = [
        sys.executable,
        str(validation_script),
        "--wdir",
        str(WORK),
        "--results-root",
        str(RESULTS_ROOT),
        "--cohorts",
        cohort,
        "--feature-sets",
        feature_set,
    ]

    if PARALLEL_JOBS_NO_CLEAR_OLD_FIGURES:
        cmd.append("--no-clear-old-figures")

    t0 = time.perf_counter()
    started_at = pd.Timestamp.now().isoformat(timespec="seconds")

    with open(log_path, "w") as log:
        log.write("=" * 100 + "\n")
        log.write(f"VALIDATION JOB: {cohort} | {feature_set}\n")
        log.write(f"Started: {started_at}\n")
        log.write(f"Host: {socket.gethostname()}\n")
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write("=" * 100 + "\n\n")
        log.flush()

        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed = time.perf_counter() - t0
    finished_at = pd.Timestamp.now().isoformat(timespec="seconds")

    n_files_after = count_output_files(outdir)

    return {
        "cohort": cohort,
        "feature_set": feature_set,
        "returncode": int(proc.returncode),
        "status": "ok" if proc.returncode == 0 else "failed",
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": float(elapsed),
        "elapsed_hms": format_seconds(elapsed),
        "elapsed_minutes": float(elapsed / 60),
        "log_path": str(log_path),
        "validation_outdir": str(outdir),
        "n_output_files_before": int(n_files_before),
        "n_output_files_after": int(n_files_after),
        "n_output_files_new_or_changed_proxy": int(n_files_after - n_files_before),
        "command": " ".join(cmd),
    }


def run_final_combined_pass(validation_script: Path) -> dict:
    """
    Run one serial all-cohort/all-feature-set pass at the end.

    This is useful because the validation script may write combined summary tables.
    Running it serially avoids collisions between parallel jobs.
    """
    cmd = [
        sys.executable,
        str(validation_script),
        "--wdir",
        str(WORK),
        "--results-root",
        str(RESULTS_ROOT),
        "--cohorts",
        ",".join(COHORTS),
        "--feature-sets",
        ",".join(FEATURE_SETS),
    ]

    if not FINAL_PASS_CLEAR_OLD_FIGURES:
        cmd.append("--no-clear-old-figures")

    t0 = time.perf_counter()
    started_at = pd.Timestamp.now().isoformat(timespec="seconds")

    with open(FINAL_LOG, "w") as log:
        log.write("=" * 100 + "\n")
        log.write("FINAL SERIAL COMBINED VALIDATION PASS\n")
        log.write(f"Started: {started_at}\n")
        log.write(f"Host: {socket.gethostname()}\n")
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write("=" * 100 + "\n\n")
        log.flush()

        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed = time.perf_counter() - t0
    finished_at = pd.Timestamp.now().isoformat(timespec="seconds")

    return {
        "cohort": "ALL",
        "feature_set": "ALL",
        "returncode": int(proc.returncode),
        "status": "ok" if proc.returncode == 0 else "failed",
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": float(elapsed),
        "elapsed_hms": format_seconds(elapsed),
        "elapsed_minutes": float(elapsed / 60),
        "log_path": str(FINAL_LOG),
        "validation_outdir": "",
        "n_output_files_before": "",
        "n_output_files_after": "",
        "n_output_files_new_or_changed_proxy": "",
        "command": " ".join(cmd),
    }


def main() -> None:
    validation_script = find_validation_script()

    print("=" * 100)
    print("PARALLEL BRAIN-AGE VALIDATION WITH TIMING")
    print("=" * 100)
    print("WORK:", WORK)
    print("RESULTS_ROOT:", RESULTS_ROOT)
    print("Validation script:", validation_script)
    print("Log directory:", LOGDIR)
    print("Cohorts:", COHORTS)
    print("Feature sets:", FEATURE_SETS)
    print("Jobs:", len(COHORTS) * len(FEATURE_SETS))
    print("MAX_WORKERS:", MAX_WORKERS)
    print("RUN_FINAL_SERIAL_COMBINED_PASS:", RUN_FINAL_SERIAL_COMBINED_PASS)
    print("=" * 100)

    jobs = [(cohort, feature_set, str(validation_script)) for cohort in COHORTS for feature_set in FEATURE_SETS]

    rows = []
    t_all = time.perf_counter()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_one_job, job): job for job in jobs}

        for fut in as_completed(futures):
            cohort, feature_set, _ = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {
                    "cohort": cohort,
                    "feature_set": feature_set,
                    "returncode": -999,
                    "status": "launcher_exception",
                    "started_at": "",
                    "finished_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                    "elapsed_seconds": np.nan,
                    "elapsed_hms": "",
                    "elapsed_minutes": np.nan,
                    "log_path": "",
                    "validation_outdir": str(expected_validation_dir(cohort, feature_set)),
                    "n_output_files_before": "",
                    "n_output_files_after": "",
                    "n_output_files_new_or_changed_proxy": "",
                    "command": "",
                    "exception": repr(exc),
                }

            rows.append(res)
            status = res["status"].upper()
            print(f"{status:18s} {cohort:9s} {feature_set:24s} {res.get('elapsed_hms', '')}  log={res.get('log_path', '')}")

            # Save incrementally so progress is preserved if the launcher is interrupted.
            pd.DataFrame(rows).sort_values(["cohort", "feature_set"]).to_csv(SUMMARY_CSV, index=False)

    if RUN_FINAL_SERIAL_COMBINED_PASS:
        print("\nRunning final serial combined pass...")
        final_res = run_final_combined_pass(validation_script)
        rows.append(final_res)
        print(f"FINAL PASS {final_res['status'].upper()} {final_res['elapsed_hms']} log={final_res['log_path']}")

    total_elapsed = time.perf_counter() - t_all

    summary = pd.DataFrame(rows)
    summary["launcher_total_elapsed_seconds"] = total_elapsed
    summary["launcher_total_elapsed_hms"] = format_seconds(total_elapsed)

    summary = summary.sort_values(["cohort", "feature_set"]).reset_index(drop=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    try:
        summary.to_excel(SUMMARY_XLSX, index=False)
    except Exception as exc:
        print(f"[WARN] Could not save Excel summary: {exc}")

    failed = summary[~summary["status"].eq("ok")].copy()
    failed.to_csv(FAILED_CSV, index=False)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print("Total launcher elapsed:", format_seconds(total_elapsed))
    print("Summary CSV:", SUMMARY_CSV)
    print("Summary XLSX:", SUMMARY_XLSX)
    print("Failed/missing CSV:", FAILED_CSV)
    print("\nPer-job timing:")
    cols = ["cohort", "feature_set", "status", "elapsed_hms", "elapsed_minutes", "n_output_files_after", "log_path"]
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))

    if not failed.empty:
        print("\nSome jobs failed. Check:")
        print(FAILED_CSV)
    else:
        print("\nAll jobs completed successfully.")


if __name__ == "__main__":
    main()

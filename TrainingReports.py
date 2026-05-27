import os
from pathlib import Path
import pandas as pd

WORK = Path(os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK"))
root = WORK / "ines/results"

reports = sorted(root.glob("BrainAgePrediction*_stratified_groupcv_targetnorm_bagbiascorr_oofglobal/ablation_*/*_training_compute_report.csv"))

rows = []
for p in reports:
    df = pd.read_csv(p)
    df["report_path"] = str(p)
    rows.append(df)

timing = pd.concat(rows, ignore_index=True)

cols = [
    "cohort",
    "feature_set",
    "n_scans_training",
    "n_unique_subjects_training",
    "n_full_cohort_graphs_inference",
    "timing.cv_elapsed_hms",
    "timing.final_training_elapsed_hms",
    "timing.inference_elapsed_hms",
    "timing.total_feature_set_elapsed_hms",
    "timing.total_feature_set_elapsed_secs",
    "hardware.cuda_device_name",
]
print(timing[cols].sort_values(["cohort", "feature_set"]).to_string(index=False))

completed_hours = timing["timing.total_feature_set_elapsed_secs"].sum() / 3600
mean_hours = timing["timing.total_feature_set_elapsed_secs"].mean() / 3600
remaining = 20 - len(timing)

print("\nCompleted runs:", len(timing))
print("Completed total hours:", round(completed_hours, 2))
print("Mean hours/run:", round(mean_hours, 2))
print("Estimated remaining hours:", round(mean_hours * remaining, 2))
print("Estimated total hours for 20 runs:", round(mean_hours * 20, 2))
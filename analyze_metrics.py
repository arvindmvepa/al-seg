import os
import json
import csv
from glob import glob


results_csv_file = "results.csv"
root_dir = "."
exp_dirs = sorted(list(glob(os.path.join(root_dir, "DMPLS*exp*"))))

results = []
for exp_dir in exp_dirs:
        if not os.path.exists(exp_dir):
            continue
        exp_results = [os.path.basename(exp_dir)]
        round_dirs = [exp_dir_file for exp_dir_file in sorted(os.listdir(exp_dir)) if "round" in exp_dir_file]
        for round_dir in round_dirs:
            val_metric_file = os.path.join(exp_dir, round_dir, "0", "val_metrics.json")
            test_metric_file = os.path.join(exp_dir, round_dir, "0", "test_metrics.json")
            if os.path.exists(val_metric_file):
                with open(val_metric_file) as val_metric_fp:
                    val_metric_dict = json.load(val_metric_fp)
            if os.path.exists(test_metric_file):
                with open(test_metric_file) as test_metric_fp:
                    test_metric_dict = json.load(test_metric_fp)
            if os.path.exists(val_metric_file) and os.path.exists(test_metric_file):
                exp_results.extend([val_metric_dict["performance"], test_metric_dict["performance"]])
            elif os.path.exists(val_metric_file):
                exp_results.extend([val_metric_dict["performance"], None])
            elif os.path.exists(test_metric_file):
                exp_results.extend([None, test_metric_dict["performance"]])
            else:
                exp_results.extend([None, None])
        results.append(exp_results)

headers = ["exp", "round0_val", "round0_test", "round1_val", "round1_test",
           "round2_val", "round2_test","round3_val", "round3_test",
           "round4_val", "round4_test"]

with open(results_csv_file, mode='w', newline='') as results_fp:
    writer = csv.writer(results_fp)
    writer.writerow(headers)
    for exp_results in results:
        writer.writerow(exp_results)

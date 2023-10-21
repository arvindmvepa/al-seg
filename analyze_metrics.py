import os
import json
import csv
from glob import glob


results_csv_file = "results.csv"
root_dir = "."
exp_dirs = sorted(list(glob(os.path.join(root_dir, "*", "DMPLS*exp*v6*"))))

results = []
for exp_dir in exp_dirs:
        if not os.path.exists(exp_dir):
            continue
        round_results = []
        round_dirs = sorted(list(glob(os.path.join(exp_dir, "round*"))))
        for round_dir in round_dirs:
            model_dirs = sorted(list(glob(os.path.join(round_dir, "*"))))
            val_max = 0.0
            test_for_val_max = 0.0
            for model_dir in model_dirs:
                val_metric_dict = None
                test_metric_dict = None
                val_metric_file = os.path.join(exp_dir, round_dir, model_dir, "val_metrics.json")
                test_metric_file = os.path.join(exp_dir, round_dir, model_dir, "test_metrics.json")
                if os.path.exists(val_metric_file):
                    with open(val_metric_file) as val_metric_fp:
                        val_metric_dict = json.load(val_metric_fp)
                if os.path.exists(test_metric_file):
                    with open(test_metric_file) as test_metric_fp:
                        test_metric_dict = json.load(test_metric_fp)
                if (val_metric_dict is None) or (test_metric_dict is None):
                    continue
                val_result = val_metric_dict["performance"]
                if val_result > val_max:
                    val_max = val_result
                    test_for_val_max = test_metric_dict["performance"]
            round_results.extend([val_max, test_for_val_max])
        if round_results:
            exp_results = [os.path.basename(exp_dir)]
            exp_results.extend(round_results)
            results.append(exp_results)

headers = ["exp", "round0_val", "round0_test", "round1_val", "round1_test",
           "round2_val", "round2_test","round3_val", "round3_test",
           "round4_val", "round4_test", "round5_val", "round5_test"]

with open(results_csv_file, mode='w', newline='') as results_fp:
    writer = csv.writer(results_fp)
    writer.writerow(headers)
    for exp_results in results:
        writer.writerow(exp_results)

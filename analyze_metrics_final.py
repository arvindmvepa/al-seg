import os
import json
import csv
from glob import glob


def get_round_num(round_dir):
    return int(round_dir.split("round_")[-1])


results_csv_file = "results.csv"
root_dir = "."
exp_dirs = sorted(list(glob(os.path.join(root_dir, "*", "DMPLS*exp*v8*"))))

results = []
for exp_dir in exp_dirs:
        if not os.path.exists(exp_dir):
            continue
        round_results = []
        round_dirs = sorted(glob(os.path.join(exp_dir, "round*")), key=get_round_num)
        for round_dir in round_dirs:
            model_dirs = sorted([dirpath for dirpath in list(glob(os.path.join(round_dir, "*"))) if os.path.isdir(dirpath)])
            val_max = None
            test_for_val_max = None
            num_models = 0
            for model_dir in model_dirs:
                print(model_dir)
                val_metric_dict = None
                test_metric_dict = None
                val_metric_file = os.path.join(model_dir, "val_metrics.json")
                test_metric_file = os.path.join(model_dir, "test_metrics.json")
                if os.path.exists(val_metric_file):
                    with open(val_metric_file) as val_metric_fp:
                        val_metric_dict = json.load(val_metric_fp)
                if os.path.exists(test_metric_file):
                    with open(test_metric_file) as test_metric_fp:
                        test_metric_dict = json.load(test_metric_fp)
                if (val_metric_dict is None) or (test_metric_dict is None):
                    continue
                print(val_metric_dict)
                print(test_metric_dict)

                val_result = val_metric_dict["performance"]
                test_result = test_metric_dict["performance"]

                if val_max is None:
                    val_max = val_result
                    test_for_val_max = test_result
                elif val_result > val_max:
                    val_max = val_result
                    test_for_val_max = test_result
                num_models += 1
            if (val_max is not None) and (test_for_val_max is not None):
                round_results.extend([val_max, test_for_val_max, num_models])
        if round_results:
            exp_results = [os.path.basename(exp_dir)]
            exp_results.extend(round_results)
            results.append(exp_results)

headers = ["exp", "round0_val", "round0_test", "num_models", "round1_val", "round1_test", "num_models",
           "round2_val", "round2_test", "num_models", "round3_val", "round3_test", "num_models",
           "round4_val", "round4_test", "num_models", "round5_val", "round5_test", "num_models",
           "round6_val", "round6_test", "num_models", "round7_val", "round7_test", "num_models",
           "round8_val", "round8_test", "num_models", "round9_val", "round9_test", "num_models",
           "round10_val", "round10_test", "num_models", "round11_val", "round11_test", "num_models",
           "round12_val", "round12_test", "num_models", "round13_val", "round13_test", "num_models",
           "round14_val", "round14_test", "num_models", "round15_val", "round15_test", "num_models"]

with open(results_csv_file, mode='w', newline='') as results_fp:
    writer = csv.writer(results_fp)
    writer.writerow(headers)
    for exp_results in results:
        writer.writerow(exp_results)

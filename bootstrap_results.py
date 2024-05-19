import os
import json
from wsl4mis.code.dataloaders.dataset import BaseDataSets
from wsl4mis.code.val_2D import test_single_volume_cct
from torch.utils.data import DataLoader
import numpy as np
import sys
from tqdm import tqdm
sys.path.append("./wsl4mis/code")
from active_learning.dataset.data_params import data_params
import torch
from wsl4mis.code.networks.net_factory import net_factory
from glob import glob


def get_round_num(round_dir):
    return int(round_dir.split("round_")[-1])


def get_base_exp_name(exp_name):
    exp_field = "exp"
    exp_field_len_w_index = len(exp_field)+1
    exp_field_index = exp_name.find(exp_field)
    exp_name = exp_name[:exp_field_index]+exp_name[exp_field_index+exp_field_len_w_index:]
    return exp_name


def collect_exp_groups(exp_dirs):
    exp_groups = dict()
    for exp_name in exp_dirs:
        base_exp_name = get_base_exp_name(exp_name)
        if base_exp_name in exp_groups:
            exp_groups[base_exp_name] += [exp_name]
        else:
            exp_groups[base_exp_name] = [exp_name]
    return exp_groups


def collect_exp_results(exp_dirs, results_file_name="test_bs_results.txt"):
    exp_groups = collect_exp_groups(exp_dirs)
    print("exp_groups", exp_groups)
    results_dict = {}
    for base_exp_name, exp_group in exp_groups.items():
        exp_dict = {}
        for i, exp_dir_ in enumerate(exp_group):
            print("exp_dir_", exp_dir_)
            round_dirs_ = glob(os.path.join(exp_dir_, "round_*"))
            round_dirs_ = sorted(round_dirs_, key=get_round_num)
            print("round_dirs_", round_dirs_)
            round_dict = {}
            for round_dir_ in round_dirs_:
                cur_results_file_name = os.path.join(round_dir_, results_file_name)
                if os.path.exists(cur_results_file_name):
                    with open(cur_results_file_name, "r") as f:
                        im_scores_list = f.readlines()
                    im_scores_list = [float(im_score.strip()) for im_score in im_scores_list]
                    round_num = get_round_num(round_dir_)
                    round_dict[round_num] = im_scores_list
            if len(round_dict) > 0:
                exp_dict[exp_dir_] = round_dict
        if len(exp_dict) > 0:
            results_dict[base_exp_name] = exp_dict
            print("accepted ", base_exp_name)
        else:
            print("rejected ", base_exp_name)
    return results_dict


def get_mean_results(results_dict):
    mean_results_dict = {}
    for base_exp_name, exp_dict in results_dict.items():
        exp_names = list(exp_dict.keys())
        rounds = list(exp_dict[exp_names[0]].keys())
        mean_exp_dict = {}
        for round_ in rounds:
            mean_scores = []
            for exp_name in exp_names:
                mean_scores += exp_dict[exp_name][round_]
            mean_exp_dict[round_] = np.mean(mean_scores, axis=0)
        mean_results_dict[base_exp_name] = mean_exp_dict
    return mean_results_dict


def get_ci_results(exp_dirs, results_file_name="test_bs_results.txt"):
    results_dict = collect_exp_results(exp_dirs, results_file_name=results_file_name)
    mean_results_dict = get_mean_results(results_dict)
    ci_results_dict = {}
    print("mean_results_dict", mean_results_dict)
    for base_exp_name, exp_dict in mean_results_dict.items():
        rounds = list(exp_dict[base_exp_name].keys())
        ci_exp_dict = {}
        for round_ in rounds:
            ci_exp_dict[round_] = np.percentile(mean_results_dict[base_exp_name][round_], [2.5, 97.5])
        ci_results_dict[base_exp_name] = ci_exp_dict
    return ci_results_dict


def load_best_model(model_dir, seg_model='unet_cct', in_chns=1, num_classes=4):
    model = net_factory(net_type=seg_model, in_chns=in_chns, class_num=num_classes)
    best_model_path = os.path.join(model_dir, '{}_best_model.pth'.format(seg_model))
    model.load_state_dict(torch.load(best_model_path))
    return model


def save_results_to_file(results, save_file):
    with open(save_file, "w") as f:
        for result in results:
            # convert to float32 to avoid rounding to inf
            score = np.float32(result)
            f.write(f"{np.round(score, 7)}\n")
            f.flush()


def generate_test_predictions(model_dir, seg_model='unet_cct', in_chns=1, num_classes=4, ann_type="scribble",
                              dataset="ACDC", gpus="cuda:0"):
    data_params_ = data_params[dataset][ann_type]
    data_root = data_params_["data_root"]
    test_file = data_params_["test_file"]
    model = load_best_model(model_dir, seg_model=seg_model, in_chns=in_chns, num_classes=num_classes).to(gpus)
    model.eval()
    db_eval = BaseDataSets(split="val", val_file=test_file, data_root=data_root)
    evalloader = DataLoader(db_eval, batch_size=1, shuffle=False, num_workers=1)
    metric_list = []
    for i_batch, sampled_batch in tqdm(enumerate(evalloader)):
        metric_i = test_single_volume_cct(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes,
                                          gpus=gpus)
        metric_i = np.array(metric_i)
        dice_i = np.mean(metric_i[:, 0])
        metric_list += [dice_i]
    return metric_list


def generate_bootstrap_results(predictions, num_bootstraps=1000, seed=0):
    n = len(predictions)
    bootstrap_samples = []
    for i in range(num_bootstraps):
        random_state = np.random.RandomState(seed+i)
        sample = random_state.choice(predictions, n, replace=True)
        bootstrap_samples.append(sample)
    bootstrap_samples = np.array(bootstrap_samples)
    means = np.mean(bootstrap_samples, axis=1)
    return means


root_dir = "/home/amvepa91"
exp_length = 5
results_file = "test_bs_results.txt"
exp_dirs = sorted(list(glob(os.path.join(root_dir, "al-seg*", "DMPLS*coreset_pos_loss1_wt035_pos_loss2_wt005_use_phase_use_patient_v15"))))
"""
for exp_dir in exp_dirs:
        if not os.path.exists(exp_dir):
            continue
        print(exp_dir)
        round_results = []
        round_dirs = sorted(glob(os.path.join(exp_dir, "round*")), key=get_round_num)
        for round_dir in round_dirs:
            print(round_dir)
            model_dirs = sorted([dirpath for dirpath in list(glob(os.path.join(round_dir, "*"))) if os.path.isdir(dirpath)])
            val_max = None
            model_for_val_max = None
            num_models = 0
            for model_dir in sorted(model_dirs)[:exp_length]:
                print(model_dir)
                val_metric_dict = None
                test_metric_dict = None
                val_metric_file = os.path.join(model_dir, "val_metrics.json")
                if os.path.exists(val_metric_file):
                    with open(val_metric_file) as val_metric_fp:
                        val_metric_dict = json.load(val_metric_fp)
                else:
                    continue
                val_result = val_metric_dict["performance"]
                if val_max is None:
                    val_max = val_result
                    model_for_val_max = model_dir
                elif isinstance(val_result, float) and val_result > val_max:
                    val_max = val_result
                    model_for_val_max = model_dir
                num_models += 1
            cur_results_file = os.path.join(round_dir, results_file)
            if not os.path.exists(cur_results_file):
                test_results = generate_test_predictions(model_for_val_max)
                bs_test_results = generate_bootstrap_results(test_results)
                save_results_to_file(bs_test_results, cur_results_file)
                confidence_interval = np.percentile(bs_test_results, [2.5, 97.5])
                print("result")
                print("95% Confidence Interval:", confidence_interval)
            else:
                print("results already exist")
"""
# collect all the results
results = get_ci_results(exp_dirs)
print(results)

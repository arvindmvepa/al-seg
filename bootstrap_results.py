import os
import json
from wsl4mis.code.dataloaders.dataset import BaseDataSets
from wsl4mis.code.val_2D import test_single_volume_cct
from torch.utils.data import DataLoader
import numpy as np
import sys
sys.path.append("./wsl4mis/code")
from active_learning.dataset.data_params import data_params
import torch
from wsl4mis.code.networks.net_factory import net_factory
from glob import glob


def get_round_num(round_dir):
    return int(round_dir.split("round_")[-1])

def load_best_model(model_dir, seg_model='unet_cct', in_chns=1, num_classes=4):
    model = net_factory(net_type=seg_model, in_chns=in_chns, class_num=num_classes)
    best_model_path = os.path.join(model_dir, '{}_best_model.pth'.format(seg_model))
    model.load_state_dict(torch.load(best_model_path))
    return model

def generate_test_predictions(model_dir, seg_model='unet_cct', in_chns=1, num_classes=4, ann_type="scribble",
                              dataset="ACDC", gpus="cuda:0"):
    data_params_ = data_params[dataset][ann_type]
    data_root = data_params_["data_root"]
    test_file = data_params_["test_file"]
    model = load_best_model(model_dir, seg_model=seg_model, in_chns=in_chns, num_classes=num_classes).to(gpus)
    model.eval()
    db_eval = BaseDataSets(split="val", val_file=test_file, data_root=data_root)
    evalloader = DataLoader(db_eval, batch_size=1, shuffle=False, num_workers=1)
    metric_list = 0.0
    results_map = {}
    for i_batch, sampled_batch in enumerate(evalloader):
        metric_i = test_single_volume_cct(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes,
                                          gpus=gpus)
        metric_i = np.array(metric_i)
        results_map[sampled_batch["case"][0]] = np.mean(metric_i[:, 0])
        metric_list += metric_i
    return metric_list


def generate_bootstrap_results(predictions, num_bootstraps=1000):
    n = len(predictions)
    bootstrap_samples = np.random.choice(predictions, (n, num_bootstraps), replace=True)
    means = np.mean(bootstrap_samples, axis=0)
    return means


root_dir = "/home/amvepa91"
exp_length = 5
exp_dirs = sorted(list(glob(os.path.join(root_dir, "al-seg2", "DMPLS*coreset_pos_wt1500_norm_pos_loss1_wt035_pos_loss2_wt005_use_phase_use_patient_label1_v15"))))

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
            test_results = generate_test_predictions(model_for_val_max)
            bs_test_results = generate_bootstrap_results(test_results)
            confidence_interval = np.percentile(bs_test_results, [2.5, 97.5])
            print("result")
            print("95% Confidence Interval:", confidence_interval)
        exit()

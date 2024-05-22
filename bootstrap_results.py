import os
import json
from wsl4mis.code.dataloaders.dataset import BaseDataSets
from wsl4mis.code.val_2D import test_single_volume_cct, test_single_volume
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
        base_exp_name = os.path.basename(get_base_exp_name(exp_name))
        if base_exp_name in exp_groups:
            exp_groups[base_exp_name] += [exp_name]
        else:
            exp_groups[base_exp_name] = [exp_name]
    return exp_groups


def collect_exp_results(exp_dirs, results_file_name="test_bs_results.txt"):
    print("Collecting results from experiments...")
    exp_groups = collect_exp_groups(exp_dirs)
    results_dict = {}
    for base_exp_name, exp_group in exp_groups.items():
        exp_dict = {}
        print("base_exp_name: ", base_exp_name)
        for i, exp_dir_ in enumerate(exp_group):
            print("\tspecific_exp_name: ", exp_dir_)
            round_dirs_ = glob(os.path.join(exp_dir_, "round_*"))
            round_dirs_ = sorted(round_dirs_, key=get_round_num)
            round_dict = {}
            for round_dir_ in round_dirs_:
                print("\t\tround_dir: ", round_dir_)
                cur_results_file_name = os.path.join(round_dir_, results_file_name)
                if os.path.exists(cur_results_file_name):
                    with open(cur_results_file_name, "r") as f:
                        im_scores_list = f.readlines()
                    im_scores_list = [float(im_score.strip()) for im_score in im_scores_list]
                    round_num = get_round_num(round_dir_)
                    round_dict[round_num] = im_scores_list
                else:
                    print("\t\tNo results found for ", round_dir_)
            if len(round_dict) > 0:
                exp_dict[os.path.basename(exp_dir_)] = round_dict
            else:
                print("\tNo results found for ", exp_dir_)
        if len(exp_dict) > 0:
            results_dict[base_exp_name] = exp_dict
            print("accepted ", base_exp_name)
        else:
            print("rejected ", base_exp_name)
    print("Finished collecting experiment results!")
    return results_dict


def get_mean_results(results_dict, num_rounds=9):
    print("Calculating mean results...")
    mean_results_dict = {}
    for base_exp_name, exp_dict in results_dict.items():
        print("base_exp_name: ", base_exp_name)
        # check if all experiments have the same number of rounds
        exp_names = list(exp_dict.keys())
        # calculate mean scores
        rounds = list(exp_dict[exp_names[0]].keys())
        mean_exp_dict = {}
        for round_ in rounds:
            scores = []
            for exp_name in exp_names:
                if round_ not in exp_dict[exp_name]:
                    print("\t\tMissing round ", round_, " in ", exp_name)
                    continue
                scores += [exp_dict[exp_name][round_]]
            scores = np.array(scores)
            print("\tarray scores shape: ", scores.shape)
            mean_exp_dict[round_] = np.mean(scores, axis=0)
        mean_results_dict[base_exp_name] = mean_exp_dict
    print("Finished calculating mean results!")
    return mean_results_dict


def get_ci_results(exp_dirs, results_file_name="test_bs_results.txt"):
    results_dict = collect_exp_results(exp_dirs, results_file_name=results_file_name)
    mean_results_dict = get_mean_results(results_dict)
    ci_results_dict = {}
    print("Calculating confidence intervals...")
    for base_exp_name, exp_dict in mean_results_dict.items():
        print("base_exp_name: ", base_exp_name)
        rounds = list(exp_dict.keys())
        ci_exp_dict = {}
        for round_ in rounds:
            print("\tround: ", round_)
            results = mean_results_dict[base_exp_name][round_]
            print("\tNumber of bootstraps: ", len(results))
            ci = np.percentile(mean_results_dict[base_exp_name][round_], [2.5, 97.5])
            std = np.std(mean_results_dict[base_exp_name][round_])
            print("\tci: ", ci)
            print("\tstd: {}, 2*std: {} ".format(std, 2*std))
            ci_exp_dict[round_] = 2*std
        ci_results_dict[base_exp_name] = ci_exp_dict
    return ci_results_dict


def load_best_model(best_model_path, seg_model='unet_cct', in_chns=1, num_classes=4, device="cuda:0"):
    print("Loading best model from: ", best_model_path)
    print(f"seg_model: {seg_model}, in_chns: {in_chns}, num_classes: {num_classes}, device: {device}")
    model = net_factory(net_type=seg_model, in_chns=in_chns, class_num=num_classes)
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)))
    model = model.to(device)
    return model


def save_results_to_file(results, save_file):
    with open(save_file, "w") as f:
        for result in results:
            # convert to float32 to avoid rounding to inf
            score = np.float32(result)
            f.write(f"{np.round(score, 7)}\n")
            f.flush()


def generate_test_predictions(model, seg_model="unet_cct", num_classes=4, ann_type="scribble", dataset="ACDC",
                              device="cuda:0"):
    data_params_ = data_params[dataset][ann_type]
    data_root = data_params_["data_root"]
    test_file = data_params_["test_file"]
    model.eval()
    db_eval = BaseDataSets(split="val", val_file=test_file, data_root=data_root)
    evalloader = DataLoader(db_eval, batch_size=1, shuffle=False, num_workers=1)
    metric_list = []
    if seg_model == "unet_cct":
        eval_vol_func = test_single_volume_cct
    elif seg_model == "unet":
        eval_vol_func = test_single_volume
    else:
        raise ValueError(f"Invalid seg_model: {seg_model}")
    for i_batch, sampled_batch in tqdm(enumerate(evalloader)):
        metric_i = eval_vol_func(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, gpus=device)
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


if __name__ == '__main__':
    root_dirs = ["/home/amvepa91", "/home/asjchoi", r"C:\Users\Arvind\Documents"]
    results_file = "test_bs_results.txt"
    for root_dir in root_dirs:
        # scai
        glob_path = os.path.join(root_dir, "al-seg*", "DMPLS*sup*CHAOS")
        exp_dirs = sorted(list(glob(glob_path)))
        print(exp_dirs)
        overwrite = False
        device = "cuda:0"
        exp_dirs = [exp_dir for exp_dir in exp_dirs if ("CHAOS" not in exp_dir) and ("DAVIS" not in exp_dir) and ("MSCMR" not in exp_dir)]
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
                    model_dir_for_val_max = None
                    num_models = 0
                    for model_dir in sorted(model_dirs):
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
                            model_dir_for_val_max = model_dir
                        num_models += 1
                    cur_results_file = os.path.join(round_dir, results_file)
                    if model_dir_for_val_max is not None:
                        if (not os.path.exists(cur_results_file)) or overwrite:
                            best_model_path = list(glob(os.path.join(model_dir_for_val_max, '*_best_model.pth')))
                            if len(best_model_path) > 0:
                                best_model_path = best_model_path[0]
                                seg_model = os.path.basename(best_model_path).split("_best_model.pth")[0]
                                model = load_best_model(best_model_path, seg_model=seg_model, device=device)
                                if "sup" in exp_dir:
                                    ann_type = "label"
                                else:
                                    ann_type = "scribble"
                                print("ann_type", ann_type)
                                test_results = generate_test_predictions(model, seg_model=seg_model, ann_type=ann_type, device=device)
                                bs_test_results = generate_bootstrap_results(test_results)
                                save_results_to_file(bs_test_results, cur_results_file)
                                confidence_interval = np.percentile(bs_test_results, [2.5, 97.5])
                                print("result")
                                print("95% Confidence Interval:", confidence_interval)
                            else:
                                print("no model found")
                        else:
                            print("results already exist")
                    else:
                        print("no model found")
        # collect all the results
        results = get_ci_results(exp_dirs)
        print(results)

from abc import abstractmethod
from active_learning.model.base_model import BaseModel, SoftmaxMixin
from active_learning.model.db_scoring_functions import db_scoring_functions
import json
import os
import numpy as np
from tqdm import tqdm
from wsl4mis.code.networks.net_factory import net_factory
from wsl4mis.code.dataloaders.dataset import BaseDataSets, RandomGenerator
from wsl4mis.code.val_2D import test_single_volume_cct
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import h5py


class WSL4MISModel(SoftmaxMixin, BaseModel):
    """WSL4MIS Model class"""

    def __init__(self, dataset="ACDC", ann_type="scribble", ensemble_size=1, in_chns=1,
                 seg_model='unet_cct', batch_size=6, base_lr=0.01, max_iterations=60000,
                 deterministic=1, patch_size=(256, 256), inf_train_type="preds", feature_decoder_index=0, seed=0,
                 gpus="cuda:0", tag=""):
        super().__init__(ann_type=ann_type, dataset=dataset, ensemble_size=ensemble_size, seed=seed, gpus=gpus,
                         tag=tag)
        self.in_chns = in_chns
        self.seg_model = seg_model
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.deterministic = deterministic
        self.base_lr = base_lr
        self.patch_size = patch_size
        self.inf_train_type = inf_train_type
        self.feature_decoder_index = feature_decoder_index
        self.gpus = gpus

    def inf_train_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        train_preds_path = os.path.join(snapshot_dir, "train_preds.npz")
        if os.path.exists(train_preds_path):
            print(f"Train preds already exist in {train_preds_path}")
            return
        model = self.load_best_model(snapshot_dir).to(self.gpus)
        if self.inf_train_type == "features":
            model = model.encoder
        elif self.inf_train_type != "preds":
            raise ValueError(f"self.inf_train_type {self.inf_train_type } is not recognized. ust be either 'features' or 'preds'")
        model.eval()
        full_db_train = BaseDataSets(split="train", transform=transforms.Compose([RandomGenerator(self.patch_size)]),
                                     sup_type=self.ann_type, train_file=self.orig_train_im_list_file,
                                     data_root=self.data_root)
        full_trainloader = DataLoader(full_db_train, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        train_preds = {}
        for i_batch, sampled_batch in tqdm(enumerate(full_trainloader)):
            volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch['idx']
            volume_batch, label_batch, idx = volume_batch.to(self.gpus), label_batch.to(self.gpus), idx.cpu()[0]
            slice_basename = os.path.basename(full_db_train.sample_list[idx])
            outputs = model(volume_batch)
            outputs_ = self.extract_train_features(outputs)
            train_preds[slice_basename] = np.float16(outputs_.cpu().detach().numpy())
        np.savez_compressed(train_preds_path, **train_preds)

    def inf_eval_model(self, eval_file, model_no, snapshot_dir, round_dir, metrics_file, cur_total_oracle_split=0,
                       cur_total_pseudo_split=0):
        model = self.load_best_model(snapshot_dir).to(self.gpus)
        model.eval()
        db_eval = BaseDataSets(split="val", val_file=eval_file, data_root=self.data_root)
        evalloader = DataLoader(db_eval, batch_size=1, shuffle=False, num_workers=1)
        metric_list = 0.0
        results_map = {}
        for i_batch, sampled_batch in enumerate(evalloader):
            prediction_i,_ = test_single_volume_cct(
                sampled_batch["image"], sampled_batch["label"],
                model, classes=self.num_classes, gpus=self.gpus)
            metric_i = np.array(metric_i)
            results_map[sampled_batch["case"][0]] = np.mean(metric_i[:, 0])
            metric_list += metric_i
        metric_list = metric_list / len(db_eval)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        metrics = {"performance": performance, "mean_hd95": mean_hd95}
        metrics_file = os.path.join(snapshot_dir, metrics_file)
        with open(metrics_file, "w") as outfile:
            json_object = json.dumps(metrics, indent=4)
            outfile.write(json_object)
        results_map_file = os.path.join(snapshot_dir, os.path.basename(metrics_file)+"_map.json")
        with open(results_map_file, "w") as outfile:
            json_object = json.dumps(results_map, indent=4)
            outfile.write(json_object)

    def inf_val_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        self.inf_eval_model(eval_file=self.orig_val_im_list_file, model_no=model_no, snapshot_dir=snapshot_dir,
                            metrics_file="val_metrics.json", round_dir=round_dir,
                            cur_total_oracle_split=cur_total_oracle_split,
                            cur_total_pseudo_split=cur_total_pseudo_split)

    def inf_test_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        self.inf_eval_model(eval_file=self.orig_test_im_list_file, model_no=model_no, snapshot_dir=snapshot_dir,
                            metrics_file="test_metrics.json", round_dir=round_dir,
                            cur_total_oracle_split=cur_total_oracle_split,
                            cur_total_pseudo_split=cur_total_pseudo_split)

    def generate_model_results(self, snapshot_dir, prediction_ims=None):
        model = self.load_best_model(snapshot_dir).to(self.gpus)
        model.eval()
        db_eval = BaseDataSets(split="val", val_file=self.orig_test_im_list_file, data_root=self.data_root)
        evalloader = DataLoader(db_eval, batch_size=1, shuffle=False, num_workers=1)
        for i_batch, sampled_batch in enumerate(evalloader):
            case = sampled_batch["case"][0]
            case_wo_ext = os.path.basename(case).split(".")[0]
            if not any(prediction_im in sampled_batch["case"][0] for prediction_im in prediction_ims):
                continue
            prediction_i = test_single_volume_cct(sampled_batch["image"], sampled_batch["label"], model,
                                                  classes=self.num_classes, gpus=self.gpus)
            prediction_i = np.stack(prediction_i, axis=0)
            pred_h5 = os.path.join(snapshot_dir, case_wo_ext + "_pred.h5")
            with h5py.File(pred_h5, "w") as h5f:
                h5f.create_dataset("data", data=prediction_i, dtype=np.float32)

    def load_best_model(self, snapshot_dir):
        model = net_factory(net_type=self.seg_model, in_chns=self.in_chns, class_num=self.num_classes)
        best_model_path = os.path.join(snapshot_dir, '{}_best_model.pth'.format(self.seg_model))
        model.load_state_dict(torch.load(best_model_path))
        return model

    def extract_model_prediction(self, raw_model_outputs, batch_size=1):
        outputs = self._extract_model_prediction_channel(raw_model_outputs)
        outputs = torch.softmax(outputs, dim=1)
        if batch_size == 1:
            outputs = outputs[0]
        return outputs

    def extract_train_features(self, raw_model_outputs, batch_size=1):
        if self.inf_train_type == "features":
            return raw_model_outputs[self.feature_decoder_index]
        else:
            return self.extract_model_prediction(raw_model_outputs, batch_size=batch_size)

    @abstractmethod
    def _extract_model_prediction_channel(self, raw_model_outputs):
        raise NotImplementedError()

    def get_round_train_file_paths(self, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        new_train_im_list_file = os.path.join(round_dir,
                                              "train_al" + str(cur_total_oracle_split) + "_" + self.tag + \
                                              "_seed" + str(self.seed) \
                                                  if self.tag else \
                                                  "train_al" + str(cur_total_oracle_split) + "_seed" + str(self.seed))
        new_train_im_list_file = new_train_im_list_file + ".txt"
        return {self.file_keys[0]: new_train_im_list_file}

    def _init_train_file_info(self):
        self.orig_train_im_list_file = self.data_params['train_file']
        self.all_train_files_dict = dict()
        with open(self.orig_train_im_list_file, "r") as f:
            self.all_train_files_dict[self.file_keys[0]] = f.read().splitlines()

    def _init_val_file_info(self):
        self.orig_val_im_list_file = self.data_params["val_file"]

    def _init_test_file_info(self):
        self.orig_test_im_list_file = self.data_params["test_file"]

    def max_iter(self, cur_total_oracle_split, cur_total_pseudo_split):
        return int(self.max_iterations * (cur_total_oracle_split + cur_total_pseudo_split))

    @property
    def file_keys(self):
        return ['im']

    @property
    def im_key(self):
        return self.file_keys[0]


class DeepBayesianWSL4MISMixin:

    def inf_train_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        model = self.load_best_model(snapshot_dir).to(self.gpus)
        model.train()
        full_db_train = BaseDataSets(split="train", transform=transforms.Compose([RandomGenerator(self.patch_size)]),
                                     sup_type=self.ann_type, train_file=self.orig_train_im_list_file,
                                     data_root=self.data_root)
        full_trainloader = DataLoader(full_db_train, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        train_file = self.get_round_train_file_paths(round_dir=round_dir,
                                                     cur_total_oracle_split=cur_total_oracle_split,
                                                     cur_total_pseudo_split=cur_total_pseudo_split)[self.file_keys[0]]

        ann_db_train = BaseDataSets(split="train", transform=transforms.Compose([RandomGenerator(self.patch_size)]),
                                    sup_type=self.ann_type, train_file=train_file, data_root=self.data_root)
        train_preds = {}

        print("Start Monte Carlo dropout forward passes on the inferences!")
        print("Each inference will be repeated {} times.".format(self.T))
        with torch.no_grad():  # All computations inside this context will not track gradients
            for i_batch, sampled_batch in tqdm(enumerate(full_trainloader)):
                volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch['idx']
                volume_batch, label_batch, idx = volume_batch.to(self.gpus), label_batch.to(self.gpus), idx.cpu()[0]

                # skip images that are already annotated
                if full_db_train.sample_list[idx] in ann_db_train.sample_list:
                    continue

                slice_basename = os.path.basename(full_db_train.sample_list[idx])

                # Use repeat_interleave to create a batch with the same volume repeated T times
                volume_batch_repeated = volume_batch.repeat_interleave(self.T, dim=0)

                # Use the model to get the repeated outputs
                outputs = model(volume_batch_repeated)
                outputs = self.extract_model_prediction(outputs, batch_size=self.T)
                db_scores = self.get_db_score(outputs)
                train_preds[slice_basename] = np.float32(db_scores.cpu().detach().numpy())

        train_preds_path = os.path.join(snapshot_dir, "train_preds.npz")
        np.savez_compressed(train_preds_path, **train_preds)

    def get_db_score(self, preds):
        db_score_func = db_scoring_functions[self.db_score_func]
        return db_score_func(preds)



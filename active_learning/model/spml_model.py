from active_learning.model.base_model import BaseModel, MajorityVoteMixin, SoftmaxMixin
import json
import os
import subprocess
import re
import sys
from functools import partial


class SPMLModel(BaseModel):
    """SPML Model class"""

    def __init__(self, ann_type="box", data_root="spml_data/PASCAL", ensemble_size=1, epoch_len=10578,
                 num_epochs=3, seed=0, gpus="0", tag="",backbone_types="panoptic_deeplab_101", 
                 embedding_dim=64, prediction_types="segsort", lr_policy='poly', use_syncbn=True, 
                 warmup_iteration=100, lr=3e-3, wd=5e-4, batch_size=4, crop_size=256, image_scale=0.5, 
                 memory_bank_size=2, kmeans_iterations=10,kmeans_num_clusters=6, 
                 sem_ann_loss_types="segsort", sem_occ_loss_types="segsort", img_sim_loss_types="segsort", 
                 feat_aff_loss_types="none", sem_ann_concentration=None, sem_occ_concentration=None, 
                 img_sim_concentration=None, feat_aff_concentration=None, sem_ann_loss_weight=None, 
                 sem_occ_loss_weight=None, word_sim_loss_weight=None, img_sim_loss_weight=None, 
                 feat_aff_loss_weight=None, pretrained="spml_pretrained/resnet-101-cuhk.pth",
                 val_split='val', test_split='test'):
        super().__init__(ann_type=ann_type, data_root=data_root, ensemble_size=ensemble_size, seed=seed,
                         gpus=gpus, tag=tag)
        self._set_loss_weights(sem_ann_concentration, sem_occ_concentration, img_sim_concentration,
                               feat_aff_concentration, sem_ann_loss_weight, sem_occ_loss_weight, word_sim_loss_weight,
                               img_sim_loss_weight, feat_aff_loss_weight)
        self.exec_python = sys.executable
        self.backbone_types = backbone_types
        self.embedding_dim = embedding_dim
        self.prediction_types = prediction_types
        self.lr_policy = lr_policy
        self.use_syncbn = use_syncbn
        self.epoch_len = epoch_len
        self.num_epochs = num_epochs
        self.warmup_iteration = warmup_iteration
        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.image_scale = image_scale
        self.memory_bank_size = memory_bank_size
        self.kmeans_iterations = kmeans_iterations
        self.kmeans_num_clusters = kmeans_num_clusters
        self.sem_ann_loss_types = sem_ann_loss_types
        self.sem_occ_loss_types = sem_occ_loss_types
        self.img_sim_loss_types = img_sim_loss_types
        self.feat_aff_loss_types = feat_aff_loss_types
        self.pretrained = pretrained
        self.orig_train_split = self.model_params['train_split']
        self.val_split = val_split
        self.test_split = test_split

    def train_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        self._write_config_file(snapshot_dir, cur_total_oracle_split, cur_total_pseudo_split)

        train_script = self.get_train_script(snapshot_dir, round_dir, cur_total_oracle_split, cur_total_pseudo_split)
        prototype_script = self.get_prototype_script(snapshot_dir, round_dir, cur_total_oracle_split, cur_total_pseudo_split)
        scripts = {'train': train_script, 'prototype': prototype_script}

        for script_type, script in scripts.items():
            self.run_spml_script(script_type, script, model_no, snapshot_dir, cur_total_oracle_split)
    
    def inf_model(self, script_type, model_no, snapshot_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        script = self.get_inference_script(script_type, snapshot_dir, cur_total_oracle_split, cur_total_pseudo_split)
        self.run_spml_script(script_type, script, model_no, snapshot_dir, cur_total_oracle_split)

    def metrics_model(self, script_type, model_no, snapshot_dir, cur_total_oracle_split=0):
        script = self.get_metrics_script(script_type, snapshot_dir)
        self.run_spml_script(script_type, script, model_no, snapshot_dir, cur_total_oracle_split)
    
    def run_spml_script(self, script_type, script, model_no, snapshot_dir, cur_total_oracle_split=0):
        stdout_file = self._generate_stdout_bash_string(script_type, snapshot_dir, cur_total_oracle_split, model_no)
        stderr_file = self._generate_stderr_bash_string(script_type, snapshot_dir, cur_total_oracle_split, model_no)
        self._save_execute_params(script_type, script, stdout_file, stderr_file, snapshot_dir)

        full_script = f"{script} {stdout_file} {stderr_file}"
        print(full_script)
        subprocess.run(full_script, env=self.subproc_env, shell=True)
    
    def _save_execute_params(self, script_type, script, stdout_bash_string, stderr_bash_string, snapshot_dir):
        execute_params = dict()
        execute_params['env'] = self.subproc_env
        execute_params['ANN_TYPE'] = self.ann_type
        execute_params[f'{script_type}_SCRIPT'.upper()] = script
        execute_params[f'{script_type}_STDOUT_FILE'.upper()] = stdout_bash_string
        execute_params[f'{script_type}_STDERR_FILE'.upper()] = stderr_bash_string
        execute_params_file = os.path.join(snapshot_dir, f"{script_type}_execute_params.json")
        with open(execute_params_file, "w") as outfile:
            json_object = json.dumps(execute_params, indent=4)
            outfile.write(json_object)
        return execute_params
    
    def get_train_script(self, snapshot_dir, round_dir, cur_total_oracle_split, cur_total_pseudo_split):
        train_data_list = self.get_round_train_file_paths(round_dir=round_dir,
                                                          cur_total_oracle_split=cur_total_oracle_split,
                                                          cur_total_pseudo_split=cur_total_pseudo_split)[self.file_keys[0]]
        train_script = f"{self.exec_python} spml/pyscripts/train/train.py --data_dir {self.data_root} " \
                       f"--data_list {train_data_list} --snapshot_dir {os.path.join(snapshot_dir, 'stage1')} " \
                       f"--cfg_path {os.path.join(snapshot_dir, 'config_emb.yaml')}"
        return train_script

    def get_prototype_script(self, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0):
        memory_data_list = self.get_round_train_file_paths(round_dir=round_dir,
                                                           cur_total_oracle_split=cur_total_oracle_split,
                                                           cur_total_pseudo_split=cur_total_pseudo_split)[self.file_keys[1]]
        train_split = self.train_split(cur_total_oracle_split, cur_total_pseudo_split)
        prototype_script = f"{self.exec_python } spml/pyscripts/inference/prototype.py --data_dir {self.data_root} " \
                           f"--data_list {memory_data_list} " \
                           f"--save_dir {os.path.join(snapshot_dir, 'stage1', 'results', train_split)} " \
                           f"--snapshot_dir {os.path.join(snapshot_dir, 'stage1')} --label_divisor 2048 " \
                           f"--kmeans_num_clusters 12,12 --cfg_path {os.path.join(snapshot_dir, 'config_emb.yaml')}"
        return prototype_script
    
    def get_inference_script(self, script_type, snapshot_dir, cur_total_oracle_split, cur_total_pseudo_split):
        if script_type == 'inf_train':
            train_split = self.train_split(cur_total_oracle_split, cur_total_pseudo_split)
            return self._get_inference_script(snapshot_dir, self.orig_train_split, train_split, self.orig_train_im_list_file)
        elif script_type == 'inf_val':
            return self._get_inference_script(snapshot_dir, self.val_split, train_split, self.val_pim_list_file)
        elif script_type == 'inf_test':
            return self._get_inference_script(snapshot_dir, self.test_split, train_split, self.test_pim_list_file)
        else:
            raise ValueError(f"script_type {script_type} not recognized")

    def get_metrics_script(self, script_type, snapshot_dir):
        if script_type == 'metrics_val':
            return self._get_metrics_script(snapshot_dir, self.val_split)
        elif script_type == 'metrics_test':
            return self._get_metrics_script(snapshot_dir, self.test_split)
        else:
            raise ValueError(f"script_type {script_type} not recognized")

    def _get_metrics_script(self, snapshot_dir, inference_split):
        metrics_script = f"{self.exec_python} spml/pyscripts/benchmark/benchmark_by_mIoU.py " \
                         f"--gt_dir {self.data_root}/VOC2012/segcls " \
                         f" --pred_dir {snapshot_dir}/stage1/results/{inference_split}/semantic_gray  " \
                         f" --num_classes 21  --img_scale {self.image_scale}"
        return metrics_script

    def get_round_train_file_paths(self, round_dir, cur_total_oracle_split, **kwargs):
        new_train_im_list_file = os.path.join(round_dir,
                                              self.left_base_im_list + "_al" + str(cur_total_oracle_split) + "_" + \
                                              self.tag + "_seed" + str(self.seed) + "_" + self.right_base_im_list
                                              if self.tag else
                                              self.left_base_im_list + "al" + str(cur_total_oracle_split) + "-" + \
                                              self.tag + "seed" + str(self.seed) + "_" + self.right_base_im_list)
        new_train_im_list_file = new_train_im_list_file + ".txt"
        new_train_pim_list_file = os.path.join(round_dir,
                                               self.left_base_pim_list + "_al" + str(cur_total_oracle_split) + \
                                               "_" + self.tag + "_seed" + str(self.seed) + "_" + \
                                               self.right_base_pim_list
                                               if self.tag else
                                               self.left_base_pim_list + "al" + str(cur_total_oracle_split) + \
                                               "-" + self.tag + "seed" + str(self.seed) + "_" + \
                                               self.right_base_pim_list)
        new_train_pim_list_file = new_train_pim_list_file + ".txt"
        return {self.file_keys[0]: new_train_im_list_file, self.file_keys[1]: new_train_pim_list_file}

    def _init_train_file_info(self):
        self.left_base_im_list = self.model_params['left_base_im_list']
        self.right_base_im_list = self.model_params['right_base_im_list']
        self.left_base_pim_list = self.model_params['left_base_pim_list']
        self.right_base_pim_list = self.model_params['right_base_pim_list']
        self.orig_train_im_list_file = self.model_params["orig_train_im_list_file"]
        self.orig_train_pim_list_file = self.model_params["orig_train_pim_list_file"]
        self.all_train_files_dict = dict()
        with open(self.orig_train_im_list_file, "r") as f:
            self.all_train_files_dict[self.file_keys[0]] = f.read().splitlines()
        with open(self.orig_train_pim_list_file, "r") as f:
            self.all_train_files_dict[self.file_keys[1]] = f.read().splitlines()

    def _init_val_file_info(self):
        self.val_pim_list_file = self.model_params["val_pim_list"]
    
    def _generate_stdout_bash_string(self, file_type, snapshot_dir, cur_total_oracle_split, model_no):
        return f"> {os.path.join(snapshot_dir, file_type + f'_{self.ann_type}_AL_PROP' + str(cur_total_oracle_split) + '_MODEL_NO' + str(model_no) + '.results')}"

    def _generate_stderr_bash_string(self, file_type, snapshot_dir, cur_total_oracle_split, model_no):
        return f"2> {os.path.join(snapshot_dir, file_type + f'_{self.ann_type}_AL_PROP' + str(cur_total_oracle_split) + '_MODEL_NO' + str(model_no) + '.error')}"

    def max_iter(self, cur_total_oracle_split, cur_total_pseudo_split):
        return int(self.num_epochs * self.epoch_len * (cur_total_oracle_split + cur_total_pseudo_split))

    def _write_config_file(self, snapshot_dir, cur_total_oracle_split, cur_total_pseudo_split):
        with open("spml/configs/voc12_template.yaml", "r") as source:
            lines = source.read().splitlines()
        with open(os.path.join(snapshot_dir, "config_emb.yaml"), "w") as source:
            for line in lines:
                line = re.sub(r'TRAIN_SPLIT', self.train_split(cur_total_oracle_split, cur_total_pseudo_split), line)
                line = re.sub(r'BACKBONE_TYPES', self.backbone_types, line)
                line = re.sub(r'PREDICTION_TYPES', self.prediction_types, line)
                line = re.sub(r'EMBEDDING_MODEL', "", line)
                line = re.sub(r'PREDICTION_MODEL', "", line)
                line = re.sub(r'EMBEDDING_DIM', str(self.embedding_dim), line)
                line = re.sub(r'GPUS', self._convert_cuda_gpus_to_spml(), line)
                line = re.sub(r'BATCH_SIZE', str(self.batch_size), line)
                line = re.sub(r'LABEL_DIVISOR', "2048", line)
                line = re.sub(r'USE_SYNCBN', str(self.use_syncbn), line)
                line = re.sub(r'LR_POLICY', self.lr_policy, line)
                max_iter = str(self.max_iter(cur_total_oracle_split, cur_total_pseudo_split))
                line = re.sub(r'SNAPSHOT_STEP', str(max_iter), line)
                line = re.sub(r'MAX_ITERATION', str(max_iter), line)
                line = re.sub(r'WARMUP_ITERATION', str(self.warmup_iteration), line)
                line = re.sub(r'LR', str(self.lr), line)
                line = re.sub(r'WD', str(self.wd), line)
                line = re.sub(r'MEMORY_BANK_SIZE', str(self.memory_bank_size), line)
                line = re.sub(r'KMEANS_ITERATIONS', str(self.kmeans_iterations), line)
                line = re.sub(r'KMEANS_NUM_CLUSTERS', str(self.kmeans_num_clusters), line)
                line = re.sub(r'TRAIN_CROP_SIZE', str(self.crop_size), line)
                line = re.sub(r'TEST_SPLIT', str(self.val_split), line)
                line = re.sub(r'TEST_IMAGE_SIZE', str(self.crop_size), line)
                line = re.sub(r'TEST_CROP_SIZE_H', str(self.crop_size), line)
                line = re.sub(r'TEST_CROP_SIZE_W', str(self.crop_size), line)
                line = re.sub(r'TEST_STRIDE', str(self.crop_size), line)
                line = re.sub(r'PRETRAINED', self.pretrained, line)
                line = re.sub(r'SEM_ANN_LOSS_TYPES', self.sem_ann_loss_types, line)
                line = re.sub(r'SEM_OCC_LOSS_TYPES', self.sem_occ_loss_types, line)
                line = re.sub(r'IMG_SIM_LOSS_TYPES', str(self.img_sim_loss_types), line)
                line = re.sub(r'FEAT_AFF_LOSS_TYPES', str(self.feat_aff_loss_types), line)
                line = re.sub(r'WORD_SIM_LOSS_TYPES', "", line)
                line = re.sub(r'SEM_ANN_CONCENTRATION', str(self.sem_ann_concentration), line)
                line = re.sub(r'SEM_OCC_CONCENTRATION', str(self.sem_occ_concentration), line)
                line = re.sub(r'IMG_SIM_CONCENTRATION', str(self.img_sim_concentration), line)
                line = re.sub(r'FEAT_AFF_CONCENTRATION', str(self.feat_aff_concentration), line)
                line = re.sub(r'WORD_SIM_CONCENTRATION', "", line)
                line = re.sub(r'SEM_ANN_LOSS_WEIGHT', str(self.sem_ann_loss_weight), line)
                line = re.sub(r'SEM_OCC_LOSS_WEIGHT', str(self.sem_occ_loss_weight), line)
                line = re.sub(r'IMG_SIM_LOSS_WEIGHT', str(self.img_sim_loss_weight), line)
                line = re.sub(r'FEAT_AFF_LOSS_WEIGHT', str(self.feat_aff_loss_weight), line)
                line = re.sub(r'WORD_SIM_LOSS_WEIGHT', str(self.word_sim_loss_weight), line)
                line = re.sub(r'IMAGE_SCALE', str(self.image_scale), line)
                source.write(line+"\n")

    def _set_loss_weights(self, sem_ann_concentration, sem_occ_concentration, img_sim_concentration,
                          feat_aff_concentration, sem_ann_loss_weight, sem_occ_loss_weight, word_sim_loss_weight,
                          img_sim_loss_weight, feat_aff_loss_weight):
        if sem_ann_concentration is None:
            self.sem_ann_concentration = 6
        else:
            self.sem_ann_concentration = sem_ann_concentration
        if img_sim_concentration is None:
            self.img_sim_concentration = 16
        else:
            self.img_sim_concentration = img_sim_concentration
        if feat_aff_concentration is None:
            self.feat_aff_concentration = 0
        else:
            self.feat_aff_concentration = feat_aff_concentration
        if word_sim_loss_weight is None:
            self.word_sim_loss_weight = 0
        else:
            self.word_sim_loss_weight = word_sim_loss_weight
        if img_sim_loss_weight is None:
            self.img_sim_loss_weight = 0.1
        else:
            self.img_sim_loss_weight = img_sim_loss_weight
        if feat_aff_loss_weight is None:
            self.feat_aff_loss_weight = 0
        else:
            self.feat_aff_loss_weight = feat_aff_loss_weight
        if sem_occ_concentration is None:
            if (self.ann_type == "box") or ("liger" in self.ann_type):
                self.sem_occ_concentration = 8
            elif self.ann_type == "scribble":
                self.sem_occ_concentration = 12
        else:
            self.sem_occ_concentration = sem_occ_concentration
        if sem_ann_loss_weight is None:
            if (self.ann_type == "box") or ("liger" in self.ann_type):
                self.sem_ann_loss_weight = 0.3
            elif self.ann_type == "scribble":
                self.sem_ann_loss_weight = 1.0
        else:
            self.sem_ann_loss_weight = sem_ann_loss_weight
        if sem_occ_loss_weight is None:
            if (self.ann_type == "box") or ("liger" in self.ann_type):
                self.sem_occ_loss_weight = 0.3
            elif self.ann_type == "scribble":
                self.sem_occ_loss_weight = 0.5
        else:
            self.sem_occ_loss_weight = sem_occ_loss_weight

    def _convert_cuda_gpus_to_spml(self):
        return ",".join([str(i) for i,_ in enumerate(self.gpus.split(","))])

    @property
    def file_keys(self):
        return ['im', 'pim']

    @property
    def im_key(self):
        return self.file_keys[0]

    @property
    def model_string(self):
        return "spml"
    
    @property
    def subproc_env(self):
        env = dict()
        env = os.environ.copy()
        env['PATH'] = f"{os.environ['PATH']}"
        env['PYTHONPATH'] ="spml"
        if isinstance(self.gpus, str):
            env['CUDA_VISIBLE_DEVICES'] = self.gpus
        else:
            raise ValueError("cuda_devices must be str type")
        return env

    def __repr__(self):
        mapping = self.__dict__
        mapping["model_cls"] = "SPMLModel"
        return json.dumps(mapping, indent=4)


class SPMLwMajorityVote(MajorityVoteMixin, SPMLModel):

    def _get_inference_script(self, snapshot_dir, save_dir_split, semantic_memory_split, data_list):
        inference_script = f"{self.exec_python} spml/pyscripts/inference/inference.py --data_dir {self.data_root} " \
                           f"--data_list {data_list} " \
                           f"--save_dir {snapshot_dir}/stage1/results/{save_dir_split} " \
                           f"--snapshot_dir {snapshot_dir}/stage1 " \
                           f"--semantic_memory_dir {snapshot_dir}/stage1/results/{semantic_memory_split}/semantic_prototype " \
                           f"--label_divisor 2048 --kmeans_num_clusters 12,12 " \
                           f"--cfg_path {snapshot_dir}/config_emb.yaml "
        return inference_script

class SPMLwSoftmax(SoftmaxMixin, SPMLModel):

    def _get_inference_script(self, snapshot_dir, save_dir_split, semantic_memory_split, data_list):
        inference_script = f"{self.exec_python} spml/pyscripts/inference/inference_segsort_softmax.py " \
                           f"--data_dir {self.data_root} --data_list {data_list} " \
                           f"--save_dir {snapshot_dir}/stage1/results/{save_dir_split} " \
                           f"--snapshot_dir {snapshot_dir}/stage1 " \
                           f"--semantic_memory_dir {snapshot_dir}/stage1/results/{semantic_memory_split}/semantic_prototype " \
                           f"--label_divisor 2048 --kmeans_num_clusters 12,12 " \
                           f"--cfg_path {snapshot_dir}/config_emb.yaml "
        return inference_script

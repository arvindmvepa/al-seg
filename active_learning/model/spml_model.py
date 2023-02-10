from active_learning.model.base_model import BaseModel
import json
import os
import subprocess
import time
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm


class SPMLModel(BaseModel):
    """SPML Model class"""

    def __init__(self, ann_type="box", data_root="./spml", ensemble_size=1, epoch_len=10578, num_epochs=3,
                 seed=0, tag="", virtualenv='/home/asjchoi/SPML_Arvind/spml-env'):
        super().__init__(ann_type=ann_type, data_root=data_root, ensemble_size=ensemble_size, epoch_len=epoch_len,
                         num_epochs=num_epochs, seed=seed, tag=tag, virtualenv=virtualenv)

    def train_model(self, model_no, snapshot_dir, round_dir, cur_total_oracle_split=0, cur_total_pseudo_split=0,
                    cuda_devices="0", save_params=None):
        if save_params is None:
            save_params = dict()
        env = dict()
        env['AL_PROP'] = str(cur_total_oracle_split)
        env['MODEL_NO'] = str(model_no)
        max_iter = str(self.max_iter(cur_total_oracle_split, cur_total_pseudo_split))
        env['MAX_ITERATION'] = max_iter
        env['TRAIN_DATA_LIST'] = self.get_round_train_file_paths(round_dir=round_dir,
                                                                 cur_total_oracle_split=cur_total_oracle_split,
                                                                 cur_total_pseudo_split=cur_total_pseudo_split)[self.file_keys[0]]
        env['ORIG_TRAIN_DATA_LIST'] = self.orig_train_im_list_file
        env['ORIG_MEMORY_DATA_LIST'] = self.orig_train_pim_list_file
        env['MEMORY_DATA_LIST'] = self.get_round_train_file_paths(round_dir,
                                                                  cur_total_oracle_split=cur_total_oracle_split,
                                                                  cur_total_pseudo_split=cur_total_pseudo_split)[self.file_keys[1]]
        env['SNAPSHOT_STEP'] = max_iter
        env['SNAPSHOT_DIR'] = snapshot_dir
        env['VIRTUAL_ENV'] = self.virtualenv
        env['PATH'] = f"{env['VIRTUAL_ENV']}/bin:{os.environ['PATH']}"
        env['PYTHONPATH'] ="spml"
        env['TRAIN_SPLIT'] = self.train_split(cur_total_oracle_split, cur_total_pseudo_split)
        env['ORIG_TRAIN_SPLIT'] = self.model_params['train_split']
        if isinstance(cuda_devices, str):
            env['CUDA_VISIBLE_DEVICES'] = cuda_devices
        else:
            raise ValueError("cuda_devices must be str type")
        exec_script = self.model_params["exec_script"]
        stdout_script = f" > {os.path.join(snapshot_dir, 'box_AL_PROP' + str(cur_total_oracle_split) + '_MODEL_NO' + str(model_no) + '.results')}"
        stderr_script = f" 2> {os.path.join(snapshot_dir, 'box_AL_PROP' + str(cur_total_oracle_split) + '_MODEL_NO' + str(model_no) + '.error')}"

        # save parameters
        params = dict()
        params['env'] = env
        params['MODEL'] = str(self)
        params['ANN_TYPE'] = self.ann_type
        params['EXEC_SCRIPT'] = exec_script
        params['STDOUT_SCRIPT'] = stdout_script
        params['STDERR_SCRIPT'] = stderr_script
        params.update(save_params)

        params_file = os.path.join(snapshot_dir, "params.json")
        with open(params_file, "w") as outfile:
            json_object = json.dumps(params)
            outfile.write(json_object)

        # train model
        print(f"\tTrain model {model_no}")
        subprocess.run(f"{exec_script} {stdout_script} {stderr_script}", env=env, shell=True)

    def get_ensemble_scores(self, score_func, im_score_file, round_dir, ignore_ims_dict, skip=False):
        print("Starting to Ensemble Predictions")
        f = open(im_score_file, "w")
        train_results_dir = os.path.join(round_dir, "*", self.model_params['train_results_dir'])
        filt_models_result_files = self._filter_unann_ims(train_results_dir, ignore_ims_dict)
        for models_result_file in tqdm(zip(*filt_models_result_files)):
            results_arr, base_name = self._convert_ensemble_results_to_arr(models_result_file)
            # calculate the score_func over the ensemble of predictions
            score = score_func(results_arr)
            f.write(f"{base_name},{np.round(score, 7)}\n")
            f.flush()
        f.close()

    def get_round_train_file_paths(self, round_dir, cur_total_oracle_split, **kwargs):
        new_train_im_list_file = os.path.join(round_dir,
                                              self.left_base_im_list + "_al" + str(cur_total_oracle_split) + "_" + \
                                              self.tag + "_seed" + str(self.seed) + "_" + self.right_base_im_list
                                              if self.tag else
                                              self.left_base_im_list + "al" + str(cur_total_oracle_split) + "-" + \
                                              self.tag + "seed" + str(self.seed) + "_" + self.right_base_im_list)
        new_train_pim_list_file = os.path.join(round_dir,
                                               self.left_base_pim_list + "_al" + str(cur_total_oracle_split) + \
                                               "_" + self.tag + "_seed" + str(self.seed) + "_" + \
                                               self.right_base_pim_list
                                               if self.tag else
                                               self.left_base_pim_list + "al" + str(cur_total_oracle_split) + \
                                               "-" + self.tag + "seed" + str(self.seed) + "_" + \
                                               self.right_base_pim_list)

        return {self.file_keys[0]: new_train_im_list_file, self.file_keys[1]: new_train_pim_list_file}

    def _convert_ensemble_results_to_arr(self, models_result_file):
        results = []
        for model_result_file in models_result_file:
            arr = np.asarray(Image.open(model_result_file).convert('L'))
            results.append(arr)
        results_arr = np.stack(results)
        # use the first model's pred_file basename because it's the same image
        base_name = os.path.basename(models_result_file[0])
        return results_arr, base_name

    def _filter_unann_ims(self, train_results_dir, ignore_ims_dict):
        # load models' results files
        models_result_files = [sorted(glob(os.path.join(result_dir, "*"))) for result_dir in
                               sorted(list(glob(train_results_dir)))]
        # filter model results in which we already have annotations
        # Note: filter predictions for first model and use the filtering results for the other models
        filt_models_result_files = []
        for _ in range(self.ensemble_size):
            filt_models_result_files.append([])
        for i, result_file in enumerate(models_result_files[0]):
            remove = False
            for im_file in ignore_ims_dict[self.im_key]:
                if os.path.basename(result_file)[:-4] in im_file:
                    remove = True
                if remove:
                    break
            if not remove:
                for j in range(self.ensemble_size):
                    filt_models_result_files[j].append(models_result_files[j][i])
        return filt_models_result_files

    def _init_train_file_info(self):
        self.left_base_im_list = self.model_params['left_base_im_list']
        self.right_base_im_list = self.model_params['right_base_im_list']
        self.left_base_pim_list = self.model_params['left_base_pim_list']
        self.right_base_pim_list = self.model_params['right_base_pim_list']
        self.orig_train_im_list_file = os.path.join(self.data_root,
                                                    "datasets",
                                                    "voc12",
                                                    self.left_base_im_list + "_" + self.right_base_im_list)
        self.orig_train_pim_list_file = os.path.join(self.data_root,
                                                     "datasets",
                                                     "voc12",
                                                     self.left_base_pim_list + "_" + self.right_base_pim_list)
        self.all_train_files_dict = dict()
        self.all_train_files_dict[self.file_keys[0]] = open(self.orig_train_im_list_file).readlines()
        self.all_train_files_dict[self.file_keys[1]] = open(self.orig_train_pim_list_file).readlines()


    def max_iter(self, cur_total_oracle_split, cur_total_pseudo_split):
        return int(self.num_epochs * self.epoch_len * (cur_total_oracle_split + cur_total_pseudo_split))

    @property
    def file_keys(self):
        return ['im', 'pim']

    @property
    def im_key(self):
        return self.file_keys[0]

    @property
    def model_string(self):
        return "spml"

    def __repr__(self):
        mapping = self.__dict__
        mapping["model_cls"] = "SPMLModel"
        return json.dumps(mapping)

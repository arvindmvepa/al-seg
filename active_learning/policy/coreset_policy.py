import os
import h5py 
import numpy as np
from torchvision import transforms

from active_learning.policy.base_policy import BaseActiveLearningPolicy
from active_learning.kcenter_greedy import kCenterGreedy
from wsl4mis.code.dataloaders.dataset import RandomGenerator


class CoresetPolicy(BaseActiveLearningPolicy):
    """Policy which applies coreset sampling to split data
    """

    def __init__(self, model, model_uncertainty=None, ensemble_kwargs=None, uncertainty_kwargs=None,
                 save_im_score_file="scores.txt", rounds=(), exp_dir="test", pseudolabels=False, 
                 tag="", seed=0, patch_size=(256, 256), data_root="./wsl4mis_data/ACDC", 
                 train_files="ACDC_training_slices"):
        super().__init__(model=model, model_uncertainty=model_uncertainty, ensemble_kwargs=ensemble_kwargs,
                         uncertainty_kwargs=uncertainty_kwargs, rounds=rounds, pseudolabels=pseudolabels,
                         exp_dir=exp_dir, tag=tag, seed=seed)
        self.save_im_score_file = save_im_score_file
        self.prev_round_im_score_file = None
        self.data_root = data_root
        self.train_files = train_files

        self.full_path = os.path.join(data_root, train_files)
        self.transform = transforms.Compose([RandomGenerator((256, 256))])

    def _run_round(self):
        if self._round_num < (self.num_rounds - 1):
            save_im_score_file = os.path.join(self.round_dir, self.save_im_score_file)
            ensemble_kwargs = {"inf_train": True}
            model_uncertainty_kwargs = {"im_score_file": save_im_score_file,
                                        "ignore_ims_dict": self.cur_oracle_ims,
                                        "round_dir": self.round_dir}
            self._run_round_models(ensemble_kwargs=ensemble_kwargs, calculate_model_uncertainty=True,
                                   uncertainty_kwargs=model_uncertainty_kwargs)
            self.prev_round_im_score_file = save_im_score_file
        else:
            self._run_round_models()

    def data_split(self):
        if self._round_num == 0:
            return self.random_split()
        else:
            return self.coreset_split()
        
    def coreset_split(self):
        print("Splitting data using coreset sampling!")
        return self._data_split(self._coreset_sample_unann_indices)
    
    def _coreset_sample_unann_indices(self):
        unann_im_dict, num_samples = self._get_unann_train_file_paths(), self._get_unann_num_samples()
        # get the scores per image from the score file in this format: img_name, score
        im_scores_list = open(self.prev_round_im_score_file).readlines()

        # get img_name from scores.txt file
        im_list = [im_score.strip().split(",")[0] for im_score in im_scores_list]

        # filter out training set images that we are using as our labeled set
        filtered_im_list = [im for im in im_list if
                            all(im not in ann_im for ann_im in self.cur_oracle_ims[self.im_key])]
        
        filtered_im_full_path_list = [os.path.join(self.full_path, im_path) for im_path in filtered_im_list]


        cases = []
        # Get the image features 
        for case in filtered_im_full_path_list:
            h5f = h5py.File(case, 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
            cases.append(sample['image'].numpy())

        cases_arr = np.concatenate(cases, axis=0)

        KCG = kCenterGreedy(cases_arr)
        print("Start coreset sampling")
        core_set = KCG.select_batch_(already_selected=[], N=num_samples)
        core_set = [filtered_im_list[i] for i in core_set]

        # retrieve indices of unann files
        unann_ims = unann_im_dict[self.im_key]
        sampled_unann_indices = [i for i, unann_im in enumerate(unann_ims) for top_im in core_set if top_im in unann_im]
        return sampled_unann_indices
        

        

        

from random import Random
import os


class BaseActiveLearningPolicy:
    """Base class for active learning policy which utilizes random split.

    Attributes
    ----------
    model : BaseModel
        Model object using in the policy
    model_uncertainty : BaseModelUncertainty
        ModelUncertainty object used by `model` in the policy
    ensemble_kwargs : dict, optional
        kwargs passed to the ensembling method for `model`
    uncertainty_kwargs : dict, optional
        kwargs passed to the uncertainty method for `model_uncertainty`
    num_rounds : int
    rounds : iterable
        An iterable of tuples, where each tuple represents the additional
        oracle/pseudolabel data proportion for each round
    round_dir : str
    pseudolabels : Undefined
    tag : str
    seed : int
    all_train_files_dict : dict
        A dict with keys from `file_keys` and values a list of files for
        each key, where each list index corresponds to the same datum
        but different file type
    file_keys : iterable
        The different file types for the data
    im_key : str
        The "primary" file type
    cur_total_oracle_split : float
        Current proportion of data labeled by oracle
    cur_total_pseudo_split : float
        Current proportion of data used for pseudolabels
    cur_oracle_ims : dict
        A subset of `all_train_files_dict` with the same keys but only
        containing images currently labeled by the oracle
    cur_pseudo_ims : dict
        A subset of `all_train_files_dict` with the same keys but only
        containing images currently used as pseudolabels

    Methods
    -------
    run()
        Runs the given policy for `num_rounds`
    data_split()
        Marks a subset of unannotated data to be labeled by the oracle
    random_split()
        Randomly splits unannotated data to be labeled by the oracle

    """

    def __init__(self, model, model_uncertainty=None, data_geometry=None, ensemble_kwargs=None,
                 uncertainty_kwargs=None, geometry_kwargs=None, rounds=(),
                 exp_dir="test", pseudolabels=False,
                 tag="", seed=0):
        self.model = model
        self.model_uncertainty = model_uncertainty
        self.data_geometry = data_geometry
        self.ensemble_kwargs = ensemble_kwargs if ensemble_kwargs else dict()
        self.uncertainty_kwargs = uncertainty_kwargs if uncertainty_kwargs else dict()
        self.geometry_kwargs = geometry_kwargs if geometry_kwargs else dict()
        self.num_rounds = len(rounds)
        self.rounds = iter(rounds)
        self._round_num = None
        self.round_dir = None
        self.exp_dir = exp_dir
        self.pseudolabels = pseudolabels
        self.tag = tag
        self.seed = seed
        self.random_gen = Random(self.seed)

        self.all_train_files_dict = self.model.all_train_files_dict
        self.data_root = self.model.data_root
        self.file_keys = self.model.file_keys
        self.im_key = self.model.im_key

        self.cur_total_oracle_split = 0.0
        self.cur_total_pseudo_split = 0.0

        self.cur_oracle_ims = {key: [] for key in self.file_keys}
        self.cur_pseudo_ims = {key: [] for key in self.file_keys}

        self.setup()

    def setup(self):
        self.setup_data_geometry()

    def run(self):
        for i in range(self.num_rounds):
            self._setup_round()
            self._run_round()

    def data_split(self):
        return self.random_split()

    def random_split(self):
        print("Splitting data randomly!")
        return self._data_split(self._random_sample_unann_files)

    def setup_data_geometry(self):
        if self.data_geometry:
            self.data_geometry.setup(self.data_root, self.all_train_files_dict[self.im_key])

    def _run_round(self):
        self._run_round_models()

    def _run_round_models(self, im_score_file=None, calculate_model_uncertainty=False, calculate_data_geometry=False,
                          ensemble_kwargs=None, uncertainty_kwargs=None, geometry_kwargs=None):
        if ensemble_kwargs is None:
            ensemble_kwargs = dict()
        ensemble_kwargs.update(self.ensemble_kwargs)
        if uncertainty_kwargs is None:
            uncertainty_kwargs = dict()
        uncertainty_kwargs.update(self.uncertainty_kwargs)
        if geometry_kwargs is None:
            geometry_kwargs = dict()
        geometry_kwargs.update(self.geometry_kwargs)

        assert (not calculate_model_uncertainty) or (not calculate_data_geometry), \
            "Cannot calculate both model uncertainty and data geometry in the same round"

        assert (im_score_file and (calculate_model_uncertainty or calculate_data_geometry)) or \
               (not (calculate_model_uncertainty or calculate_data_geometry)), \
            "Must provide an im_score_file if calculating model uncertainty or data geometry"

        if calculate_data_geometry:
            self.data_geometry.calculate_representativeness(im_score_file=im_score_file,
                                                            num_samples=self._get_unann_num_samples(),
                                                            round_num=self._round_num,
                                                            already_selected=self.cur_oracle_ims[self.im_key],
                                                            **geometry_kwargs)
            self.cur_im_score_file = im_score_file

        new_ann_ims = self.data_split()
        for im_type, files in new_ann_ims.items():
            print(f"Old length of {im_type} data: {len(self.cur_oracle_ims[im_type])}")
            print(f"Added length of {im_type} data: {len(files)}")
        # add the newly annotated files to our list of annotated files
        for key in self.file_keys:
            self.cur_oracle_ims[key] = self.cur_oracle_ims[key] + new_ann_ims[key]
        self.model.train_ensemble(round_dir=self.round_dir, cur_total_oracle_split=self.cur_total_oracle_split,
                                  cur_total_pseudo_split=self.cur_total_pseudo_split,
                                  inf_train=calculate_model_uncertainty, **ensemble_kwargs)
        if calculate_model_uncertainty:
            self.model_uncertainty.calculate_uncertainty(im_score_file=im_score_file, **uncertainty_kwargs)
            self.cur_im_score_file = im_score_file

    def _data_split(self, splt_func):
        new_train_file_paths = self.model.get_round_train_file_paths(self.round_dir, self.cur_total_oracle_split)
        unann_im_dict = self._get_unann_train_file_paths()
        sampled_unann_indices = splt_func()
        sampled_unann_im_dict = {key: [] for key in self.file_keys}
        for key in self.file_keys:
            unann_ims = unann_im_dict[key]
            sampled_unann_ims = [unann_ims[i] for i in sampled_unann_indices]
            sampled_unann_im_dict[key] = sampled_unann_ims
            new_train_file_path = new_train_file_paths[key]
            with open(new_train_file_path, 'w') as new_file:
                new_file.write('\n'.join(sampled_unann_ims + self.cur_oracle_ims[key]))
        return sampled_unann_im_dict

    def _random_sample_unann_files(self):
        num_samples = self._get_unann_num_samples()
        indices = list(range(num_samples))
        return self.random_gen.sample(indices, num_samples)

    def _get_unann_train_file_paths(self):
        unann_im_dict = {key: [] for key in self.file_keys}
        # get all the unannotated images by key
        # assume that, if there are multiple files, they are paired by index and of equal length
        for i, key in enumerate(self.file_keys):
            unann_im_dict[key] = self._get_unann_files(key)
        return unann_im_dict

    def _get_unann_files(self, key):
        all_ims = self.all_train_files_dict[key]
        cur_ann_ims = self.cur_oracle_ims[key]
        unann_ims = [im for im in all_ims if im not in cur_ann_ims]
        return unann_ims

    def _get_unann_num_samples(self):
        """num samples = prop * (unannotated images + annotated images) - (annotated images)"""
        cur_ann_ims = self.cur_oracle_ims[self.im_key]
        unann_ims = self._get_unann_files(self.im_key)
        return int(len(unann_ims + cur_ann_ims) * self.cur_total_oracle_split) - len(cur_ann_ims)

    def _setup_round(self):
        round_params = next(self.rounds)
        round_oracle_split, round_pseudo_split = round_params
        self.cur_total_oracle_split += round_oracle_split
        self.cur_total_pseudo_split += round_pseudo_split

        if self._round_num is None:
            self._round_num = 0
        else:
            self._round_num += 1
        self._create_round_dir()

        print(f"Round {self._round_num}, round params: {round_params}")

    def _create_round_dir(self):
        self.round_dir = os.path.join(self.exp_dir, f"round_{self._round_num}")
        if not os.path.exists(self.round_dir):
            os.makedirs(self.round_dir)


class RandomActiveLearningPolicy(BaseActiveLearningPolicy):
    pass
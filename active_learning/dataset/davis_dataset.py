from active_learning.dataset.chaos_dataset import BaseDataset
from collections import defaultdict
import numpy as np


class DAVIS_Dataset(BaseDataset):

    def _load_meta_data(self, im_data_file):
        meta_data = {}
        meta_data['VOLUME_NAME'] = self._extract_volume_name(im_data_file)
        meta_data['SLICE_NO'] = self._extract_slice_no(im_data_file)
        return meta_data

    def process_meta_data(self, image_meta_data):
        # encode all cfg features
        extra_features_lst = []
        counter = defaultdict(lambda: len(counter))
        for im_meta_datum in image_meta_data:
            extra_features = []

            # add patient number
            volume_name = im_meta_datum['VOLUME_NAME']
            print("debug: volume_name: ", volume_name)
            patient_num = counter[volume_name]
            print("debug: patient_num: ", patient_num)
            extra_features.append(patient_num)

            # add constant for phase
            extra_features.append(0)

            # add slice position
            slice_num = im_meta_datum['SLICE_NO']
            extra_features.append(slice_num)

            extra_features_lst.append(np.array(extra_features))

        return np.array(extra_features_lst)

    def get_non_image_indices(self):
        non_image_indices = dict()
        non_image_indices['patient_starting_index'] = 0
        non_image_indices['patient_ending_index'] = non_image_indices['patient_starting_index'] + 1
        non_image_indices['phase_starting_index'] = non_image_indices['patient_ending_index']
        non_image_indices['phase_ending_index'] = non_image_indices['phase_starting_index'] + 1
        non_image_indices['slice_pos_starting_index'] = non_image_indices['phase_ending_index']
        non_image_indices['slice_pos_ending_index'] = non_image_indices['slice_pos_starting_index'] + 1
        non_image_indices['uncertainty_starting_index'] = non_image_indices['slice_pos_ending_index']
        non_image_indices['uncertainty_ending_index'] = non_image_indices['uncertainty_starting_index'] + 1
        return non_image_indices

    def _extract_volume_name(self, im_path):
        volume_name_raw = im_path[self._get_volume_name_start_index(im_path):]
        print("debug: volume_name_raw: ", volume_name_raw)
        volume_name = volume_name_raw[:volume_name_raw.index("_")]
        return volume_name

    def _get_volume_name_start_index(self, im_path):
        patient_prefix = "training_slices/"
        patient_prefix_len = len(patient_prefix)
        patient_prefix_index = im_path.index(patient_prefix)
        patient_num_start_index = patient_prefix_index + patient_prefix_len
        return patient_num_start_index

    def _extract_slice_no(self, im_path):
        slice_no_string_raw = im_path[self._get_slice_no_start_index(im_path):]
        slice_no_string = slice_no_string_raw[:slice_no_string_raw.index(".")]
        return int(slice_no_string)

    def _get_slice_no_start_index(self, im_path):
        slice_no_prefix = "slice_"
        slice_no_len = len(slice_no_prefix)
        slice_no_index = im_path.index(slice_no_prefix)
        slice_no_start_index = slice_no_index + slice_no_len
        return slice_no_start_index
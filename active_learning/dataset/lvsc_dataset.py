from active_learning.dataset.base_dataset import BaseDataset
import numpy as np
from collections import defaultdict


class LVSC_Dataset(BaseDataset):

    def _load_meta_data(self, im_data_file):
        meta_data = {}
        meta_data['PATIENT'] = self._extract_patient_num(im_data_file)
        meta_data['PHASE'] = self._extract_phase(im_data_file)
        meta_data['SLICE_NO'] = self._extract_slice_no(im_data_file)
        return meta_data

    def process_meta_data(self, image_meta_data):
        # encode all cfg features
        extra_features_lst = []
        for im_meta_datum in image_meta_data:
            extra_features = []
            # add patient number
            patient_num = im_meta_datum['PATIENT']
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

    def _extract_patient_num(self, im_path):
        patient_num_string_raw = im_path[self._get_patient_num_start_index(im_path):]
        patient_num_string = patient_num_string_raw[:patient_num_string_raw.index("_")]
        return int(patient_num_string)

    def _get_patient_num_start_index(self, im_path):
        patient_prefix = "DET"
        patient_prefix_len = len(patient_prefix)
        patient_prefix_index = im_path.index(patient_prefix)
        patient_num_start_index = patient_prefix_index + patient_prefix_len
        return patient_num_start_index

    def _extract_phase(self, im_path):
        phase_string_raw = im_path[self._get_phase_start_index(im_path):]
        phase_string = phase_string_raw[:phase_string_raw.index("_")]
        vol_type = phase_string[:2]
        vol_num = int(phase_string[2:])
        if vol_type == "LA":
            return vol_num
        elif vol_type == "SA":
            # vol number doesn't go over 100, so should be find adding this amt to get unique ids
            return vol_num + 100
        else:
            raise ValueError("Volume type not recognized")

    def _get_phase_start_index(self, im_path):
        phase_prefix = "_"
        phase_prefix_len = len(phase_prefix)
        phase_prefix_index = im_path.index(phase_prefix_len)
        phase_start_index = phase_prefix_index + phase_prefix_len
        return phase_start_index

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

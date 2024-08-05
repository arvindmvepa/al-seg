from active_learning.dataset.base_dataset import BaseDataset
import numpy as np
from collections import defaultdict


class ACDC_Dataset(BaseDataset):

    def _load_meta_data(self, im_data_file):
        meta_data_file = self._get_meta_data_path(im_data_file)
        meta_data = {}
        with open(meta_data_file, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                if key == 'ED' or key == 'ES':
                    value = int(value)
                if key == 'Height' or key == 'Weight':
                    value = float(value)
                meta_data[key] = value
        return meta_data

    def _get_meta_data_path(self, im_path):
        meta_data_path = self._extract_patient_prefix(im_path) + ".cfg"
        return meta_data_path

    def process_meta_data(self, image_meta_data):
        # calculate number of slices per frame
        num_slices_dict = dict()
        for file_name in self.all_train_im_files:
            patient_frame_no = self._extract_patient_frame_no_str(file_name)
            if patient_frame_no not in num_slices_dict:
                num_slices_dict[patient_frame_no] = 1
            else:
                num_slices_dict[patient_frame_no] += 1

        # calculate number of groups
        groups_dict = defaultdict(lambda: len(groups_dict))
        for im_meta_datum in image_meta_data:
            groups_dict[im_meta_datum['Group']]
        self.num_groups = len(groups_dict)
        one_hot_group = [0] * self.num_groups

        # calculate z-score params for height and weight
        height_mean = 0
        height_sstd = 0
        weight_mean = 0
        weight_sstd = 0
        height_lst = [im_meta_datum['Height'] for im_meta_datum in image_meta_data]
        weight_lst = [im_meta_datum['Weight'] for im_meta_datum in image_meta_data]
        height_mean = np.mean(height_lst)
        weight_mean = np.mean(weight_lst)
        height_sstd = np.std(height_lst)
        weight_sstd = np.std(weight_lst)

        # Generate histogram of height and weight
        num_bins = 15
        for values, name in zip([height_lst, weight_lst], ['Height', 'Weight']):
            print("Histogram for {}".format(name))
            # Calculate the bin edges using percentiles
            percentiles = np.linspace(0, 100, num_bins + 1)
            bin_edges = np.percentile(values, percentiles)

            # Count occurrences in each bin
            hist, _ = np.histogram(values, bins=bin_edges)

            # Generate text version of the histogram
            for i in range(len(hist)):
                bin_range = f'{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}'
                print(f'{bin_range}: #: {hist[i]}')

        # encode all cfg features
        extra_features_lst = []
        for im_meta_datum, file_name in zip(image_meta_data, self.all_train_im_files):
            extra_features = []
            # add patient number
            patient_num = self._extract_patient_num(file_name)
            extra_features.append(patient_num)
            # add if frame is ED or ES (one hot encoded)
            patient_frame_no = self._extract_patient_frame_no_str(file_name)
            frame_num = self._extract_frame_no(file_name)
            if im_meta_datum['ED'] == frame_num:
                extra_features.append(1)
            elif im_meta_datum['ES'] == frame_num:
                extra_features.append(0)
            else:
                raise ValueError("Frame number not found in ED or ES")
            # add Group ID (one hot encoded)
            group_id = groups_dict[im_meta_datum['Group']]
            im_one_hot_group = one_hot_group.copy()
            im_one_hot_group[group_id] = 1
            extra_features.extend(im_one_hot_group)

            # add Height and Weight
            z_score_height = (im_meta_datum['Height'] - height_mean) / height_sstd
            z_score_weight = (im_meta_datum['Weight'] - weight_mean) / weight_sstd

            extra_features.append(z_score_height)
            extra_features.append(z_score_weight)

            # add relative slice position and general slice position
            slice_num = self._extract_slice_no(file_name)
            extra_features.append(slice_num / num_slices_dict[patient_frame_no])
            extra_features.append(slice_num)

            """"
            # add pathology group
            if patient_num <= 20:
                extra_features.append(0)
            elif patient_num <= 40:
                extra_features.append(1)
            elif patient_num <= 60:
                extra_features.append(2)
            elif patient_num <= 80:
                extra_features.append(3)
            elif patient_num <= 100:
                extra_features.append(4)
            else:
                raise ValueError("Patient number not found in pathology group")
            """
            # add height group
            """
            height = im_meta_datum['Height']
            if height < 160.0:
                extra_features.append(0)
            elif height < 163.0:
                extra_features.append(1)
            elif height < 166.0:
                extra_features.append(2)
            elif height < 170.0:
                extra_features.append(3)
            elif height < 172.0:
                extra_features.append(4)
            elif height < 175.0:
                extra_features.append(5)
            elif height < 176.0:
                extra_features.append(6)
            elif height < 180.0:
                extra_features.append(7)
            elif height < 184.0:
                extra_features.append(8)
            elif height <= 192.0:
                extra_features.append(9)
            else:
                raise ValueError("Patient height invalid")

            # add weight group
            weight = im_meta_datum['Weight']
            if weight < 55.0:
                extra_features.append(0)
            elif weight < 60.0:
                extra_features.append(1)
            elif weight < 68.0:
                extra_features.append(2)
            elif weight < 74.0:
                extra_features.append(3)
            elif weight < 77.0:
                extra_features.append(4)
            elif weight < 80.0:
                extra_features.append(5)
            elif weight < 85.0:
                extra_features.append(6)
            elif weight < 93.0:
                extra_features.append(7)
            elif weight < 98.4:
                extra_features.append(8)
            elif weight <= 123.0:
                extra_features.append(9)
            else:
                raise ValueError("Patient height invalid")
            """
            height = im_meta_datum['Height']
            if height < 158.0:
                extra_features.append(0)
            elif height < 160.0:
                extra_features.append(1)
            elif height < 163.0:
                extra_features.append(2)
            elif height < 166.0:
                extra_features.append(3)
            elif height < 169.0:
                extra_features.append(4)
            elif height < 170.0:
                extra_features.append(5)
            elif height < 172.0:
                extra_features.append(6)
            elif height < 174.0:
                extra_features.append(7)
            elif height < 175.0:
                extra_features.append(8)
            elif height <= 176.0:
                extra_features.append(9)
            elif height < 178.0:
                extra_features.append(10)
            elif height < 180.0:
                extra_features.append(11)
            elif height < 184.0:
                extra_features.append(12)
            elif height < 186.0:
                extra_features.append(13)
            elif height <= 192.0:
                extra_features.append(14)
            else:
                raise ValueError("Patient height invalid")


            extra_features_lst.append(np.array(extra_features))

        return np.array(extra_features_lst)

    def get_non_image_indices(self):
        non_image_indices = dict()
        non_image_indices['patient_starting_index'] = 0
        non_image_indices['patient_ending_index'] = non_image_indices['patient_starting_index'] + 1
        non_image_indices['phase_starting_index'] = non_image_indices['patient_ending_index']
        non_image_indices['phase_ending_index'] = non_image_indices['phase_starting_index'] + 1
        non_image_indices['group_starting_index'] = non_image_indices['phase_ending_index']
        non_image_indices['group_ending_index'] = non_image_indices['group_starting_index'] + self.num_groups
        non_image_indices['height_starting_index'] = non_image_indices['group_ending_index']
        non_image_indices['height_ending_index'] = non_image_indices['height_starting_index'] + 1
        non_image_indices['weight_starting_index'] = non_image_indices['height_ending_index']
        non_image_indices['weight_ending_index'] = non_image_indices['weight_starting_index'] + 1
        non_image_indices['slice_rel_pos_starting_index'] = non_image_indices['weight_ending_index']
        non_image_indices['slice_rel_pos_ending_index'] = non_image_indices['slice_rel_pos_starting_index'] + 1
        non_image_indices['slice_pos_starting_index'] = non_image_indices['slice_rel_pos_ending_index']
        non_image_indices['slice_pos_ending_index'] = non_image_indices['slice_pos_starting_index'] + 1
        non_image_indices['path_group_starting_index'] = non_image_indices['slice_pos_ending_index']
        non_image_indices['path_group_ending_index'] = non_image_indices['path_group_starting_index'] + 1
        non_image_indices['uncertainty_starting_index'] = non_image_indices['path_group_ending_index']
        non_image_indices['uncertainty_ending_index'] = non_image_indices['uncertainty_starting_index'] + 1
        return non_image_indices


    def _extract_patient_frame_no_str(self, im_path):
        return self._extract_patient_prefix(im_path) + "_" + str(self._extract_frame_no(im_path))

    def _get_patient_num_start_index(self, im_path):
        patient_prefix = "patient"
        patient_prefix_len = len(patient_prefix)
        patient_prefix_index = im_path.index(patient_prefix)
        patient_num_start_index = patient_prefix_index + patient_prefix_len
        return patient_num_start_index

    def _get_patient_num_end_index(self, im_path):
        patient_num_len = 3
        patient_num_start_index = self._get_patient_num_start_index(im_path)
        return patient_num_start_index + patient_num_len

    def _extract_patient_num(self, im_path):
        return int(im_path[self._get_patient_num_start_index(im_path):self._get_patient_num_end_index(im_path)])

    def _extract_patient_prefix(self, im_path):
        patient_prefix_end_index = self._get_patient_prefix_end_index(im_path)
        patient_prefix = im_path[:patient_prefix_end_index]
        return patient_prefix

    def _get_patient_prefix_end_index(self, im_path):
        patient_prefix = "patient"
        patient_prefix_len = len(patient_prefix)
        patient_prefix_index = im_path.index(patient_prefix)
        patient_num_len = 3
        patient_prefix_end_index = patient_prefix_index + patient_prefix_len + patient_num_len
        return patient_prefix_end_index

    def _extract_frame_no(self, im_path):
        frame_prefix = "frame"
        frame_num_len = 2
        frame_and_num_prefix_len = len(frame_prefix) + frame_num_len
        frame_and_num_end_index = im_path.index(frame_prefix) + frame_and_num_prefix_len
        frame_no = im_path[frame_and_num_end_index-frame_num_len:frame_and_num_end_index]
        return int(frame_no)

    def _extract_slice_no(self, im_path):
        slice_str = "slice_"
        slice_str_len = len(slice_str)
        slice_num_len = 2
        slice_start_index = im_path.index(slice_str) + slice_str_len
        slice_end_index = slice_start_index + slice_num_len
        slice_str = im_path[slice_start_index:slice_end_index]
        if slice_str.isdigit():
            slice_num = int(slice_str)
        else:
            slice_str = slice_str[0]
            slice_num = int(slice_str)
        return slice_num
import numpy as np
from numpy.linalg import norm


def euclidean_w_config(image_vec1, image_vec2, num_im_features, phase_starting_index, phase_ending_index,
                       group_starting_index, group_ending_index, height_starting_index, height_ending_index,
                       width_starting_index, width_ending_index, slice_pos_starting_index, slice_pos_ending_index,
                       extra_feature_weight=1.0):
    l2_norm_val = norm(image_vec1[:num_im_features], image_vec2[:num_im_features])
    phase_score = 1 - np.abs(image_vec1[phase_starting_index:phase_ending_index] - image_vec2[phase_starting_index:phase_ending_index])
    group_score = 1 - np.abs(image_vec1[group_starting_index:group_ending_index] - image_vec2[group_starting_index:group_ending_index])
    height_score = np.abs(image_vec1[height_starting_index:height_ending_index] - image_vec2[height_starting_index:height_ending_index])
    width_score = np.abs(image_vec1[width_starting_index:width_ending_index] - image_vec2[width_starting_index:width_ending_index])
    slice_pos_score = 1 - np.abs(image_vec1[slice_pos_starting_index:slice_pos_ending_index] - image_vec2[slice_pos_starting_index:slice_pos_ending_index])
    return l2_norm_val + extra_feature_weight * (phase_score + group_score + height_score + width_score + slice_pos_score)





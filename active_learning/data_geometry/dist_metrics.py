import numpy as np
from sklearn.metrics import DistanceMetric
from scipy.spatial.distance import cosine


def metric_w_config(image_vec1, image_vec2, image_metric, max_dist, wt_max_dist_mult, num_im_features,
                    patient_starting_index, patient_ending_index, phase_starting_index, phase_ending_index,
                    group_starting_index, group_ending_index, height_starting_index, height_ending_index,
                    weight_starting_index, weight_ending_index, slice_rel_pos_starting_index,
                    slice_rel_pos_ending_index, slice_pos_starting_index, slice_pos_ending_index,
                    uncertainty_starting_index, uncertainty_ending_index,  extra_feature_wt=-0.0, patient_wt=0.0,
                    phase_wt=0.0, group_wt=0.0, height_wt=0.0, weight_wt=0.0, slice_rel_pos_wt=0.0, slice_mid_wt=0.0,
                    slice_pos_wt=0.0, uncertainty_wt=0.0):
    assert image_vec1.shape == image_vec2.shape
    if image_metric == "cosine":
        image_metric = lambda x,y: cosine(x,y)
    elif image_metric == "dot":
        image_metric = lambda x,y: -np.dot(x,y)
    else:
        metric_inst = DistanceMetric.get_metric(image_metric)
        image_metric = lambda x,y: metric_inst.pairwise([x], [y])[0]
    metric_val = np.sum(image_metric(image_vec1[:num_im_features], image_vec2[:num_im_features]))
    non_image_vec1, non_image_vec2 = image_vec1[num_im_features:], image_vec2[num_im_features:]
    patient_score = 1 - np.sum(non_image_vec1[patient_starting_index:patient_ending_index] == non_image_vec2[patient_starting_index:patient_ending_index])
    phase_score = np.sum(np.abs(non_image_vec1[phase_starting_index:phase_ending_index] - non_image_vec2[phase_starting_index:phase_ending_index]))
    group_score = 1 - np.dot(non_image_vec1[group_starting_index:group_ending_index], non_image_vec2[group_starting_index:group_ending_index])
    height_score = np.sum(np.abs(non_image_vec1[height_starting_index:height_ending_index] - non_image_vec2[height_starting_index:height_ending_index]))
    weight_score = np.sum(np.abs(non_image_vec1[weight_starting_index:weight_ending_index] - non_image_vec2[weight_starting_index:weight_ending_index]))
    slice_rel_pos_score = np.sum(np.abs(non_image_vec1[slice_rel_pos_starting_index:slice_rel_pos_ending_index] - non_image_vec2[slice_rel_pos_starting_index:slice_rel_pos_ending_index]))
    slice_mid_score = np.sum(1 - np.abs(non_image_vec1[slice_rel_pos_starting_index:slice_rel_pos_ending_index] - 0.5) - np.abs(non_image_vec2[slice_rel_pos_starting_index:slice_rel_pos_ending_index] - 0.5))
    slice_pos_score = 1 - np.sum(non_image_vec1[slice_pos_starting_index:slice_pos_ending_index] == non_image_vec2[slice_pos_starting_index:slice_pos_ending_index])
    uncertainty_score = np.sum((non_image_vec1[uncertainty_starting_index:uncertainty_ending_index] + non_image_vec2[uncertainty_starting_index:uncertainty_ending_index])/2.0)

    # scale all the weights to be less than max_dist * wt_max_dist_mult
    if max_dist is not None:
        wt_max_dist = max_dist * wt_max_dist_mult
        patient_wt = max(min(patient_wt, wt_max_dist), -wt_max_dist)
        phase_wt = max(min(phase_wt, wt_max_dist), -wt_max_dist)
        group_wt = max(min(group_wt, wt_max_dist), -wt_max_dist)
        height_wt = max(min(height_wt, wt_max_dist), -wt_max_dist)
        weight_wt = max(min(weight_wt, wt_max_dist), -wt_max_dist)
        slice_mid_wt= max(min(slice_mid_wt, wt_max_dist), - wt_max_dist)
        slice_rel_pos_wt = max(min(slice_rel_pos_wt, wt_max_dist), -wt_max_dist)
        slice_pos_wt = max(min(slice_pos_wt, wt_max_dist), -wt_max_dist)
        uncertainty_wt = max(min(uncertainty_wt, wt_max_dist), -wt_max_dist)


    patient_score = patient_score * patient_wt
    phase_score = phase_score * phase_wt
    group_score = group_score * group_wt
    height_score = height_score * height_wt
    weight_score = weight_score * weight_wt
    slice_rel_pos_score = slice_rel_pos_score * slice_rel_pos_wt
    slice_mid_score = slice_mid_score * slice_mid_wt
    slice_pos_score = slice_pos_score * slice_pos_wt
    uncertainty_score = uncertainty_score * uncertainty_wt

    non_image_score = extra_feature_wt * (patient_score + phase_score + group_score + height_score + weight_score + slice_rel_pos_score + slice_mid_score + slice_pos_score + uncertainty_score)

    return metric_val + non_image_score
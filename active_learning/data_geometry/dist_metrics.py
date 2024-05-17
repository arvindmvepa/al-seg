import numpy as np
from sklearn.metrics import DistanceMetric
from scipy.spatial.distance import cosine


def metric_w_config(image_vec1, image_vec2, image_metric, max_dist, wt_max_dist_mult, num_im_features, num_label_features,
                    patient_starting_index=None, patient_ending_index=None, phase_starting_index=None,
                    phase_ending_index=None, group_starting_index=None, group_ending_index=None,
                    height_starting_index=None, height_ending_index=None, weight_starting_index=None,
                    weight_ending_index=None, slice_rel_pos_starting_index=None, slice_rel_pos_ending_index=None,
                    slice_pos_starting_index=None, slice_pos_ending_index=None, uncertainty_starting_index=None,
                    uncertainty_ending_index=None, normalize_pos_by_label_ct=False, pos_wt=1.0, extra_feature_wt=0.0,
                    patient_wt=0.0, phase_wt=0.0, group_wt=0.0, height_wt=0.0, weight_wt=0.0, slice_rel_pos_wt=0.0,
                    slice_mid_wt=0.0, slice_pos_wt=0.0, uncertainty_wt=0.0):
    assert image_vec1.shape == image_vec2.shape
    if image_metric == "cosine":
        image_metric = lambda x,y,z: (1 - cosine(x,y))
    elif image_metric == "dot":
        image_metric = lambda x,y,z: -np.dot(x,y*z)
    elif image_metric == "euclidean":
        image_metric = lambda x,y,z: np.sqrt(np.sum(z*((x-y)**2)))
    else:
        raise ValueError("image_metric must be one of 'cosine', 'dot', or 'euclidean'")
    num_position_features = num_label_features
    im1_position_features, im2_position_features = image_vec1[:num_position_features], image_vec2[:num_position_features]
    mdl_features_starting_index = num_position_features
    im1_mdl_features, im2_mdl_features = image_vec1[mdl_features_starting_index:num_im_features], \
                                         image_vec2[mdl_features_starting_index:num_im_features]
    labels_starting_index = num_im_features
    labels = image_vec2[labels_starting_index:labels_starting_index + num_label_features]
    non_image_features_starting_index = labels_starting_index + num_label_features
    non_image_vec1, non_image_vec2 = image_vec1[non_image_features_starting_index:], \
                                     image_vec2[non_image_features_starting_index:]
    if num_position_features > 0:
        print("labels.shape: ", labels.shape)
        print("nonzero labels: ", len(np.flatnonzero(labels)))
        position_metric_val = np.sum(image_metric(im1_position_features, im2_position_features, labels))
        # normalize the position metric by the number of non-zero labels (getting average weighted distance)
        if normalize_pos_by_label_ct:
            position_metric_val /= len(np.flatnonzero(labels))
        position_metric_val *= pos_wt
    else:
        position_metric_val = 0
    mdl_metric_val = np.sum(image_metric(im1_mdl_features, im2_mdl_features, np.ones(im1_mdl_features.shape)))
    metric_val = position_metric_val + mdl_metric_val
    print("position_metric_val: ", position_metric_val)
    print("mdl_metric_val: ", mdl_metric_val)

    if (patient_starting_index is not None) and (patient_ending_index is not None):
        patient_score = 1 - np.sum(non_image_vec1[patient_starting_index:patient_ending_index] == non_image_vec2[patient_starting_index:patient_ending_index])
    else:
        patient_score = 0
    if (phase_starting_index is not None) and (phase_ending_index is not None):
        phase_score = np.sum(np.abs(non_image_vec1[phase_starting_index:phase_ending_index] - non_image_vec2[
                                                                                              phase_starting_index:phase_ending_index]))
    else:
        phase_score = 0
    if (group_starting_index is not None) and (group_ending_index is not None):
        group_score = 1 - np.dot(non_image_vec1[group_starting_index:group_ending_index], non_image_vec2[
                                                                                          group_starting_index:group_ending_index])
    else:
        group_score = 0
    if (height_starting_index is not None) and (height_ending_index is not None):
        height_score = np.sum(np.abs(non_image_vec1[height_starting_index:height_ending_index] - non_image_vec2[
                                                                                                height_starting_index:height_ending_index]))
    else:
        height_score = 0
    if (weight_starting_index is not None) and (weight_ending_index is not None):
        weight_score = np.sum(np.abs(non_image_vec1[weight_starting_index:weight_ending_index] - non_image_vec2[
                                                                                                weight_starting_index:weight_ending_index]))
    else:
        weight_score = 0
    if (slice_rel_pos_starting_index is not None) and (slice_rel_pos_ending_index is not None):
        slice_rel_pos_score = np.sum(np.abs(non_image_vec1[slice_rel_pos_starting_index:slice_rel_pos_ending_index] - non_image_vec2[
                                                                                                                        slice_rel_pos_starting_index:slice_rel_pos_ending_index]))
        slice_mid_score = np.sum(
            1 - np.abs(non_image_vec1[slice_rel_pos_starting_index:slice_rel_pos_ending_index] - 0.5) - np.abs(
                non_image_vec2[slice_rel_pos_starting_index:slice_rel_pos_ending_index] - 0.5))
    else:
        slice_rel_pos_score = 0
        slice_mid_score = 0
    if (slice_pos_starting_index is not None) and (slice_pos_ending_index is not None):
        slice_pos_score = 1 - np.sum(non_image_vec1[slice_pos_starting_index:slice_pos_ending_index] == non_image_vec2[
                                                                                                      slice_pos_starting_index:slice_pos_ending_index])
    else:
        slice_pos_score = 0
    if (uncertainty_starting_index is not None) and (uncertainty_ending_index is not None):
        uncertainty_score = np.sum(np.abs(non_image_vec1[uncertainty_starting_index:uncertainty_ending_index] - non_image_vec2[
                                                                                                                  uncertainty_starting_index:uncertainty_ending_index]))
    else:
        uncertainty_score = 0

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
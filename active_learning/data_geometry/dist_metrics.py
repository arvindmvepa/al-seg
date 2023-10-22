import numpy as np
from sklearn.metrics import DistanceMetric
from scipy.spatial.distance import cosine


def metric_w_config(image_vec1, image_vec2, image_metric, num_im_features, patient_starting_index, patient_ending_index,
                    phase_starting_index, phase_ending_index, group_starting_index, group_ending_index,
                    height_starting_index, height_ending_index, weight_starting_index, weight_ending_index,
                    slice_rel_pos_starting_index, slice_rel_pos_ending_index, slice_pos_starting_index,
                    slice_pos_ending_index, extra_feature_wt=-0.0, patient_wt=0.0, phase_wt=0.0, group_wt=0.0,
                    height_wt=0.0, weight_wt=0.0, slice_pos_wt=0.0):
    if image_metric == "cosine":
        image_metric = lambda x,y: (1 - cosine(x,y))
    elif image_metric == "dot":
        image_metric = lambda x,y: -np.dot(x,y)
    else:
        metric_inst = DistanceMetric.get_metric(image_metric)
        image_metric = lambda x,y: metric_inst.pairwise([x], [y])[0]
    metric_val = image_metric(image_vec1[:num_im_features], image_vec2[:num_im_features])
    patient_score = 1 - np.sum(image_vec1[patient_starting_index:patient_ending_index] == image_vec2[patient_starting_index:patient_ending_index])
    phase_score = np.abs(image_vec1[phase_starting_index:phase_ending_index] - image_vec2[phase_starting_index:phase_ending_index])
    group_score = 1 - np.dot(image_vec1[group_starting_index:group_ending_index], image_vec2[group_starting_index:group_ending_index])
    height_score = np.abs(image_vec1[height_starting_index:height_ending_index] - image_vec2[height_starting_index:height_ending_index])
    weight_score = np.abs(image_vec1[weight_starting_index:weight_ending_index] - image_vec2[weight_starting_index:weight_ending_index])
    slice_pos_score = np.abs(image_vec1[slice_rel_pos_starting_index:slice_rel_pos_ending_index] - image_vec2[slice_rel_pos_starting_index:slice_rel_pos_ending_index])
    return metric_val + extra_feature_wt * (patient_score*patient_wt + phase_score*phase_wt + group_score*group_wt + height_score*height_wt + weight_score*weight_wt + slice_pos_score*slice_pos_wt)




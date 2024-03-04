import os
from active_learning.policy.policy_builder import PolicyBuilder


if __name__ == '__main__':
    root_dir = "."
    model_dirs = [r"/home/asjchoi/al-seg/DMPLS_exp0_coreset_fuse20_pos_loss1_wt02_pos_loss2_wt01_pos_loss3_wt005_use_slice_pos_use_phase_use_patient_uncertainty1_v10/round_1/0",
                  r"/home/asjchoi/al-seg/DMPLS_exp0_coreset_fuse20_pos_loss1_wt02_pos_loss2_wt01_pos_loss3_wt005_use_slice_pos_use_phase_use_patient_uncertainty1_v10/round_2/0",
                  r"/home/asjchoi/al-seg/DMPLS_exp0_coreset_fuse20_pos_loss1_wt02_pos_loss2_wt01_pos_loss3_wt005_use_slice_pos_use_phase_use_patient_uncertainty1_v10/round_3/0",
                  r"/home/asjchoi/al-seg/DMPLS_exp0_coreset_fuse20_pos_loss1_wt02_pos_loss2_wt01_pos_loss3_wt005_use_slice_pos_use_phase_use_patient_uncertainty1_v10/round_4/0",
                  r"/home/amvepa91/al-seg2/DMPLS_exp4_coreset_v10/round_1/0",
                  r"/home/amvepa91/al-seg2/DMPLS_exp4_coreset_v10/round_2/0",
                  r"/home/amvepa91/al-seg2/DMPLS_exp4_coreset_v10/round_3/0",
                  r"/home/amvepa91/al-seg2/DMPLS_exp4_coreset_v10/round_4/0",
                  r"/home/amvepa91/al-seg1/DMPLS_exp3_coregcn_model_fts_ss10_rstart_v10/round_1/0",
                  r"/home/amvepa91/al-seg1/DMPLS_exp3_coregcn_model_fts_ss10_rstart_v10/round_2/0",
                  r"/home/amvepa91/al-seg1/DMPLS_exp3_coregcn_model_fts_ss10_rstart_v10/round_3/0",
                  r"/home/amvepa91/al-seg1/DMPLS_exp3_coregcn_model_fts_ss10_rstart_v10/round_4/0",
                  r"C:\Users\Arvind\Documents\al-seg-arvind2\DMPLS_exp1_random_v10\round_1\0",
                  r"C:\Users\Arvind\Documents\al-seg-arvind2\DMPLS_exp1_random_v10\round_2\0",
                  r"C:\Users\Arvind\Documents\al-seg-arvind2\DMPLS_exp1_random_v10\round_3\0",
                  r"C:\Users\Arvind\Documents\al-seg-arvind2\DMPLS_exp1_random_v10\round_4\0"]
    prediction_ims = ["patient117_frame13.h5", "patient150_frame12.h5", "patient133_frame10.h5"]

    for model_dir in model_dirs:
        model_pth = os.path.join(model_dir, "unet_cct_best_model.pth")
        if not os.path.exists(model_pth):
            continue
        model_no = int(os.path.basename(model_dir))
        round_num = int(os.path.basename(os.path.dirname(model_dir)).split("_")[1])
        exp_dir = os.path.dirname(os.path.dirname(model_dir))
        cur_exp_dir = os.path.join(".", os.path.basename(exp_dir))
        exp_file = os.path.join(cur_exp_dir, "exp.yml")
        if not os.path.exists(exp_file):
            continue
        policy = PolicyBuilder.build_policy(exp_file)
        policy.generate(model_no=model_no, round_num=round_num, prediction_ims=prediction_ims)
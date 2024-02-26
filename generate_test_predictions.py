import os
from active_learning.policy.policy_builder import PolicyBuilder
from glob import glob


root_dir = "."
exp_dirs = sorted(list(glob(os.path.join(root_dir, "*", "DMPLS_*_coreset_fuse20_pos_loss1_wt02_pos_loss2_wt01_pos_loss3_wt005_use_slice_pos_use_phase_use_patient_uncertainty1_v10*"))))
exp_dirs = exp_dirs + sorted(list(glob(os.path.join(root_dir, "*", "DMPLS_*_random*_v10"))))

results = []
for exp_dir in exp_dirs:
        if not os.path.exists(exp_dir):
            continue
        exp_file = os.path.join(exp_dir, "exp.yml")
        policy = PolicyBuilder.build_policy(exp_file)
        policy.evaluate()
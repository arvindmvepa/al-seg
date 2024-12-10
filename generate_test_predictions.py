import os
from active_learning.policy.policy_builder import PolicyBuilder
from glob import glob

if __name__ == '__main__':
    root_dir = "."
    exp_dirs = sorted(list(glob(os.path.join(root_dir, "DMPLS_*"))))
    results = []
    for exp_dir in exp_dirs:
            if not os.path.exists(exp_dir):
                continue
            exp_file = os.path.join(exp_dir, "exp.yml")
            policy = PolicyBuilder.build_policy(exp_file)
            policy.evaluate()

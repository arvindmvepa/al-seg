from active_learning.policy.policy_builder import PolicyBuilder


if __name__ == "__main__":
    exp_files = ["exp0.yml", "exp1.yml", "exp2.yml", "exp3.yml", "exp4.yml", "exp5.yml", "exp6.yml", "exp7.yml",
                 "exp8.yml", "exp9.yml", "exp10.yml", "exp11.yml",  "exp12.yml", "exp13.yml", "exp14.yml", "exp15.yml",
                 "exp16.yml", "exp17.yml", "exp18.yml", "exp19.yml", "exp20.yml"]
    for exp_file in exp_files:
        policy = PolicyBuilder.build_policy(exp_file)
        policy.run()

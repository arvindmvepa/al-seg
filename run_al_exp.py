from active_learning.policy.policy_builder import PolicyBuilder


if __name__ == "__main__":
    exp_files = ["exp.yml", "exp1.yml", "exp2.yml", "exp3.yml", "exp4.yml", "exp5.yml", "exp6.yml"]
    for exp_file in exp_files:
        policy = PolicyBuilder.build_policy(exp_file)
        policy.run()

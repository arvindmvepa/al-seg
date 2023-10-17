from active_learning.policy.policy_builder import PolicyBuilder


if __name__ == "__main__":
    exp_files = ["exp10.yml", "exp11.yml", "exp12.yml", "exp13.yml", "exp14.yml"]
    for exp_file in exp_files:
        policy = PolicyBuilder.build_policy(exp_file)
        policy.run()

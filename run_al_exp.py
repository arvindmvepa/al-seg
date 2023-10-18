from active_learning.policy.policy_builder import PolicyBuilder


if __name__ == "__main__":
    exp_files = ["exp30.yml", "exp31.yml", "exp32.yml", "exp33.yml", "exp34.yml"]
    for exp_file in exp_files:
        policy = PolicyBuilder.build_policy(exp_file)
        policy.run()

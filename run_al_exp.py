from active_learning.policy.policy_builder import PolicyBuilder


if __name__ == "__main__":
    exp_files = ["exp787.yml", "exp788.yml", "exp789.yml", "exp790.yml", "exp791.yml"]
    for exp_file in exp_files:
        policy = PolicyBuilder.build_policy(exp_file)
        policy.run()

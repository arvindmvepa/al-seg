from active_learning.policy.policy_builder import PolicyBuilder


if __name__ == "__main__":
    exp_files = ["exp399.yml", "exp400.yml", "exp401.yml", "exp402.yml", "exp403.yml"]
    for exp_file in exp_files:
        policy = PolicyBuilder.build_policy(exp_file)
        policy.run()

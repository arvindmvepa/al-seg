from active_learning.policy.policy_builder import PolicyBuilder


if __name__ == "__main__":
    indices = list(range(1100, 1105))
    exp_files = [f"exp{index}.yml" for index in indices]
    for exp_file in exp_files:
        policy = PolicyBuilder.build_policy(exp_file)
        policy.run()

from active_learning.policy.policy_builder import PolicyBuilder


if __name__ == "__main__":
    #indices = list(range(1155, 1160))
    #exp_files = [f"exp{index}.yml" for index in indices]
    #for exp_file in exp_files:
    exp_file = "exp_strong_3d.yml"
    policy = PolicyBuilder.build_policy(exp_file)
    policy.run()
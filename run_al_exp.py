from active_learning.policy.policy_builder import PolicyBuilder


if __name__ == "__main__":
    exp_files = ["exp867.yml", "exp868.yml", "exp869.yml", "exp870.yml", "exp871.yml"]
    for exp_file in exp_files:
        policy = PolicyBuilder.build_policy(exp_file)
        policy.run()

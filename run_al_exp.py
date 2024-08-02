from active_learning.policy.policy_builder import PolicyBuilder


if __name__ == "__main__":
    #indices = list(range(1055, 1065))
    #exp_files = [f"exp{index}.yml" for index in indices]
    #for exp_file in exp_files:
    #    policy = PolicyBuilder.build_policy(exp_file)
    #    policy.run()
    exp_file = "exp_resenet50_coreset.yml"
    policy = PolicyBuilder.build_policy(exp_file)
    policy.run()

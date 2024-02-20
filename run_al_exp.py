from active_learning.policy.policy_builder import PolicyBuilder


if __name__ == "__main__":
    exp_files = ["exp468.yml", "exp469.yml", "exp470.yml", "exp471.yml", "exp472.yml",
                 "exp488.yml", "exp489.yml", "exp490.yml", "exp491.yml", "exp492.yml",
                 ]
    for exp_file in exp_files:
        policy = PolicyBuilder.build_policy(exp_file)
        policy.run()

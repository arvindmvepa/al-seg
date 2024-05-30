from active_learning.policy.policy_builder import PolicyBuilder


if __name__ == "__main__":
    exp_files = ["exp934.yml", "exp935.yml",  "exp936.yml", "exp937.yml", "exp938.yml", "exp939.yml", "exp940.yml", "exp941.yml", "exp942.yml", "exp943.yml", "exp944.yml", "exp945.yml", "exp946.yml", "exp947.yml", "exp948.yml"]
    for exp_file in exp_files:
        policy = PolicyBuilder.build_policy(exp_file)
        policy.run()

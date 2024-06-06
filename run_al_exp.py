from active_learning.policy.policy_builder import PolicyBuilder


if __name__ == "__main__":
    exp_files = ["exp974.yml", "exp975.yml", "exp976.yml", "exp977.yml", "exp978.yml",
                 "exp979.yml", "exp980.yml", "exp981.yml", "exp982.yml", "exp983.yml",
                 "exp984.yml", "exp985.yml", "exp986.yml", "exp987.yml", "exp988.yml",
                 "exp989.yml", "exp990.yml", "exp991.yml", "exp992.yml", "exp993.yml"]
    for exp_file in exp_files:
        policy = PolicyBuilder.build_policy(exp_file)
        policy.run()

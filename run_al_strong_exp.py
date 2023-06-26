from active_learning.policy.policy_builder import PolicyBuilder


if __name__ == "__main__":
    policy = PolicyBuilder.build_policy("exp_mac_strong.yml")
    policy.run()
from active_learning.policy.policy_builder import PolicyBuilder
import sys

if __name__ == "__main__":
    policy = PolicyBuilder.build_policy("exp.yml")
    policy.run()
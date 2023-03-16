from active_learning.policy.policy_builder import PolicyBuilder
import sys

if __name__ == "__main__":
    sys.path.append("./spml")
    sys.path.append("./wsl4mis")
    policy = PolicyBuilder.build_policy("exp.yml")
    policy.run()
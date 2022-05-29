from spml.active_learning.policy.base_policy import RandomActiveLearningPolicy
from spml.active_learning.policy.ranked_policy import RankedPolicy


class PolicyFactory:

    def __init__(self):
        pass

    @staticmethod
    def create_policy(policy_type, **policy_kwargs):
        if policy_type == "random":
            policy = RandomActiveLearningPolicy(**policy_kwargs)
        elif policy_type == "ranked":
            policy = RankedPolicy(**policy_kwargs)
        else:
            raise ValueError(f"There is no policy_type {policy_type}")
        return policy
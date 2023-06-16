from strong_supervision.models.model import RunExperiment


if __name__ == '__main__':
    exp = RunExperiment("exp_mac_strong.yml")
    exp.run()
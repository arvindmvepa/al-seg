from strong_supervision.models.model import RunExperiment


if __name__ == '__main__':
    exp = RunExperiment("exp_mac_strong.yml")
    # train_model(arch=model_architect, encoder_name=encoder_name, encoder_weights=encoder_weights,
    #             data_root=data_root, snapshot_dir=snapshot_dir, gpus=gpus)
    exp.run()
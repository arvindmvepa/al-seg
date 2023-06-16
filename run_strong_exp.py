from strong_supervision.models.model import train_model


data_root = '/Users/zukangy/Desktop/projects/active_learning/al-seg/wsl4mis/data/ACDC/'
model_architect = 'Unet'
encoder_name = 'resnet18'
encoder_weights = 'imagenet'
snapshot_dir = './test/'
gpus = 'mps'

train_model(arch=model_architect, encoder_name=encoder_name, encoder_weights=encoder_weights,
            data_root=data_root, snapshot_dir=snapshot_dir, gpus=gpus)
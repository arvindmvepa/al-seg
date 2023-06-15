import seg_models.segmentation_models_pytorch as smp

# Path: strong_supervision/models/net_factory.py

# Create segmentation model with pretrained encoder.
def net_factory(num_classes, encoder_name="resnet18", encoder_weights='imagenet',
                activation='softmax2d'):
    model = smp.FPN(encoder_name=encoder_name, 
                    encoder_weights=encoder_weights,
                    classes=num_classes,
                    activation=activation)
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, 
                                                         encoder_weights)
    return model, preprocessing_fn
    
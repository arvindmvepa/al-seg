import seg_models.segmentation_models_pytorch as smp

# Path: strong_supervision/models/net_factory.py
class StrongModel(object):
    """Create segmentation model with pretrained encoder"""
    def __init__(self, arch='Unet', encoder_name='resnet18', encoder_weights='imagenet', num_classes=4,
                 activation='softmax', in_channels=1, gpus="0"):
        self.arch = arch
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.num_classes = num_classes
        self.activation = activation
        self.in_channels = in_channels
        self.gpus = gpus

    def create_model(self):
        if self.arch == 'Unet':
            model = smp.Unet(encoder_name=self.encoder_name, 
                             encoder_weights=self.encoder_weights,
                             classes=self.num_classes,
                             activation=self.activation,
                             in_channels=self.in_channels)
            
        if self.arch == 'UnetPlusPlus':
            model = smp.UnetPlusPlus(encoder_name=self.encoder_name, 
                                            encoder_weights=self.encoder_weights,
                                            classes=self.num_classes,
                                            activation=self.activation,
                                            in_channels=self.in_channels)
            
        if self.arch == 'MAnet':
            model = smp.MAnet(encoder_name=self.encoder_name, 
                                    encoder_weights=self.encoder_weights,
                                    classes=self.num_classes,
                                    activation=self.activation,
                                    in_channels=self.in_channels)
            
        if self.arch == 'Linknet':
            model = smp.Linknet(encoder_name=self.encoder_name, 
                                        encoder_weights=self.encoder_weights,
                                        classes=self.num_classes,
                                        activation=self.activation,
                                        in_channels=self.in_channels)
            
        if self.arch == 'FPN':
            model = smp.FPN(encoder_name=self.encoder_name, 
                                    encoder_weights=self.encoder_weights,
                                    classes=self.num_classes,
                                    activation=self.activation,
                                    in_channels=self.in_channels)
            
        if self.arch == 'PSPNet':
            model = smp.PSPNet(encoder_name=self.encoder_name, 
                                    encoder_weights=self.encoder_weights,
                                    classes=self.num_classes,
                                    activation=self.activation,
                                    in_channels=self.in_channels)
        
        if self.arch == 'DeepLabV3':
            model = smp.DeepLabV3(encoder_name=self.encoder_name, 
                                        encoder_weights=self.encoder_weights,
                                        classes=self.num_classes,
                                        activation=self.activation,
                                        in_channels=self.in_channels)
            
        if self.arch == 'DeepLabV3Plus':
            model = smp.DeepLabV3Plus(encoder_name=self.encoder_name, 
                                      encoder_weights=self.encoder_weights,
                                      classes=self.num_classes,
                                      activation=self.activation,
                                      in_channels=self.in_channels)
            
        if self.arch == 'PAN':
            model = smp.PAN(encoder_name=self.encoder_name, 
                                    encoder_weights=self.encoder_weights,
                                    classes=self.num_classes,
                                    activation=self.activation,
                                    in_channels=self.in_channels)
        if self.gpus == "mps":
            return model.to("mps")    
        else:
            return model.cuda()
    
    def get_preprocessing_config(self):
        preprocessing_config = smp.encoders.get_preprocessing_fn(self.encoder_name, 
                                                                 self.encoder_weights)
        return preprocessing_config
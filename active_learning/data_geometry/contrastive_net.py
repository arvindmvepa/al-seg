import torch.nn as nn


class ContrastiveLearner(nn.Module):
    """Combined model for contrastive learning based on SimCLR"""

    def __init__(self, encoder, projection_dim=64):
        super(ContrastiveLearner, self).__init__()
        self.encoder = encoder
        encoder_out_dim = self.encoder.resnet.fc.in_features
        self.projection_head = ProjectionHead(encoder_out_dim, projection_dim)

    def forward(self, x_i, x_j=None):
        if self.training:
            h_i = self.encoder(x_i)
            h_j = self.encoder(x_j)

            z_i = self.projection_head(h_i)
            z_j = self.projection_head(h_j)
            return z_i, z_j
        else:
            return self.encoder(x_i)


class ProjectionHead(nn.Module):
    """Head for contrastive learning"""

    def __init__(self, input_dim=128, output_dim=64):
        super(ProjectionHead, self).__init__()

        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        print(x.shape)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

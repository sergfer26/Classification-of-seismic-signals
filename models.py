import torch
from torch import nn
import torch.nn.functional as F


## Defining the architecture of the Encoder.
class Encoder(nn.Module):

    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()

        ## Convolutional layer
        self.Conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(12,6), stride=(4, 2), padding=(2, 1))

        ## Apply Instance Normalization.
        self.Bn1 = nn.InstanceNorm2d(num_features=32)

        ## Convolutional layer
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1)

        ## Apply Instance Normalization.
        self.Bn2 = nn.InstanceNorm2d(num_features=64)

        ## Convolutional layer
        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)

        ## Apply Instance Normalization.
        self.Bn3 = nn.InstanceNorm2d(num_features=32)
        ## Fully Connected Layer.
        self.Fc = nn.Linear(in_features=32 * 62 * 62, out_features=16) # Previously calculated

    ## Forward Pass.
    def forward(self, x):
        x = F.dropout(F.leaky_relu(self.Bn1(self.Conv1(x))), p=0.1, training=self.training)
        x = F.dropout(F.leaky_relu(self.Bn2(self.Conv2(x))), p=0.1, training=self.training)
        x = F.dropout(F.leaky_relu(self.Bn3(self.Conv3(x))), p=0.1, training=self.training)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return torch.tanh(self.Fc(x))


class Decoder(nn.Module):

    def __init__(self, out_channels=3):
        super(Decoder, self).__init__()

        ## Fully Connected Layer.
        self.Fc = nn.Linear(in_features=16, out_features=32 * 62 * 62)

        ## Apply Instance Normalization.
        self.Bn3 = nn.InstanceNorm2d(num_features=32)

        ## Convolutional layer
        self.Conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)

        ## Apply Instance Normalization.
        self.Bn2 = nn.InstanceNorm2d(num_features=64)
        
        ## Convolutional layer
        self.Conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=1)

        ## Apply Instance Normalization.
        self.Bn1 = nn.InstanceNorm2d(num_features=32)

        ## Convolutional layer
        self.Conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=out_channels, kernel_size=(12,6), stride=(4, 2), padding=(2, 1))

    ## Forward Pass.
    def forward(self, z):
        z = self.Fc(z)
        z = F.dropout(torch.tanh(self.Bn3(z.view(z.size(0), 32, 62, 62))), p=0.1, training=self.training)
        z = F.dropout(F.leaky_relu(self.Bn2(self.Conv3(z))), p=0.1, training=self.training)
        z = F.dropout(F.leaky_relu(self.Bn1(self.Conv2(z))), p=0.1, training=self.training)
        z = F.leaky_relu(self.Conv1(z))
        return z


## Wrapper class for the entire autoencoder.
class AutoEncoder(nn.Module):

    def __init__(self, images=3):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(in_channels=images)
        self.decoder = Decoder(out_channels=images)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        Encoding = self.encoder(x)
        Reconstruction = self.decoder(Encoding)
        return Encoding, Reconstruction

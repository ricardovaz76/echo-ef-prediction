import torch.nn as nn


# Container for both ResNet encoder and UNet decoder
class UNet(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):

        # encode forward pass
        # returns a list of feature maps for skip connections
        f1, f2, f3 = self.encoder(x)

        # decode forward pass
        out = self.decoder(f3, f2, f1)

        return out

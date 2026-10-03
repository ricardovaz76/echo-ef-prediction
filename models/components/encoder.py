import torch.nn as nn
import torchvision


# Replaces all ResNet batchnorms with GroupNorms since we are working with batchsize of 1
def replace_bn_with_gn(model, num_groups=32):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features

            # num_groups must divide num_channels evenly
            groups = num_groups
            while num_channels % groups != 0:
                groups -= 1
            setattr(model, name, nn.GroupNorm(groups, num_channels))
        else:
            replace_bn_with_gn(module, num_groups)
    return model


class ResNetEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Initialize ResNet
        resnet = torchvision.models.resnet34(weights=None)

        # Alter ResNet convolution to accept
        # grayscale channel
        resnet.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Initial downsample block
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        # ResNet Downsampling
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        # x: (B*T, 1, 112, 112)

        x = self.stem(x)      # -> 28x28
        f1 = self.layer1(x)   # -> 28x28
        f2 = self.layer2(f1)  # -> 14x14
        f3 = self.layer3(f2)  # -> 7x7

        return f1, f2, f3

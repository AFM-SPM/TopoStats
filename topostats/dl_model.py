"""ResUnet model for image segmentation."""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block.

    Preserves the spatial dimensions as it uses padding and has a stride of 1.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # was told not to use inplace = True here, unsure why. Allegedly it's because pytorch saves memory
        # by overwriting values of a tensor, and given that we use the input in the residual connection, it might
        # cause issues if the input is overwritten.
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut link
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            # If the channels are different, apply a 1x1 conv to match the dimensions
            # this will cause the input to be transformed to the same shape as the output of the block
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # create the residual connection
        residual = self.shortcut(x)
        # apply first conv, batchnorm, and relu
        out = self.relu(self.bn1(self.conv1(x)))
        # apply second conv and batchnorm
        out = self.bn2(self.conv2(out))
        # add the residual connection to the output - since the dimensions are the same, we can add them directly
        out += residual
        # apply relu to the output
        out = self.relu(out)
        return out


class ResUNet(nn.Module):
    """ResUnet model for image segmentation."""

    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()
        # downsampling - encoder
        self.encoder1 = ResBlock(in_channels, out_channels=64)
        # pooling layer to reduce the spatial dimensions by half
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = ResBlock(64, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = ResBlock(128, out_channels=256)

        # upsampling - decoder
        # stage 2
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # reduce the number of channels to match the encoder output for concatenation - allows the
        # skip connection to not have to apply a conv layer to match the dimensions, so it can be just a
        # pure addition of the two feature maps - i think.
        self.conv_reduce2 = nn.Conv2d(256, 128, kernel_size=1)
        self.decoder2 = ResBlock(128, out_channels=128)

        # stage 1
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_reduce1 = nn.Conv2d(128, 64, kernel_size=1)
        self.decoder1 = ResBlock(64, out_channels=64)

        # final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # encoder
        # apply the first encoder block and then pool the output to reduce spatial dimensions
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)

        # repeat
        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2)

        # bottleneck
        bottleneck = self.bottleneck(pool2)

        # decoder
        # upsample the bottleneck output to match the spatial dimensions of the encoder output
        dec2 = self.upconv2(bottleneck)
        # concatenate the output of the upconv with the corresponding encoder output to form a skip connection
        # UNet skip connection
        dec2 = torch.cat((dec2, enc2), dim=1)  # concatenate along the channel dimension -
        # 128 + 128 = 256 channels
        dec2 = self.conv_reduce2(dec2)  # reduce the number of channels to match the encoder output
        # for concatenation, allowing the residual connection to be a pure addition of the two feature maps
        # input: 256 channels,output: 128 channels
        dec2 = self.decoder2(dec2)  # apply the decoder block to the concatenated feature maps -
        # input: 128 channels, output: 128 channels

        # repeat
        dec1 = self.upconv1(dec2)
        # UNet skip connection
        dec1 = torch.cat((dec1, enc1), dim=1)  # concatenate along the channel dimension -
        # 64 + 64 = 128 channels
        dec1 = self.conv_reduce1(dec1)  # reduce the number of channels to match the encoder output
        # for concatenation, allowing the residual connection to be a pure addition of the two feature maps
        # input: 128 channels,output: 64 channels
        dec1 = self.decoder1(dec1)  # apply the decoder block to the concatenated feature maps -
        # input: 64 channels, output: 64 channels

        # final output layer
        out = self.final_conv(dec1)  # apply the final conv layer to get the desired number of output channels
        return out

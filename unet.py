# Code generate with help of YouTube video "Implementing original U-Net from scratch using PyTorch"
# by Abhishek Thakur. Thanks for the great video! From there I started modifying the base U-Net 
# model and adding supporting modules.

import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
# from pytorch_model_summary import summary

def double_conv(in_c, out_c):
    """ Performs 2 3x3 convolutions with ReLU activation """
    conv = nn.Sequential(
        # Consider using 'padding' and 'padding_mode' to implement edge handling.
        # Also, what are dilation and bias? 
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True)
    )
    
    return conv

def crop_and_copy(enc_tensor, dec_tensor):
    enc_size = enc_tensor.size()[2] # grab the spatial size of the tensor on encoding side of U-Net
    dec_size = dec_tensor.size()[2] # grab the spatial size of the tensor on decoding side of U-Net

    delta = (enc_size - dec_size) // 2

    return torch.cat([enc_tensor[:, :, delta:enc_size-delta, delta:enc_size-delta], dec_tensor], 1)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv_1 = double_conv(4, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_transp_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_transp_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_transp_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_transp_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.up_conv_1 = double_conv(1024, 512)
        self.up_conv_2 = double_conv(512, 256)
        self.up_conv_3 = double_conv(256, 128)
        self.up_conv_4 = double_conv(128, 64)

        self.out_conv = nn.Conv2d(in_channels=64, out_channels=9, kernel_size=1, stride=1)

    def forward(self, image):
        """ Implements U-Net model forward propogation. """

        # Consider replacing the Encoding layers with a pre-trained model like Resnet trained on 
        # ImageNet (this will provide generic feature extraction) and then write a decoder that
        # takes the outputs from pre-trained encoder and extracts the features of interest.

        # Encoding layer 1 
        x1 = self.down_conv_1(image) # x1 passed across network to decoding side
        x2 = self.max_pool_2x2(x1)

        # Encoding layer 2
        x3 = self.down_conv_2(x2) # x3 passed across network to decoding side
        x4 = self.max_pool_2x2(x3)

        # Encoding layer 3
        x5 = self.down_conv_3(x4) # x5 passed across network to decoding side
        x6 = self.max_pool_2x2(x5)

        # Encoding layer 4
        x7 = self.down_conv_4(x6) # x7 passed across network to decoding side
        x8 = self.max_pool_2x2(x7)

        # Encoding layer 5
        x9 = self.down_conv_5(x8)

        # Decoding layer 1
        y1 = self.up_transp_1(x9)
        y2 = self.up_conv_1(crop_and_copy(x7, y1))

        # Decoding layer 2
        y3 = self.up_transp_2(y2)
        y4 = self.up_conv_2(crop_and_copy(x5, y3))

        # Decoding layer 3
        y5 = self.up_transp_3(y4)
        y6 = self.up_conv_3(crop_and_copy(x3, y5))

        # Decoding layer 4
        y7 = self.up_transp_4(y6)
        y8 = self.up_conv_4(crop_and_copy(x1, y7))

        out_map = self.out_conv(y8)

        print(out_map.size())


if __name__ == "__main__":
    # 572 x 572 is used because it allows us to go to depth 32x32 (5 layers).
    # If we are going to use 572 x 572 images for U-Net, then I need to add a pipeline stage that
    # grabs the surrounding 60 pixels (original = 512x512) if available or zeros if not.
    # This can be accomplished using the field id and xy coordinates.
    # Otherwise, we need to implement edge handling for the U-Net model.
    image = torch.rand((1, 4, 512, 512))

    unet = UNet()
    # print(summary(unet, image))
    unet(image)
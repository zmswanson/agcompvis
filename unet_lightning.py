import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# from pytorch_model_summary import summary
from pytorch_lightning.metrics import functional as FM

def double_conv(in_channels, out_channels):
    """ Performs 2 3x3 convolutions with ReLU activation """
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True)
    )
    
    return conv

class DownBlock(nn.Module):
    """
    Helper module to implement the downward blocks of the U-Net Model.
    Conducts two 3x3 convolution with ReLU activation and one 2x2 max pooling.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv = double_conv(self.in_channels, self.out_channels)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, input):
        conv_out = self.conv(input) # We need to save the convolution output to pass across the 'U'
        if self.pooling:
            output = self.pool(conv_out)
        else:
            output = conv_out
        return output, conv_out


class UpBlock(nn.Module):
    """
    Helper module to implement the upward blocks of the U-Net Model.
    Conducts two 3x3 convolutions with ReLU activation and one 2x2 transposed (up) convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = double_conv(self.in_channels, self.out_channels)
        self.up_conv = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=2, stride=2)

    def forward(self, up_input, down_input):
        output = self.up_conv(up_input)

        # Concatenate the up convolution ouput and the corresponding down block output
        # We don't have to perform cropping because we are using padding in the 3x3 convolutions.
        output = self.conv(torch.cat([down_input, output], 1))
        return output

class LitUNet(pl.LightningModule):
    """
    PyTorch Lightning implementation of the original U-Net model with some modifications.
    """
    def __init__(self, in_channels=4, depth=5, start_channels=64, num_classes=9):
        super(LitUNet, self).__init__()

        self.in_channels = in_channels
        self.depth = depth
        self.start_channels = start_channels
        self.num_classes = num_classes

        self.down_convs = []
        self.up_convs = []

        in_chans = 0
        out_chans = 0

        for i in range(self.depth):
            in_chans = self.in_channels if i == 0 else out_chans  # first layer is rgb/nir inputs, then number of output channels maps to number of input channels
            out_chans = self.start_channels * (2**i)              # number of output channels double every layer

            pooling = True if i < (self.depth - 1) else False     # there is no pooling on the final down convolution layer
            self.down_convs.append(DownBlock(in_channels=in_chans, out_channels=out_chans, pooling=pooling)) # append the next down convolution layer

        # Based how the up/down helpers are structured, there is one less up layer than down layers 
        for i in range(self.depth -1):
            in_chans = out_chans
            out_chans = in_chans // 2

            self.up_convs.append(UpBlock(in_channels=in_chans, out_channels=out_chans)) # append the next up convolution layer

        # Create the final 1x1 convolution
        self.conv1x1 = nn.Conv2d(in_channels=out_chans, out_channels=self.num_classes, kernel_size=1, stride=1)
        self.softmax = nn.Softmax2d()

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, x):
        downblk_outs = []
        
        for downblk in self.down_convs:
            x, conv_out = downblk(x)
            downblk_outs.append(conv_out)

        for i, upblk in enumerate(self.up_convs):
            # The final down layer outputs don't get passed across the 'U'. Thus, as we work back 
            # up the 'U' we work backwards through the list of downblock ouputs. Hence, -(i + 2).
            x = upblk(x, downblk_outs[-(i + 2)])

        x = self.conv1x1(x)
        # x = self.softmax(x)
        # x = torch.argmax(x, dim=1, keepdim=True)

        return x
        
    def training_step(self, batch, batch_nb):
        x, y, b = batch
        y_hat = self.forward(x)
        y_hat = y_hat * b
        loss = F.cross_entropy(y_hat, y, weight=torch.FloatTensor([1.0, 1.64, 1.0, 1.93, 1.12, 2.54, 2.79, 2.07, 1.22])) 
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y, b = batch
        y_hat = self.forward(x)
        y_hat = y_hat * b
        loss = F.cross_entropy(y_hat, y, weight=torch.FloatTensor([1.0, 1.64, 1.0, 1.93, 1.12, 2.54, 2.79, 2.07, 1.22]))
        self.log('val_loss', loss)
        return {'val_loss': loss}

    # def validation_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-8)


if __name__ == "__main__":
    image = torch.rand((2, 4, 512, 512))

    unet = LitUNet()
    # print(summary(unet, image))
    x = unet(image)
    print(x)
import torch
import torchvision

import pandas as pd
import numpy as np
from torchvision.models import resnet18


""""
NOTES:

1) Batch normalization is a technique to standardize the inputs to a network, 
applied to ether the activations of a prior layer or inputs directly. 
Batch normalization accelerates training, in some cases by halving the epochs or better, 
and provides some regularization, reducing generalization error.

"""
#the class inherites torch.nn.Module, which is a base class for all neural network modules
#my guess block will be used in U-net construction.
#Block consists of 2 conv2 operations (which is followed by maxpooling in the contractive path and interpolation in expansive path)
class Block(torch.nn.Module):
    #initialisef for the model
    def __init__(self, in_channels, mid_channel, out_channels, batch_norm=False):
        super().__init__()
        #
        #defining convolution function
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channel, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, padding=1)
        
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(mid_channel)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)

    # does convolution to return the output to the next layer/block
    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        #relu applies the rectified linear unit function element-wise.
        #basically relu outputs x if it positive and o if it is negative.

        #Why does it assign the output back to x if the function is done in-place? 
        x = torch.nn.functional.relu(x, inplace=True)
        
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        out = torch.nn.functional.relu(x, inplace=True)
        return out
    

class UNet(torch.nn.Module):
    #extracting path
    def up(self, x, size):
        #interpolation is used to get more data using exsiting one and hence ouput will have greater
        #dimensions than the input
        return torch.nn.functional.interpolate(x, size=size, mode=self.upscale_mode)
        #return torch.nn.ConvTranspose2d(x, size = size)
    #contraction path
    def down(self, x):
        #maxpooling is used to select key features and hence narrow down the output
        #kernel size = 2
        return torch.nn.functional.max_pool2d(x, kernel_size=2)
    
    #initialising the U-net model using the Block class
    #what is upscale_mode? I guess something that has to do with expansive path
    def __init__(self, in_channels, out_channels, batch_norm=False, upscale_mode="nearest"):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode
        
        #it seems that we actually first do expansion(or interpolaciton in the encoding part)
        #input_channels refere to RGB(3) or GreyScale(1)(depends how many different channels compose the image) - prob wrong
        #I am a bit confused by this part
        # I found an explanation for that: as we go down the contractive path, we height and width by a factor of 2.
        #Yet, we increase the number of channels by a factor of 2. I guess channels kinda represent the number of 
        #copies of the current tensos/array of an image at given block
        self.enc1 = Block(in_channels, 64, 64, batch_norm)
        self.enc2 = Block(64, 128, 128, batch_norm)
        self.enc3 = Block(128, 256, 256, batch_norm)
        self.enc4 = Block(256, 512, 512, batch_norm)
        
        self.center = Block(512, 1024, 512, batch_norm)
        
        self.dec4 = Block(1024, 512, 256, batch_norm)
        self.dec3 = Block(512, 256, 128, batch_norm)
        self.dec2 = Block(256, 128, 64, batch_norm)
        self.dec1 = Block(128, 64, 64, batch_norm)
        
        self.out = torch.nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    #forward performs one cycle of U-net(getting from input to ouput for a batch of data(epoch?))
    def forward(self, x):
        #we follow 2 conv by maxpooling 
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.down(enc1))
        enc3 = self.enc3(self.down(enc2))
        enc4 = self.enc4(self.down(enc3))
        
        # in the center we also follow 2 conv by maxpooling (Now always the case i think)
        center = self.center(self.down(enc4))
        

        dec4 = self.dec4(torch.cat([self.up(center, enc4.size()[-2:]), enc4], 1))
        dec3 = self.dec3(torch.cat([self.up(dec4, enc3.size()[-2:]), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up(dec3, enc2.size()[-2:]), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up(dec2, enc1.size()[-2:]), enc1], 1))
        
        out = self.out(dec1)
        
        return out
    

#
class PretrainedUNet(torch.nn.Module):
    def up(self, x, size):
        return torch.nn.functional.interpolate(x, size=size, mode=self.upscale_mode)
    
    def down(self, x):
        return torch.nn.functional.max_pool2d(x, kernel_size=2)
    
    def __init__(self, in_channels, out_channels, batch_norm=False, upscale_mode="nearest"):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode
        
        self.init_conv = torch.nn.Conv2d(in_channels, 64, 1)
        #VGG11 helps to solve the problem of overfitting
        #here it is done as we go down the contractive path
        ###endcoder = torchvision.models.vgg11(pretrained=True).features
        endcoder = resnet18(pretrained = True)
        #VGG11 - https://arxiv.org/pdf/1409.1556.pdf paper
        #There are 11 layers in the latest VGG model ()
        #by calling features we casses all the layers in the array 
        """self.conv1 = endcoder[0]   # 64
        self.conv2 = endcoder[3]   # 128
        self.conv3 = endcoder[6]   # 256
        self.conv3s = endcoder[8]  # 256
        self.conv4 = endcoder[11]   # 512
        self.conv4s = endcoder[13]  # 512
        self.conv5 = endcoder[16]  # 512
        self.conv5s = endcoder[18] # 512"""
        self.conv1 = endcoder.layer1 #64 64
        self.conv2 = endcoder.layer2 #64 128
        self.conv3 = endcoder.layer3 #128 256 
        self.conv4 = endcoder.layer4 #256 512
    
        self.center = Block(512, 512, 256, batch_norm)
        
        self.dec5 = Block(512 + 256, 512, 256, batch_norm)
        self.dec4 = Block(512 + 256, 512, 128, batch_norm)
        self.dec3 = Block(256 + 128, 256, 64, batch_norm)
        self.dec2 = Block(128 + 64, 128, 32, batch_norm)
        self.dec1 = Block(64 + 32, 64, 32, batch_norm)
        
        self.out = torch.nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)
    #forward performs one cycle of the pretrained U-net
    def forward(self, x):  
        init_conv = torch.nn.functional.relu(self.init_conv(x), inplace=True)

        """enc1 = torch.nn.functional.relu(self.conv1(init_conv), inplace=True)
        enc2 = torch.nn.functional.relu(self.conv2(self.down(enc1)), inplace=True)
        enc3 = torch.nn.functional.relu(self.conv3(self.down(enc2)), inplace=True)
        enc3 = torch.nn.functional.relu(self.conv3s(enc3), inplace=True)
        enc4 = torch.nn.functional.relu(self.conv4(self.down(enc3)), inplace=True)
        enc4 = torch.nn.functional.relu(self.conv4s(enc4), inplace=True)
        enc5 = torch.nn.functional.relu(self.conv5(self.down(enc4)), inplace=True)
        enc5 = torch.nn.functional.relu(self.conv5s(enc5), inplace=True)"""
        enc1 = self.conv1(init_conv)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        
        center = self.center(self.down(enc4))
        
        dec5 = self.dec5(torch.cat([self.up(center, enc4.size()[-2:]), enc4], 1))
        dec4 = self.dec4(torch.cat([self.up(dec5, enc4.size()[-2:]), enc4], 1))
        dec3 = self.dec3(torch.cat([self.up(dec4, enc3.size()[-2:]), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up(dec3, enc2.size()[-2:]), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up(dec2, enc1.size()[-2:]), enc1], 1))
        
        out = self.out(dec1)
        
        return out
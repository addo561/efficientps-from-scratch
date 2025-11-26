from torch import nn
import torch
from models.backbone.efficient_b0 import  MultiScaleFeatureExtractor
###FPN network  ,Top down

#multiscale features
features = MultiScaleFeatureExtractor()
c2,c3,c4,c5 =  features.forward(torch.rand(1,3,224,224))
print(c2.shape,c3.shape,c4.shape,c5.shape) #check input channels

#from c5 -> c2 downsampled 
class FPN(nn.Module):
    def __init__(self,output_channels=256):
        super(FPN,self).__init__()
        self.c5_conv = nn.Conv2d(in_channels=input_channels,
                                 out_channels=output_channels,
                                 kernel_size=1,
                                 stride=1)
        self.c4_conv = nn.Conv2d(in_channels=input_channels,
                                 out_channels=output_channels,
                                 kernel_size=1,
                                 stride=1)
        self.c3_conv = nn.Conv2d(in_channels=input_channels,
                                 out_channels=output_channels,
                                 kernel_size=1,
                                 stride=1)
        self.c2_conv = nn.Conv2d(in_channels=input_channels,
                                 out_channels=output_channels,
                                 kernel_size=1,
                                 stride=1)
        #use batchnorm + silu instead of  iABN Sync
        self.bn  = nn.BatchNorm2d(output_channels)
        self.silu = nn.SiLU(inplace=True)
    def forward(self,x):
        c2,c3,c4,c5 = MultiScaleFeatureExtractor().forward(x)
        block_5 = self.bn(self.silu(self.c5_conv(c5)))
        print(block_5.shape)

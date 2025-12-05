###FPN network  ,Top down

from torch import nn
import torch
from models.backbone.efficient_b0 import  MultiScaleFeatureExtractor

#multiscale features
features = MultiScaleFeatureExtractor()
c2,c3,c4,c5 =  features.forward(torch.rand(1,3,224,224))
#print(c2.shape,c3.shape,c4.shape,c5.shape) #check input channels

#from c5 -> c2 downsampled 
class FPN(nn.Module):
    def __init__(self,output_channels=256):
        super(FPN,self).__init__()
        self.extractor = MultiScaleFeatureExtractor()
        self.c5_conv = nn.Conv2d(in_channels=320,
                                 out_channels=output_channels,
                                 kernel_size=1,
                                 stride=1)
        self.c4_conv = nn.Conv2d(in_channels=40,
                                 out_channels=output_channels,
                                 kernel_size=1,
                                 stride=1)
        self.c3_conv = nn.Conv2d(in_channels=24,
                                 out_channels=output_channels,
                                 kernel_size=1,
                                 stride=1)
        self.c2_conv = nn.Conv2d(in_channels=16,
                                 out_channels=output_channels,
                                 kernel_size=1,
                                 stride=1)
        #use batchnorm + silu instead of  iABN Sync
        self.norm_act = nn.Sequential(
                    nn.BatchNorm2d(output_channels),
                    nn.SiLU(inplace=True)
        )

        self.L_relu = nn.LeakyReLU(0.1,inplace=True)
        self.upsample_scale_4  = nn.Upsample(scale_factor=4)
        self.upsample_scale_2 = nn.Upsample(scale_factor=2)
    def forward(self,x):
        c2,c3,c4,c5 = self.extractor(x)
        block_5 = self.L_relu(self.norm_act(self.c5_conv(c5)))#(1,256,7,7)
        block_4 = self.L_relu(self.norm_act(self.c4_conv(c4)))#(1,256,28,28)
        block_3 = self.L_relu(self.norm_act(self.c3_conv(c3)))#(1,256,56,56)
        block_2 = self.L_relu(self.norm_act(self.c2_conv(c2)))#(1,256,112,112)
        P5 = block_5
        #Upsample block 5 and add to  block4
        P5_up =  self.upsample_scale_4(P5)#(1,256,28,28)
        P4 = P5_up + block_4 # shape (1,256,28,28)

        #upsample P4  and add to block3
        P4_up = self.upsample_scale_2(P4) #(1,256,56,56)
        P3 = P4_up + block_3#(1,256,56,56)

        #upsample P3 and add to block2
        P3_up = self.upsample_scale_2(P3)#(1,256,112,112)
        P2 = P3_up + block_2#(1,256,112,112)

        return P2,P3,P4,P5

      
        
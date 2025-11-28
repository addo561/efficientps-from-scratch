from torch  import nn
import torch
import torch.nn.functional as F
from torchvision import models
from models.neck.fpn import FPN

### BiFPNt -> p4,p8,p16,p32
class biFPN(FPN):
    def __init__(self):
        super(biFPN,self).__init__()
        #separable convolution
        
        

    def forward(self,x):
        c2,c3,c4,c5 = self.extractor(x)
        block_5 = self.bn(self.silu(self.c5_conv(c5)))#(1,256,7,7)
        block_4 = self.bn(self.silu(self.c4_conv(c4)))#(1,256,28,28)
        block_3 = self.bn(self.silu(self.c3_conv(c3)))#(1,256,56,56)
        block_2 = self.bn(self.silu(self.c2_conv(c2)))#(1,256,112,112)
        P2 = block_2
        #Downsample block 2 and add to  block3
        downsample2 = nn.Upsample(scale_factor=0.5)
        P2_down =  downsample2(P2)#(1,256,56,56)
        P3 = P2_down + block_3 # shape (1,256,56,56)

        #downsample P3 and add to block4
        downsample2 = nn.Upsample(scale_factor=2)
        P3_down = downsample2(P3) #(1,256,28,28)
        P4 = P3_down + block_4#(1,256,28,28)

        #downsample P4 and add to block5
        P4_down = nn.Upsample(scale_factor=0.25)(P4)#(1,256,7,7)
        P5 = P4_down + block_5#(1,256,7,7)
        
        #call  forward method for FPN
        fpn2,fpn3,fpn4,fpn5 = super().forward(x)




        
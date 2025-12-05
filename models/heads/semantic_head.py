import torch
from torch import nn
from models.backbone.efficient_b0 import Depthwise_conv,pointwise_conv
### proposed semantic segmentation head 

class LSFE(nn.module):
    '''Our Large Scale Feature Extractor(LSFE)
        Args:
            input channels,output_channels
        Returns:
            fetaure map of shape -> (128,H,W)    
    '''
    def __init__(self,in_ch=256,out_ch = 128):
        super().__init__()
        #two 3 × 3 depthwise separable convolutions,with output filters of 128 for both
        self.depthwise = Depthwise_conv(ch_in=in_ch,
                                        expansion=1,
                                        k=3,
                                        custom_stride=1)
        #use batchnorm + silu instead of  iABN Sync
        self.norm_act = nn.Sequential(
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True)
        )

        self.leaky = nn.LeakyReLU(0.1,inplace=True)
    def forward(self,x):
        x = self.depthwise(x)
        conv1 = self.leaky(self.norm_act(x)) #(b,128,h,w)
        x = self.depthwise(conv1)
        conv2 = self.leaky(self.norm_act(x))  #(b,128,h,w)
        return conv2
    
class DPC(nn.Module):
    ''' Dense Prediction Cells (DPC) to help capture long range contexts.
        DPC demonstrates a better performance with a minor increase in the number of parameters 
        Args:


    '''
    def __init__(self, ):
        super().__init__() 
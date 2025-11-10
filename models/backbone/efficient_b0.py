
"""(Basic skeleton) This module if for mbconv  class and 
efficient_b0 class from papers

"""
#import pytorch library
from torch import nn

###########################################
#Depthwise Separable convolutions
###########################################
class Depthwise_conv(nn.Module):
    '''First layer for lightweigth  filtering ,Applies a single convolutional filter per input channel.

    '''
    def __init__(self,ch_in,expansion,stride):
        super(Depthwise_conv,self).__init__()
        self.conv = nn.Conv2d(
            in_channels=ch_in * expansion,
            out_channels=ch_in*expansion,
            kernel_size=3,
            groups=ch_in * expansion,
            stride = stride
            )
        
    def forward(self,x):
        return self.conv(x)

class pointwise_conv(nn.Module):
    '''Second 1x1 convolution layer, which is responsible for building new features through computing linear combinations of the input channels. 
    
    '''
    def __init__(self,ch_in,ch_out):
        super(pointwise_conv,self).__init__()
        self.conv  = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=1,
            )
    def forward(self,x):
        return self.conv(x)   


###########################################
#mobile conv block from mobilenetv2 paper 
###########################################

#for stride 1 
class Mbconv_block:
    '''whole mbconv block for efficientNet_B0'''
    def __init__(self):
        pass

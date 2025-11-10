
"""(Basic skeleton) This module if for mbconv  class and 
efficient_b0 class from papers

"""
#import pytorch library
from torch import nn

###########################################
#Depthwise Separable convolutions
###########################################
def Depthwise_conv(ch_in,expansion,custom_stride):
    '''First layer for lightweigth  filtering ,Applies a single convolutional filter per input channel.
    Args:
        ch_in: input  channels
        expansion: expansion factor
        custom_stride: specified stride to use 
    '''
    Depthwise_conv = nn.Conv2d(
            in_channels=ch_in * expansion,
            out_channels=ch_in * expansion,
            kernel_size=3,
            groups=ch_in * expansion,
            stride = custom_stride
            )
    
    return Depthwise_conv

def pointwise_conv(ch_in,ch_out) :
    '''Second 1x1 convolution layer, which is responsible for building new features through computing linear combinations of the input channels. 
    Args:
        ch_in: input channels
        ch_out: output channels
        
    '''
    conv_1x1 =  nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=1,
            )

    return conv_1x1


###########################################
#mobile conv block from mobilenetv2 paper 
###########################################

class Mbconv_block(nn.Module):
    '''whole mbconv block for efficientNet_B0'''
    def __init__(self,custom_stride,ch_in,ch_out,expansion):
        super(Mbconv_block,self).__init__()
        self.custom_stride = custom_stride
        self.conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=custom_stride)
        self.relu = nn.ReLU()
        self.Depthwise_conv = Depthwise_conv(ch_in,expansion,custom_stride)
        self.relu =  nn.ReLU()
        self.pointwise_conv = pointwise_conv(ch_in,ch_out)
        self.linear  =  nn.Linear(ch_in,ch_out)
    def forward(self,input):
        x =  self.conv_1x1(input) 
        x =  self.Depthwise_conv(self.relu(x))
        x =  self.pointwise_conv(self.relu(x))
        return  self.linear(x) if self.custom_stride  == 1 else self.linear(x) +  input

        

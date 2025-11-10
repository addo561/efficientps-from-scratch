
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
            stride = custom_stride,
            padding=1,
            bias=False
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
            bias=False
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
        #expansion
        self.conv_1x1 = nn.Conv2d(ch_in,ch_in * expansion,kernel_size=1,stride=1,bias=False)
        self.bn =  nn.BatchNorm2d(ch_in * expansion)
        self.relu1 = nn.ReLU6(inplace=True)

        #depthwise
        self.Depthwise_conv = Depthwise_conv(ch_in,expansion,custom_stride)
        self.bn1 =  nn.BatchNorm2d(ch_in * expansion)
        self.relu2 =  nn.ReLU6(inplace=True)

        #pointwise ,projection layer
        self.pointwise_conv = pointwise_conv(ch_in * expansion,ch_out)
        self.bn2 =  nn.BatchNorm2d(ch_out)
        self.short_cut = (custom_stride==1 and ch_in==ch_out)

    def forward(self,input):
        #expantion
        x =  self.conv_1x1(input) 
        x  = self.bn(x)
        x = self.relu1(x)
        #Depthwise convolution
        x =  self.Depthwise_conv(x)
        x = self.bn1(x)
        x = self.relu2(x)
        #projection
        x =  self.pointwise_conv(x)
        x = self.bn2(x)
        return  x + input if self.short_cut else x 

block       

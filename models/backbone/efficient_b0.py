
"""(Basic skeleton) This module if for mbconv  class and 
efficient_b0 class from papers

"""
#import pytorch library
from torch import nn
import  torch.nn.functional as F
import torch

###########################################
#Depthwise Separable convolutions
###########################################
def Depthwise_conv(ch_in,expansion,custom_stride,k):
    '''First layer for lightweigth  filtering ,Applies a single convolutional filter per input channel.
    Args:
        ch_in: input  channels
        expansion: expansion factor
        custom_stride: specified stride to use 
        k: kernel_size
    '''
    Depthwise_conv = nn.Conv2d(
            in_channels=ch_in * expansion,
            out_channels=ch_in * expansion,
            kernel_size=k,
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
class Excitation(nn.Module):
    def __init__(self,ch,r):
        super(Excitation,self).__init__()
        self.block = nn.Sequential(
        nn.Linear(ch,ch//r),
        nn.ReLU(inplace=True),
        nn.Linear(ch//r,ch)
    )


class Mbconv_block(nn.Module):
    '''whole mbconv block for efficientNet_B0'''
    def __init__(self,custom_stride,ch_in,ch_out,expansion,k,r=4):
        super(Mbconv_block,self).__init__()
        self.expansion = expansion
        self.ch_in = ch_in
        self.ch_out = ch_out
        hidden_dim =  ch_in * expansion
        self.reduction =  r

        #expansion
        self.conv_1x1 = nn.Conv2d(ch_in,hidden_dim,kernel_size=1,stride=1,bias=False)
        self.bn =  nn.BatchNorm2d(hidden_dim)
        self.silu1 = nn.SiLU(inplace=True)

        #depthwise
        self.Depthwise_conv = Depthwise_conv(ch_in,expansion,custom_stride,k)
        self.bn1 =  nn.BatchNorm2d(hidden_dim)
        self.silu2 =  nn.SiLU(inplace=True)

        #Squeeze it learns to emphasize the important channels and (Excitation)suppress the less useful ones.
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        #pointwise ,projection layer
        self.pointwise_conv = pointwise_conv(hidden_dim,ch_out)
        self.bn2 =  nn.BatchNorm2d(ch_out)
        self.short_cut = (custom_stride==1 and ch_in==ch_out)

    def forward(self,input):
        #Expand
        if  self.expansion > 1:
            x = self.silu1(self.bn1(self.conv_1x1(input))) #(B,C,H,W)
        else:
            x =  input  
        #Depthwise convo    
        features = self.silu2(self.bn1(self.Depthwise_conv(x))) # (B,C,H,W)

        #squeeze
        squeezed = self.squeeze(features) #squeeze HxW  to 1X1 or use torch.mean(x,dim=[2,3])
        squeezed =  squeezed.view(squeezed.size(0),-1) #(B,C)

        #excite
        excite = Excitation(squeezed.size(1),self.reduction)
        weights = excite.block(squeezed) 
        weights = F.sigmoid(weights) #(B,C)
        weights =  weights.view(weights.size(0),weights.size(1),1,1)
        recalibrated = torch.mul(weights,features)

        #projection
        projection = self.pointwise_conv(recalibrated)
        output = self.bn2(projection)
        
        return output + input if self.short_cut  else output

###########################################
# Efficientnetb0
###########################################

class EfficientB0(nn.Module):
    """Basic EfficientnetB0 architecture
    """
    def __init__(self,custom_stride,ch_in,ch_out,expansion):
        super(EfficientB0,self).__init__()
        self.expansion = expansion
        #First 3x3 conv ,1 layer with  32 output channels,batchnorm and  swish activation
        self.conv3x3 = nn.Conv2d(
            in_channels=ch_in,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.bn1  = nn.BatchNorm2d(32)
        self.silu1 = nn.SiLU(inplace=True)

        #last  1x1 conv,batchnorm and  swish activation
        self.conv1x1 = nn.Conv2d(
            in_channels=320,
            out_channels=ch_out,
            kernel_size=1,
            stride=1
            )
        self.bn2  = nn.BatchNorm2d(ch_out)
        self.silu2 = nn.SiLU(inplace=True)

        #Global avgpool , FC layer and softmax
        self.Avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch_out,ch_out)
        self.softmax=  nn.Softmax(dim=-1)

        #mbconv blocks
        '''mbconvs = nn.ModuleList([
            Mbconv_block(custom_stride=1,ch_in=32,ch_out=16,expansion=1,k=3),
            Mbconv_block(custom_stride=2,ch_in=16,ch_out=24,expansion=6,k=3) * 2,  
            Mbconv_block(custom_stride=2,ch_in=24,ch_out=40,expansion=6,k=5) * 2,  
            Mbconv_block(custom_stride=2,ch_in=40,ch_out=80,expansion=6,k=3)*  3,  
            Mbconv_block(custom_stride=1,ch_in=80,ch_out=112,expansion=6,k=5) *  3,  
            Mbconv_block(custom_stride=2,ch_in=112,ch_out=192,expansion=6,k=5) *  4, 
            Mbconv_block(custom_stride=1,ch_in=192,ch_out=320,expansion=6,k=3), 
        ])'''

        #all subsequent layer in after 1  mbconvblock  have 1  stride
        
    def forward(self,input):
        x = self.silu1(self.bn1(self.conv3x3(x))) #(b,c,h,w)
        
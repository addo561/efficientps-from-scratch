
"""(Basic skeleton) This module if for mbconv  class and 
efficient_b0 class from papers

"""
#import pytorch library
from torch import nn
import  torch.nn.functional as F
import torch
import torchvision
from torchvision.models import EfficientNet_B0_Weights,efficientnet_b0

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
            padding=k//2,
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
            bias=False,
            stride=1
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

        #Squeeze(in forward method) it learns to emphasize the important channels and (Excitation)suppress the less useful ones.
        self.excite = Excitation(hidden_dim,r)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        #pointwise ,projection layer
        self.pointwise_conv = pointwise_conv(hidden_dim,ch_out)
        self.bn2 =  nn.BatchNorm2d(ch_out)
        self.short_cut = (custom_stride==1 and ch_in==ch_out)

    def forward(self,input):
        #Expand
        if  self.expansion > 1:
            x = self.silu1(self.bn(self.conv_1x1(input))) #(B,C,H,W)
        else:
            x =  input  
        #Depthwise convo    
        features = self.silu2(self.bn1(self.Depthwise_conv(x))) # (B,C,H,W)

        #squeeze
        avgpool = self.avgpool(features) # squeeze HxW  to 1X1 or use torch.mean(x,dim=[2,3])
        squeezed =  avgpool.view(avgpool.size(0),-1) #(B,C)
        #excite
        weights = self.excite.block(squeezed) 
        weights = F.sigmoid(weights) #(B,C)
        weights =  weights.view(weights.size(0),weights.size(1),1,1)
        recalibrated = torch.mul(weights,features)

        #projection
        projection = self.pointwise_conv(recalibrated)
        output = self.bn2(projection)
        if self.short_cut:
            result =  output + input
        else:
            result = output

        return result

###########################################
# EfficientNetb0
###########################################

class EfficientNetB0(nn.Module):
    """Basic EfficientnetB0 architecture
    """
    def __init__(self,ch_in=3,ch_out=1280):
        super(EfficientNetB0,self).__init__()

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

        #Global avgpool ,FC layer 
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lastlayers = nn.Sequential(
            nn.Linear(ch_out,1000), #lets say 1000 classes
            nn.Dropout(0.2)
        )
        
        
        #mbconv blocks
        #all subsequent layer in after 1  mbconvblock  have 1  stride
        config = [
            (1,32,16,1,3,1),
            (2,16,24,6,3,2),
            (2,24,40,6,5,2),
            (2,40,80,6,3,3),
            (1,80,112,6,5,3),
            (2,112,192,6,5,4),
            (1,192,320,6,3,1),

        ]
        layers = []
        for  s,input_channels,output_channels,e,kernel,repeat in config:
                layers.append(
                    Mbconv_block(
                        custom_stride=s,
                        ch_in=input_channels,
                        ch_out=output_channels,
                        expansion=e,
                        k=kernel)
                )
                #repeated blocks with  stride 1 
                for _ in  range(1,repeat):
                    layers.append(
                        Mbconv_block(
                            custom_stride=1,
                            ch_in=output_channels,
                            ch_out=output_channels,
                            expansion=e,
                            k=kernel,
                        )
                    )
        self.mbconv_blocks = nn.Sequential(*layers)
        
        
    def forward(self,input):
        x = self.silu1(self.bn1(self.conv3x3(input))) #(b,c,h,w)
        output_mbconv_blocks = self.mbconv_blocks(x)  #b,c,h,w
        x = self.silu2(self.bn2(self.conv1x1(output_mbconv_blocks)))
        x  = self.avgpool(x) #b,c,1,1
        x = x.view(x.size(0),-1)#b,c
        x =self.lastlayers(x)
        return x
    


'''using pretrained  Efficient net. custom one  will  require high  compute. 
    our aim is image  segmentation with  efficientps
'''


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(MultiScaleFeatureExtractor,self).__init__()
        w = EfficientNet_B0_Weights.IMAGENET1K_V1 #model weights
        self.main_model = efficientnet_b0(weights=w) #main model
        self.main_model.classifier = nn.Identity()

        for b in  self.main_model.features: 
            if isinstance(b,nn.Sequential):# all sequentials
                for mbconv in b.children():
                    if mbconv.__class__.__name__ == 'MBConv':
                        for l in mbconv.children(): # contents of mbconv
                            if isinstance(l,nn.Sequential): # sequential in mbconvs
                                for  i,layer in  enumerate(l.children()):
                                    if layer.__class__.__name__ == 'SqueezeExcitation':
                                        l[i] = nn.Identity()
    
    def forward(self,x):
        # multiscale features
        c2 = self.main_model.features[:2](x)
        c3 = self.main_model.features[:3](x)
        c5 = self.main_model.features[:4](x)
        c9 = self.main_model.features[:8](x)
        return  c2,c3,c5,c9








import torch
from torch import nn
### proposed semantic segmentation head 

#added a dilation
def Separableconvolution(ch_in,ch_out,expansion,custom_stride,k, dilate:tuple=None,padding:tuple=None):
    '''First layer for lightweigth  filtering ,Applies a single convolutional filter per input channel.
    Args:
        ch_in: input  channels
        expansion: expansion factor
        custom_stride: specified stride to use 
        k: kernel_size
    '''
    block = nn.Sequential(
            nn.Conv2d(
                in_channels=ch_in * expansion,
                out_channels=ch_in * expansion,
                kernel_size=k,
                groups=ch_in * expansion,
                stride = custom_stride,
                padding= padding if padding is not  None else k//2,
                bias=False,
                dilation= dilate if dilate is not None else 1
                ),
            nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=1,
            bias=False,
            stride=1
            )

    )  
    
    return block 


class LSFE(nn.Module):
    '''Our Large Scale Feature Extractor
    '''
    
    def __init__(self,in_ch=256,out_ch = 128):
        super().__init__()
        '''Our Large Scale Feature Extractor(LSFE)
        Args:
            input channels,output_channels
        Returns:
            fetaure map of shape -> (128,H,W)    
        '''
        # 3 × 3 depthwise separable convolutions,with output filters of 128 
        self.SeparableConv1= Separableconvolution(ch_in=in_ch,
                                                  ch_out=out_ch,
                                                  expansion=1,
                                                  k=3,
                                                  custom_stride=1)
        self.SeparableConv2= Separableconvolution(ch_in=out_ch,
                                                  ch_out=out_ch,
                                                  expansion=1,
                                                  k=3,
                                                  custom_stride=1)
        #use batchnorm + silu instead of  iABN Sync
        self.norm_act = nn.Sequential(
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                nn.LeakyReLU(0.1,inplace=True)
        )

    def forward(self,x):
        x = self.SeparableConv1(x)
        conv1 = self.norm_act(x)#(b,128,h,w)
        x = self.SeparableConv2(conv1)
        conv2 = self.norm_act(x)  #(b,128,h,w)
        return conv2
    
   
class DPC(nn.Module):
    '''DPC demonstrates a better performance with a minor increase in the number of parameters 
    '''
    def __init__(self,channels=256):
        super().__init__()
        ''' Dense Prediction Cells (DPC) to help capture long range contexts.
        Args:
            channels
        Returns:
            128,h,w feature map    

        '''
        # Seperable  convolution with dilation rates
        # dilation rates (1,6)
        self.depthwise1_6 = Separableconvolution(ch_in=channels,
                                                ch_out=channels,
                                                expansion=1,
                                                custom_stride=1,
                                                dilate = (1,6),
                                                k=3,
                                                padding=(1,6))
        
        # dilation rates (1,1)
        self.depthwise1_1 = Separableconvolution(ch_in=channels,
                                                ch_out=channels,
                                                expansion=1,
                                                custom_stride=1,
                                                dilate = (1,1),
                                                k=3,
                                                padding=(1,1))
        #dilation (6,21)
        self.depthwise6_21 = Separableconvolution(ch_in=channels,
                                                ch_out=channels,
                                                expansion=1,
                                                custom_stride=1,
                                                dilate = (6,21),
                                                k=3,
                                                padding=(6,21))
        #dilation (18,15)
        self.depthwise18_15 = Separableconvolution(ch_in=channels,
                                                ch_out=channels,
                                                expansion=1,
                                                custom_stride=1,
                                                dilate = (18,15),
                                                k=3,
                                                padding=(18,15)
                                                )
        self.depthwise6_3 = Separableconvolution(ch_in=channels,
                                                ch_out=channels,
                                                expansion=1,
                                                custom_stride=1,
                                                dilate = (6,3),
                                                k=3,
                                                padding=(6,3)
                                                )
        self.norm_act = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.LeakyReLU(0.1,inplace=True)
        )
        self.conv = nn.Conv2d(in_channels=1280,out_channels=128,kernel_size=1,stride=1)
    def forward(self,x):
        block1_6 = self.norm_act(self.depthwise1_6(x)) 
        block1_1 = self.norm_act(self.depthwise1_1(block1_6)) 
        block6_21 = self.norm_act(self.depthwise6_21(block1_6)) 
        block18_15 = self.norm_act(self.depthwise18_15(block1_6))
        block6_3= self.norm_act(self.depthwise6_3(block18_15))
        output = torch.concat([block1_6,block1_1,block6_21,block18_15,block6_3],dim=1)
        output = self.conv(output)#(b,128,H,w)
        return output  #output provides a rich feature map combining information from different scales and receptive fields.      
        

class MC(nn.Module):
    '''mitigate the mismatch between large-scale and small-scale features
    '''

    def __init__(self,in_ch = 256,out_ch=128):
        super().__init__()
        '''Mismatch Correction Module (MC) that correlates the small-scale features with respect to large-scale features.
        Args:
            input channels,output_channels
        Returns:
            fetaure map of shape -> (128,H,W) upsampled by 2 
        '''
        #3 × 3 depthwise separable convolutions,with output filters of 128 
        self.SeparableConv1 = Separableconvolution(ch_in=in_ch,
                                                  ch_out=out_ch,
                                                  expansion=1,
                                                  k=3,
                                                  custom_stride=1)
        self.SeparableConv2 = Separableconvolution(ch_in=out_ch,
                                                  ch_out=out_ch,
                                                  expansion=1,
                                                  k=3,
                                                  custom_stride=1)
        #use batchnorm + silu instead of  iABN Sync
        self.norm_act = nn.Sequential(
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                nn.LeakyReLU(0.1,inplace=True)
        )
        self.upsample =  nn.Upsample(scale_factor=2)
    def forward(self,x):
        x = self.SeparableConv1(x)
        conv1 = self.norm_act(x)#(b,128,h,w)
        x = self.SeparableConv2(conv1)
        conv2 = self.norm_act(x)  #(b,128,h,w)
        return self.upsample(conv2)
    
class SemanticHead(nn.Module):
    '''  final semantic head Containing all modules '''   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass 


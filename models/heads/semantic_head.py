import torch
from torch import nn
import  torch.nn.functional as F
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
    Attributes:
        input channels,output_channels
    '''
    
    def __init__(self,in_ch=256,out_ch = 128):
        super().__init__()
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
        '''Returns fetaure map of shape -> (128,H,W) '''
        x = self.SeparableConv1(x)
        conv1 = self.norm_act(x)#(b,128,h,w)
        x = self.SeparableConv2(conv1)
        conv2 = self.norm_act(x)  #(b,128,h,w)
        return conv2
    
   
class DPC(nn.Module):
    '''DPC demonstrates a better performance with a minor increase in the number of parameters 
        Dense Prediction Cells (DPC) to help capture long range contexts.
    Atributes:
        channels
    '''
    def __init__(self,channels=256):
        super().__init__()
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
        '''128,h,w feature map   '''
        block1_6 = self.norm_act(self.depthwise1_6(x)) 
        block1_1 = self.norm_act(self.depthwise1_1(block1_6)) 
        block6_21 = self.norm_act(self.depthwise6_21(block1_6)) 
        block18_15 = self.norm_act(self.depthwise18_15(block1_6))
        block6_3= self.norm_act(self.depthwise6_3(block18_15))
        output = torch.concat([block1_6,block1_1,block6_21,block18_15,block6_3],dim=1)
        output = self.conv(output)#(b,128,H,w)
        return output  #output provides a rich feature map combining information from different scales and receptive fields.      
        

class MC(nn.Module):
    '''mitigate the mismatch between large-scale and small-scale features,Mismatch Correction Module (MC) that correlates the small-scale features with respect to large-scale features.
    Attributes:
        input_channels,output_channels
    '''

    def __init__(self,in_ch = 128,out_ch=128):
        super().__init__()
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
        '''fetaure map of shape -> (128,H,W) upsampled by 2 '''
        x = self.SeparableConv1(x)
        conv1 = self.norm_act(x)#(b,128,h,w)
        x = self.SeparableConv2(conv1)
        conv2 = self.norm_act(x)  #(b,128,h,w)
        return self.upsample(conv2)
    
class SemanticHead(nn.Module):
    '''  final semantic head Containing all modules '''   
    def __init__(self,num_cls):
        super().__init__()
        self.dpc = DPC()
        self.lsfe  =  LSFE()
        self.mc  = MC()
        self.conv = nn.Conv2d(in_channels=512,
                              out_channels=num_cls,
                              kernel_size=1,
                              stride=1)
        self.upsample = nn.Upsample(scale_factor=4)
    def forward(self,p4,p8,p16,p32):
        lsfe4 = self.lsfe(p4)
        lsfe8 = self.lsfe(p8)
        dpc16 = self.dpc(p16)#maintain to concat(1)
        dpc32 = self.dpc(p32) #maintain to concat(2)
        ###These correlation connections aggregate contextual information from small-scale features and characteristic large-scale features for better object boundary refinement.
        #combine dpc32 + dpc16 and pass through mc module
        dpc32_upsampled = F.interpolate(dpc32, 
                                        size=dpc16.shape[2:], # Get (H, W) from dpc16
                                        mode='bilinear', 
                                        align_corners=False)#make sure h,w match
        dpc16_32 = dpc16 + dpc32_upsampled
        mc_16_32 = self.mc(dpc16_32)
        #add  to lsfe8
        mc_16_32_upsampled = F.interpolate(mc_16_32,
                                           size=lsfe8.shape[2:],
                                           mode='bilinear',
                                           align_corners=False)#make sure h,w match
        lsfe_mc_16_32_8  = mc_16_32_upsampled + lsfe8 #maintain to concat(3)
        #pass through  mc and add to lsfe4
        mc_16_32_8 = self.mc(lsfe_mc_16_32_8)
        mc_16_32_8_upsampled = F.interpolate(mc_16_32_8,
                                           size=lsfe4.shape[2:], 
                                           mode='bilinear',
                                           align_corners=False)#make sure h,w match
        lsfe_mc_16_32_8_4 = mc_16_32_8_upsampled  + lsfe4 #maintain to concat(4)
        target_size = p4.shape[2:] 
        upsampled_tensors_to_concat = [
            F.interpolate(dpc16, size=target_size, mode='bilinear', align_corners=False),
            F.interpolate(dpc32, size=target_size, mode='bilinear', align_corners=False),
            F.interpolate(lsfe_mc_16_32_8, size=target_size, mode='bilinear', align_corners=False),
            F.interpolate(lsfe_mc_16_32_8_4, size=target_size, mode='bilinear', align_corners=False)
        ]
        final = torch.concat(upsampled_tensors_to_concat,dim=1)
        final = self.conv(final)
        final = self.upsample(final) #(b,num_cls,h,w)
        return final
    





        



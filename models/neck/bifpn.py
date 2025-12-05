from torch  import nn
import torch
import torch.nn.functional as F
from models.backbone.efficient_b0 import Depthwise_conv,pointwise_conv
from models.neck.fpn import FPN

### BiFPNt -> p4,p8,p16,p32
class biFPN(FPN):
    def __init__(self):
        super(biFPN,self).__init__()
        #separable convolution
        self.depthwise = Depthwise_conv(ch_in=256,
                                        expansion=1,
                                        custom_stride=1,
                                        k=3
                                        ) #All blocks have channels 256, same for pointwise
        self.pointwise =  pointwise_conv(ch_in=256,
                                         ch_out=256
                                         )
        self.downsample2 = nn.MaxPool2d(stride=2,kernel_size=2)
        self.downsample4 = nn.MaxPool2d(stride=4,kernel_size=4)

    def forward(self,x):
        c2,c3,c4,c5 = self.extractor(x)
        block_5 = self.L_relu(self.norm_act(self.c5_conv(c5)))#(1,256,7,7)
        block_4 = self.L_relu(self.norm_act(self.c4_conv(c4)))#(1,256,28,28)
        block_3 = self.L_relu(self.norm_act(self.c3_conv(c3)))#(1,256,56,56)
        block_2 = self.L_relu(self.norm_act(self.c2_conv(c2)))#(1,256,112,112)
        p2 = block_2

        ### bottomUP approach
        #Downsample block 2 and add to  block3
        P2_down =  self.downsample2(p2)#(1,256,56,56)
        p3 = P2_down + block_3 # shape (1,256,56,56)

        #downsample P3 and add to block4
        P3_down = self.downsample2(p3) #(1,256,28,28)
        p4 = P3_down + block_4#(1,256,28,28)

        #downsample P4 and add to block5
        P4_down = self.downsample4(p4)#(1,256,7,7)
        p5 = P4_down + block_5#(1,256,7,7)
        
        #call  forward method for FPN
        fpn2,fpn3,fpn4,fpn5 = super().forward(x)

        # fpn and p blocks and  pass  through depthwise separable convolution
        # P4
        P4 = p2 + fpn2
        P4 = self.depthwise(P4)
        P4 = self.pointwise(P4)
        P4 = self.L_relu(P4)

        #P8
        P8 = p3 + fpn3
        P8 = self.depthwise(P8)
        P8 = self.pointwise(P8)
        P8 = self.L_relu(P8)

        #P16
        P16 = p4 + fpn4
        P16 = self.depthwise(P16)
        P16 = self.pointwise(P16)
        P16 = self.L_relu(P16)

        #P32
        P32 = p5 + fpn5
        P32 = self.depthwise(P32)
        P32 = self.pointwise(P32)
        P32 = self.L_relu(P32)

        return  P4,P8,P16,P32






        
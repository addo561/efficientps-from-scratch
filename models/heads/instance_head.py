##instance head
from detectron2.modeling.roi_heads import ROI_MASK_HEAD_REGISTRY, BaseMaskRCNNHead,ROI_HEADS_REGISTRY,StandardROIHeads
from inplace_abn import InPlaceABN
from torch import nn
from semantic_head import Separableconvolution

# ---------------------------------------------------------
# 1. The Custom Mask Head
# ---------------------------------------------------------
@ROI_MASK_HEAD_REGISTRY.register()
class EfficientPSMaskHead(BaseMaskRCNNHead):
    def __init__(self,cfg,input_shape):
        super().__init__(cfg,input_shape)
        in_channels = input_shape.channels
        conv_dims = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        
        # 1. Stack of Depthwise Separable Convs with ABN
        self.conv_layers = []
        current_channels = in_channels
        for _ in range(num_conv):
            self.conv_layers.append(
                Separableconvolution(
                    ch_in=current_channels,
                    ch_out=conv_dims,
                    expansion=1,
                    custom_stride=1,
                    k=3       
                )
            )
            self.conv_layers.append(InPlaceABN(conv_dims,activation='leaky_relu',activation_param=0.01))
            current_channels = conv_dims
        self.conv_layers= nn.Sequential(*self.conv_layers)

        #upsampling 14x14 -> 28x28
        self.deconv  = nn.ConvTranspose2d(
            conv_dims,conv_dims,kernel_size=2,stride=2,padding=0
        )

        self.deconv_abn = InPlaceABN(conv_dims,activation='leaky_relu',activation_param=0.01)
        #Final Predictor (1x1 Conv)
        self.predictor = nn.Conv2d(conv_dims,num_classes,kernel_size=1,stride=1,padding=0)
        # Weight Initialization (Crucial for training from scratch)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="leaky_relu")


    def forward(self,x,instances):
        # x shape: [N_proposals, Channels, 14, 14]
        x =   self.conv_layers(x)
        x = self.deconv(x)
        x = self.deconv_abn(x)
        return  self.predictor(x)
    def losses(self):
        pass 


@ROI_HEADS_REGISTRY.register()
class  EfficientPSROIHeads(StandardROIHeads):
    """
    This class orchestrates the instance branch.
    
    1. It takes features from P2-P5 (strides 4-32).
    2. It receives Proposals from RPN.
    3. It runs the Box Head (Standard 2FC) -> Outputs: Class, Box
    4. It runs the Mask Head (Our Custom Class) -> Outputs: Mask
    """
    pass



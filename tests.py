#### SIMPLE TESTING 
from models.backbone.efficient_b0 import efficientnet_b0
from models.neck.bifpn  import biFPN
import torch
import pytest
from torchinfo import  summary


def testmodel():
    """checking efficientNet
    """
    input = torch.randn(2,3,224,224) #images
    custom_model = efficientnet_b0()
    out = custom_model(input)
    assert  out.size(1)==1000

def  bifpntest():
    test_image =torch.rand(1,3,224,224)
    p4,p6,p16,p32 =  biFPN().forward(test_image)
    assert  p4.shape[2] == 112 
    


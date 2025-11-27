#### SIMPLE TESTING 
from models.backbone.efficient_b0 import efficientnet_b0
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




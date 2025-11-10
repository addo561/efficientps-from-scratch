#### SIMPLE TESTING 
from models.backbone.efficient_b0 import  Mbconv_block
import torch

def testmb_con():
    """Testing  MBconv  block. with stride 1 
    """
    input = torch.randn(2,32,64,64) #images
    block = Mbconv_block(custom_stride=1,ch_in=32,ch_out=32,expansion=6)

    out = block(input)
    print(f'output shape {out.shape}')
    print(f'input  shape {input.shape}')
    print(f'{block.short_cut}') #should be True

    print(out)
testmb_con()    
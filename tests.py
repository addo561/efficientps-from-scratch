#### SIMPLE TESTING 
from models.backbone.efficient_b0 import  Mbconv_block
import torch
import unittest


class testshort_cut(unittest.TestCase):
    def testmb_con(self):
        """Testing  MBconv  block. with stride 1  for skip  connection
            Return:
                True if shortcut 
        """
        input = torch.randn(2,32,64,64) #images
        block = Mbconv_block(custom_stride=1,ch_in=32,ch_out=32,expansion=6,k=3,r=4)

        out= block(input)
        print(f'output shape {out.shape}')
        print(f'input  shape {input.shape}')
       # self.assertEqual(block.short_cut,True) #check for skip  connection

if __name__ == '__main__':
    unittest.main()
#### SIMPLE TESTING 
from models.backbone.efficient_b0 import EfficientNetB0_pretrained,efficientnet_b0
import torch
import unittest
from torchinfo import  summary


class testshort_cut(unittest.TestCase):
    def testmodel(self):
        """checking efficientNet
        """
        input = torch.randn(2,3,224,224) #images
        #model = EfficientNetB0_pretrained()
        custom_model = efficientnet_b0()

        #summary(model,input_data=input)
        print('custom one')
        summary(custom_model,input_data=input)

if __name__ == '__main__':
    unittest.main()
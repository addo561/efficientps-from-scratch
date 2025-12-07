import pytest
import torch
from models.heads.semantic_head import LSFE,DPC,SemanticHead

p4 = torch.rand(1,256,256,512)
p8 = torch.rand(1,256,128,256)
p16 = torch.rand(1,256,64,128)
p32 = torch.rand(1,256,32,64)
test_case = [(p4,(1,128,256,512)),(p32,(1,128,32,64)),(p16,(1,128,64,128)),(p8,(1,128,128,256))]


class TestSemanticHead:
    @pytest.mark.parametrize('test_input,expected',[test_case[0],test_case[-1]])
    def test_lsfe(self,test_input,expected):
        module = LSFE()
        x =  module(test_input)
        assert x.shape== expected ,f'Test failed for input shape {x.shape} expected {expected}'    

    @pytest.mark.parametrize('test_input,expected',[test_case[1],test_case[2]])  
    def test_dpc(self,test_input,expected):
        module = DPC()
        x =  module(test_input) 
        assert x.shape== expected,f'Test failed for input shape {x.shape} expected {expected}'  
    def fullhead(self):
        module =  SemanticHead()    
        x  =  module(p4,p8,p16,p32)
        assert x.shape[2] == 1024
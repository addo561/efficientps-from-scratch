import torch
from torch import nn

#########################
#We use the weighted per-pixel log-loss
#########################
class semantic_loss(nn.Module):
    def __init__(self, num_cls):
        super().__init__()
        pass
    def forward(self,logits,targets):
        pass
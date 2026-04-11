from torch import nn
import torch
from copy import deepcopy
from .mlp import SwiGLUFFNFused
from .attention import Attention3, CrossAttention
from .attention import MemoryEfficientCrossAttention, XFORMERS_IS_AVAILBLE

def pair(t):

    return t if isinstance(t, tuple) else (t, t)

def build_perceptron(input_dim,hidden_dim, output_dim, num_layers):

    return Perceptron(input_dim, hidden_dim, output_dim, num_layers)

class Perceptron(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        
        super().__init__()
        
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers - 1):
            if i < 1:
                layers.append(nn.Linear(input_dim, hidden_dim, bias=True))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        hidden_dim = input_dim if num_layers == 1 else hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim, bias=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        
        return self.mlp(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.w_1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.w_2 = nn.Linear(mlp_dim, dim)
    
    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)

        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Layer(nn.Module):
    ATTENTION_MODES = {
        "vanilla": CrossAttention,
        "xformer": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, dim_head, mlp_dim, num_head=8, dropout=0.0, xformer=True):
        super().__init__()
        attn_mode = "xformer" if XFORMERS_IS_AVAILBLE else "vanilla"
        
        attn_cls = self.ATTENTION_MODES[attn_mode]
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = attn_cls(query_dim=dim, heads=num_head, dim_head=dim_head, dropout=dropout)
        self.xformer = xformer
        if not xformer:
            self.attn1 = Attention3(dim, num_head, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffnet = SwiGLUFFNFused(in_features=dim, hidden_features=mlp_dim)
        
    def forward(self, x, slots=None, mask = None):

        if self.xformer:
            x = self.attn1(self.norm1(x)) + x
        else:
            x = self.attn1(self.norm1(x), attn_mask=mask) + x
            
        x = self.ffnet(self.norm2(x)) + x

        return x

class Transformer(nn.Module):

    def __init__(self, layer_type, dim, depth, num_head, dim_head, mlp_dim, dropout=0., xformer=False):
        super().__init__()
        self.depth = depth
        assert layer_type in ['normal']
        layers = {'normal': Layer,}
        layer = layers[layer_type](dim, dim_head, mlp_dim, num_head, dropout, xformer)
        self.layers = _get_clones(layer, depth)
    
    def __len__(self):

        return self.depth
    def forward(self, x, slots = None, mask = None):
        
        outputs = []
        for i, layer in enumerate(self.layers):

            x = layer(x, slots, mask)
            outputs.append(x)

        return outputs



def _get_clones(module, N):

    return nn.ModuleList([deepcopy(module) for i in range(N)])

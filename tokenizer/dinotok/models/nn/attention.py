import torch, math, pdb
from torch import Tensor
from copy import deepcopy
from .mlp import build_mlp
from torch import nn, einsum
from inspect import isfunction
from typing import Optional, Any
from einops import rearrange, repeat
from torch.nn import functional as F
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d    


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q = q * self.scale
        
        sim = q @ k.transpose(-2, -1)
        sim = sim.softmax(dim=-1)

        out = sim @ v
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        proj = self.to_out(out)    

        return proj


class MaskedCrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):

        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask = None):
        
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        mask = repeat(mask, 'b f -> (b h) f', h = h)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q = q * self.scale
        
        sim = torch.einsum('n m d, n k d -> n m k', q, k)

        decimal = torch.exp(sim) * (mask.unsqueeze(1)).float()
        attn = decimal / decimal.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        out = attn @ v
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)    

class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), 
            nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask = None):

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)    



class Attention2(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float=0,
        downsample_rate = 1,
    ) -> None:

        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        
        self.scale = 1 / math.sqrt(self.internal_dim / num_heads)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:

        n, b, c = x.shape
        x = rearrange(x, 'n b (h c) -> n h b c', h = self.num_heads)
        
        return x  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:

        x = rearrange(x, 'b h k c -> b k (h c)')
        return x

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor = None) -> Tensor:
        
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        bs, _, num, c_per_head = q.shape
        
        # Attention
        logits = q @ k.transpose(2, 3)  # B x N_heads x N_tokens x N_tokens
        logits = logits * self.scale

        attn = torch.softmax(logits,dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out, attn


class Attention3(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float=0,
        downsample_rate = 1,
    ) -> None:

        super().__init__()

        self.dropout = dropout
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        
        self.scale = 1 / math.sqrt(self.internal_dim / num_heads)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:

        n, b, c = x.shape
        x = rearrange(x, 'n b (h c) -> n h b c', h = self.num_heads)
        
        return x  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:

        x = rearrange(x, 'b h k c -> b k (h c)')
        return x

    def forward(self, src, slots: Tensor = None, attn_mask: Tensor = None) -> Tensor:
        
        #* Input projections
        q = self.q_proj(src)
        k = self.k_proj(src)
        v = self.v_proj(src)

        #* Separate into heads
        q, k, v = map(lambda x: self._separate_heads(x, self.num_heads), (q, k, v))
        

        assert attn_mask is None
        if attn_mask is not None:
            attn_mask = ~attn_mask

        #* Attention
        output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            is_causal = False,
            dropout_p=self.dropout if self.training else 0, scale=self.scale)

        #* Get output
        out = self._recombine_heads(output)
        out = self.out_proj(out) 

        return out

class Attention4(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float=0,
        downsample_rate = 1,
        eps = 1e-8,
    ) -> None:

        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        self.eps = eps
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        
        self.scale = 1 / math.sqrt(self.internal_dim / num_heads)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:

        n, b, c = x.shape
        x = rearrange(x, 'n b (h c) -> n h b c', h = self.num_heads)
        
        return x  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:

        x = rearrange(x, 'b h k c -> b k (h c)')
        return x

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor = None) -> Tensor:
        
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        bs, _, num, c_per_head = q.shape
        # Attention
        logits = q @ k.transpose(2, 3)  # B x N_heads x N_tokens x N_tokens

        attn = torch.softmax(logits, dim=-2) + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        out = torch.einsum('bfnm, bfmc -> bnmfc', attn, v)
        out = rearrange(out, 'b n m f c -> b n m (f c)')

        out = self.out_proj(out) 

        return out, attn
    

class DynamicConv(nn.Module):


    def __init__(self, in_channels, hidden_dim, out_channels, num_convs=1):

        super().__init__()

        self.to_q = nn.Linear(in_channels, hidden_dim)
        self.to_k = nn.Linear(in_channels, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.params = nn.Linear(hidden_dim, num_convs * hidden_dim**2)
        self.norms = nn.ModuleList([deepcopy(self.norm1) for i in range(num_convs)])
        self.relu = nn.ReLU(inplace=True)

        self.num = num_convs
        self.hidden_dim = hidden_dim
        self.proj = build_mlp(hidden_dim, hidden_dim, out_channels, 2)
        self.norm2 = nn.LayerNorm(out_channels)
    
    def forward(self, slots, x):

        queries = self.relu(self.norm1(self.to_q(slots)))
        z = self.to_k(x)

        parameters = self.params(queries)
        params = torch.chunk(parameters, self.num, dim=-1)

        for i, (norm, param) in enumerate(zip(self.norms, params)):
            param = rearrange(param, 'b f (k c) -> b f k c', k=self.hidden_dim, c=self.hidden_dim)
            if i < 1:
                z = torch.einsum('bfkc, bmc -> bfmk', param, z)
                z = self.relu(norm(z))
            else:
                z = torch.einsum('bfkc, bfmc -> bfmk', param, z)
                z = self.relu(norm(z))

        out = self.proj(z)
        out = self.relu(self.norm2(out))

        return out
"""LSRE-CAAN (Li et al., 2023): latent-space representation extractor + cross-asset attention.

Adapted from the authors' implementation; returns h_{t,m} as [B, d, 1, M].
"""
import torch
import torch.nn.functional as F
from functools import wraps
from torch import nn, einsum
from einops import rearrange, repeat


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None,):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        # context: B, N, T, C
        # x: B, N, latent_num, latent_dim

        h = self.heads

        q = self.to_q(x)
        context = default(context, x)  # return context if exists(context) else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 's b n (h d) -> s (b h) n d', h=h), (q, k, v))

        sim = einsum('s b i d, s b j d -> s b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('s b i j, s b j d -> s b i d', attn, v)
        out = rearrange(out, 's (b h) n d -> s b n (h d)', h=h)
        return self.to_out(out)


class LSRE(nn.Module):
    r"""
    This class implements the LSRE model proposed in my paper

    For more details, please refer to the papers `Online portfolio management via deep reinforcement learning with
    high-frequency data <https://www.sciencedirect.com/science/article/abs/pii/S030645732200348X>` and `Perceiver IO: A
    General Architecture for Structured Inputs & Outputs <https://arxiv.org/abs/2107.14795>`
    """

    def __init__(
            self,
            *,
            depth,
            dim,
            num_latents,
            latent_dim,
            cross_heads,
            latent_heads,
            cross_dim_head,
            latent_dim_head,
            weight_tie_layers=True
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head,),
                    context_dim=dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])
        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

    def forward(
            self,
            data,
            mask=None
    ):
        s, b, _, _ = data.shape

        # latents
        x = repeat(self.latents, 'n d -> s b n d', b=b, s=s)
        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(x, context=data, mask=mask) + x
        x = cross_ff(x) + x

        count = 0
        # layers
        for self_attn, self_ff in self.layers:
            count += 1
            x = self_attn(x) + x
            x = self_ff(x) + x  # x.shape = torch.Size([num_assets, num_latents, latent_dim]), where num_latents = 1

        assert x.shape[2] == 1  # I set num_latents = 1 in my paper
        return x.squeeze(2)  # [num_assets, 1, latent_dim] -> [num_assets, latent_dim]



class LSRECAANExtractor(nn.Module):
    def __init__(self, in_channels=4, d=32, num_layers=1, window=50):
        super().__init__()
        self.pos_emb = nn.Embedding(window, in_channels)
        self.lsre = LSRE(
            depth=num_layers,
            dim=in_channels,
            num_latents=1,
            latent_dim=d,
            cross_heads=1,
            latent_heads=1,
            cross_dim_head=d * 2,
            latent_dim_head=d,
        )
        self.linear_query = torch.nn.Linear(d, d)
        self.linear_key = torch.nn.Linear(d, d)
        self.linear_value = torch.nn.Linear(d, d)

    def forward(self, patterns, asset_mask=None):  # [B, C, l, M]
        x = patterns.permute(0, 3, 2, 1).contiguous()  # [B, M, l, C]
        x = x + self.pos_emb(torch.arange(x.shape[2]).to(x.device))[None, None]
        stock_rep = self.lsre(x, mask=None)  # [B, M, d]
        # CAAN
        query = self.linear_query(stock_rep)
        key = self.linear_key(stock_rep)
        value = self.linear_value(stock_rep)
        beta = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(float(query.shape[-1])))
        beta = F.softmax(beta, dim=-1).unsqueeze(-1)
        stock_rep = torch.sum(value.unsqueeze(1) * beta, dim=2)  # [B, M, d]
        return stock_rep.permute(0, 2, 1).unsqueeze(2)

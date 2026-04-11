from dataclasses import dataclass, field
from mimetypes import encodings_map
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import MISSING, OmegaConf
import torch.nn.functional as F
# from memory_profiler import profile
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tokenizer.dinotok.models.nn.transformer import Transformer
from einops import rearrange, repeat
from tokenizer.dinotok.models.nn.act import build_act
from tokenizer.dinotok.models.nn.norm import build_norm
from tokenizer.dinotok.models.nn.ops import (
    ChannelDuplicatingPixelUnshuffleUpSampleLayer,
    ConvLayer,
    ConvPixelShuffleUpSampleLayer,
    EfficientViTBlock,
    IdentityLayer,
    InterpolateConvUpSampleLayer,
    OpSequential,
    ResBlock,
    ResidualBlock,
)
# import matplotlib.pyplot as plt
from einops import rearrange

__all__ = ["DINOTOK", "dino_f14c768","dinov3_f16c768"]


@dataclass
class DecoderConfig:
    in_channels: int = MISSING
    latent_channels: int = MISSING
    in_shortcut: Optional[str] = "duplicating"
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    norm: Any = "trms2d"
    act: Any = "silu"
    upsample_block_type: str = "ConvPixelShuffle"
    upsample_match_channel: bool = True
    upsample_shortcut: str = "duplicating"
    out_norm: str = "trms2d"
    out_act: str = "relu"
    image_width: int = 256
    image_height: int = 256


@dataclass
class DINOTOKConfig:
    in_channels: int = 3
    codebook_groups_rec: int = 2
    latent_channels: int = 768
    latent_channels_last: int = 768
    latent_channels_first: int = 768
    feature_channels: int = 768
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderConfig(in_channels="${..in_channels}", latent_channels="${..latent_channels}")
    )
    codebook_size: int = 0
    enable_vq: bool = False

    pretrained_path: Optional[str] = None
    pretrained_source: str = "dinotok"
    dinov3_status: Optional[str] = None
    scaling_factor: Optional[float] = None

def build_block(
    block_type: str, in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str]
) -> nn.Module:
    if block_type == "ResBlock":
        assert in_channels == out_channels
        main_block = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
        )
        block = ResidualBlock(main_block, IdentityLayer())
    elif block_type == "EViT_GLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=())
    elif block_type == "EViTS5_GLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=(5,))
    else:
        raise ValueError(f"block_type {block_type} is not supported")
    return block


def build_stage_main(
    width: int, depth: int, block_type: str | list[str], norm: str, act: str, input_width: int
) -> list[nn.Module]:
    assert isinstance(block_type, str) or (isinstance(block_type, list) and depth == len(block_type))
    stage = []
    for d in range(depth):
        current_block_type = block_type[d] if isinstance(block_type, list) else block_type
        block = build_block(
            block_type=current_block_type,
            in_channels=width if d > 0 else input_width,
            out_channels=width,
            norm=norm,
            act=act,
        )
        stage.append(block)
    return stage


def build_upsample_block(block_type: str, in_channels: int, out_channels: int, factor: Optional[int], shortcut: Optional[str], size: Optional[tuple]) -> nn.Module:
    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    elif block_type == "InterpolateConv":
        if factor == None:
            block = InterpolateConvUpSampleLayer(
                in_channels=in_channels, out_channels=out_channels, kernel_size=3, size=size, factor=None
            )
        else:
            block = InterpolateConvUpSampleLayer(
                in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=factor, size=None
            )
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


def build_decoder_project_in_block(in_channels: int, out_channels: int, shortcut: Optional[str]):
    block = ConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        use_bias=True,
        norm=None,
        act_func=None,
    )
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for decoder project in")
    return block


def build_decoder_project_out_block(
    in_channels: int, out_channels: int, upsample_block_type: str, factor: Optional[int], norm: Optional[str], act: Optional[str], size: Optional[tuple]
):
    layers: list[nn.Module] = [
        build_norm(norm, in_channels),
        build_act(act),
    ]
    if factor == None:
        layers.append(
            build_upsample_block(
                block_type=upsample_block_type, in_channels=in_channels, out_channels=out_channels, size=size, factor=None, shortcut=None
            )
        )
    elif factor == 1:
        layers.append(
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
            )
        )
    else:
        layers.append(
            build_upsample_block(
                block_type=upsample_block_type, in_channels=in_channels, out_channels=out_channels, factor=factor, shortcut=None, size=None
            )
        )
    return OpSequential(layers)


class Sem_ViT(nn.Module):
    def __init__(self, layer_type, image_height, image_width, patch_size, dim, n_carrier, depth,
                 num_head, mlp_dim, out_channels, dim_head=64, dropout=0., num_register_tokens=4,):
        
        super().__init__()
        
        self.patch_size = patch_size
        assert image_height % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_width % patch_size == 0

        self.dim = dim
        scale = dim ** -0.5
        self.width = image_width // patch_size
        self.height = image_height // patch_size
        self.num_patches = self.width * self.height

        self.num_tokens = 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim) * scale)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * scale)


        self.transformer = Transformer(layer_type, dim, 3, num_head, dim_head, mlp_dim, dropout, xformer=False)
        
        self.norm_post1 = nn.LayerNorm(dim)
        self.vit_proj = nn.Linear(dim, out_channels, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def interpolate_pos_encoding(self, x, w, h):

        npatch = w * h
        N = self.position_embedding.size(1) - 1
        if npatch == N and w == h:
            return self.position_embedding

        class_pos_embed = self.position_embedding[:, :1]
        patch_pos_embed = self.position_embedding[:, 1:]
        dim = x.shape[2]
        
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w + 0.1, h + 0.1

        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, self.width, self.height, dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / self.width, h0 / self.height),
            mode='bicubic',
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        pos_embed = torch.cat((class_pos_embed, patch_pos_embed.flatten(2).transpose(1, 2)), dim=1)
        
        return pos_embed

    
    def forward(self, slots):

        ## b n c
        bs = slots.size(0)

        z_pos = self.interpolate_pos_encoding(slots, self.width, self.height)

        cls_token = repeat(self.cls_token, 'f ... -> (b f) ...',b=bs)

        cc = torch.cat((cls_token + z_pos[:, :1], slots + z_pos[:, 1:]), dim=1)

        outputs = self.transformer(self.norm_post1(cc))

        rep = self.vit_proj(outputs[-1][:,self.num_tokens:])
        return rep
    
      

class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig, enable_vq):
        super().__init__()
        self.cfg = cfg
        self.enable_vq = enable_vq
        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, str) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )
        assert isinstance(cfg.norm, str) or (isinstance(cfg.norm, list) and len(cfg.norm) == num_stages)
        assert isinstance(cfg.act, str) or (isinstance(cfg.act, list) and len(cfg.act) == num_stages)

        self.project_in = build_decoder_project_in_block(
            in_channels=cfg.latent_channels, 
            out_channels=cfg.width_list[-1],
            shortcut=None,
        )
        if self.enable_vq:
            self.project_sem = build_decoder_project_in_block(
                in_channels=cfg.latent_channels, 
                out_channels=512,
                shortcut=None,
            )

            self.sem_vit = Sem_ViT(
                layer_type='normal',
                image_height=cfg.image_height,image_width=cfg.image_width,patch_size=16,
                dim=512, n_carrier=256, depth=6, num_head=8, mlp_dim=2048,
                out_channels=768,dim_head=64, dropout=0.1, num_register_tokens=4,
            )


        self.stages: list[OpSequential] = []
        
        for stage_id, (width, depth) in reversed(list(enumerate(zip(cfg.width_list, cfg.depth_list)))):
            stage = []
            if stage_id < num_stages - 1 and depth > 0:
                upsample_block = build_upsample_block(
                    block_type="InterpolateConv",
                    in_channels=cfg.width_list[stage_id + 1],
                    out_channels=width if cfg.upsample_match_channel else cfg.width_list[stage_id + 1],
                    shortcut=None,
                    factor=2,
                    size=None
                )
                stage.append(upsample_block)
            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            norm = cfg.norm[stage_id] if isinstance(cfg.norm, list) else cfg.norm
            act = cfg.act[stage_id] if isinstance(cfg.act, list) else cfg.act
            stage.extend(
                build_stage_main(
                    width=width,
                    depth=depth,
                    block_type=block_type,
                    norm=norm,
                    act=act,
                    input_width=(
                        width if cfg.upsample_match_channel else cfg.width_list[min(stage_id + 1, num_stages - 1)]
                    ),                
                )
            )
            self.stages.insert(0, OpSequential(stage))
        self.stages = nn.ModuleList(self.stages)

        self.project_out = build_decoder_project_out_block(
            in_channels=cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
            out_channels=cfg.in_channels,
            factor=None,
            size=(cfg.image_height,cfg.image_width),
            upsample_block_type="InterpolateConv",
            norm=cfg.out_norm,
            act=cfg.out_act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sem_dino = None

        if self.enable_vq:
            sem_in = self.project_sem(x)
            sem_in = rearrange(sem_in, 'b c h w -> b (h w) c')
            sem_dino = self.sem_vit(sem_in)
            sem_dino = rearrange(sem_dino, 'b (h w) c -> b c h w', h=16, w=16).contiguous()

        x = self.project_in(x)
        for stage in reversed(self.stages):
            if len(stage.op_list) == 0:
                continue
            x = stage(x)

        x = self.project_out(x)
        return x, sem_dino
    

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim_first, e_dim_last, beta, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.beta = beta
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.e_dim_first = e_dim_first
        self.e_dim_last = e_dim_last
        self.e_dim = e_dim_first + e_dim_last
        self.embedding_first = nn.Embedding(self.n_e, self.e_dim_first)
        self.embedding_last = nn.Embedding(self.n_e, self.e_dim_last)

        self.embedding_first.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.embedding_last.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        if self.l2_norm:
            self.embedding_first.weight.data = F.normalize(self.embedding_first.weight.data, p=2, dim=-1)
            self.embedding_last.weight.data = F.normalize(self.embedding_last.weight.data, p=2, dim=-1)
        
        if self.show_usage:
            self.register_buffer("codebook_used_first", nn.Parameter(torch.zeros(65536)))
            self.register_buffer("codebook_used_last", nn.Parameter(torch.zeros(65536)))
        self.anchor = "probrandom"
        self.decay = 0.99
        self.register_buffer("embed_prob_sem", torch.zeros(self.n_e))
        self.register_buffer("embed_prob_rec", torch.zeros(self.n_e))
        # cvq resample
        self.init = False
        self.first_batch = False
        self.enable_resample = True
        

    def forward(self, z):

        z = torch.einsum('b c h w -> b h w c', z).contiguous()

        if self.l2_norm:

            z_normalized_first = F.normalize(z[..., self.e_dim_last:], p=2, dim=-1)
            z_normalized_last = F.normalize(z[..., :self.e_dim_last], p=2, dim=-1)
            z = torch.cat([z_normalized_last,z_normalized_first], dim=-1)

            embedding_first = F.normalize(self.embedding_first.weight, p=2, dim=-1)
            embedding_last = F.normalize(self.embedding_last.weight, p=2, dim=-1)

            z_normalized_first_flattened = z_normalized_first.view(-1, self.e_dim_first)
            z_normalized_last_flattened = z_normalized_last.view(-1, self.e_dim_last)
        else:
            embedding_first = self.embedding_first.weight
            embedding_last = self.embedding_last.weight

        d_first = torch.sum(z_normalized_first_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_first**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_normalized_first_flattened, torch.einsum('n d -> d n', embedding_first))
        
        d_last = torch.sum(z_normalized_last_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_last**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_normalized_last_flattened, torch.einsum('n d -> d n', embedding_last))
        # 
        d_rec = -d_first
        d_sem = -d_last
        sort_distance_sem, indices_sem = d_sem.sort(dim=1)
        sort_distance_rec, indices_rec = d_rec.sort(dim=1)
        
        encoding_indices_sem = indices_sem[:,-1]
        encodings_mat_sem = torch.zeros(encoding_indices_sem.unsqueeze(1).shape[0], self.n_e, device=z_normalized_last.device)
        encodings_mat_sem.scatter_(1, encoding_indices_sem.unsqueeze(1), 1)

        encoding_indices_rec = indices_rec[:,-1]
        encodings_mat_rec = torch.zeros(encoding_indices_rec.unsqueeze(1).shape[0], self.n_e, device=z_normalized_first.device)
        encodings_mat_rec.scatter_(1, encoding_indices_rec.unsqueeze(1), 1)

        min_encoding_indices_first = torch.argmin(d_first, dim=1)
        min_encoding_indices_last = torch.argmin(d_last, dim=1)

        z_q_first = embedding_first[min_encoding_indices_first].view(z_normalized_first.shape)
        z_q_last = embedding_last[min_encoding_indices_last].view(z_normalized_last.shape)

        perplexity = None
        min_encodings = None
        entropy_loss = None
        codebook_usage_first = 0
        codebook_usage_last = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices_first.shape[0]
            self.codebook_used_first[:-cur_len] = self.codebook_used_first[cur_len:].clone()
            self.codebook_used_first[-cur_len:] = min_encoding_indices_first
            codebook_usage_first = len(torch.unique(self.codebook_used_first)) / self.n_e

            cur_len = min_encoding_indices_last.shape[0]
            self.codebook_used_last[:-cur_len] = self.codebook_used_last[cur_len:].clone()
            self.codebook_used_last[-cur_len:] = min_encoding_indices_last
            codebook_usage_last = len(torch.unique(self.codebook_used_last)) / self.n_e


        vq_loss_first = None
        commit_loss_first = None
        vq_loss_last = None
        entropy_loss = torch.tensor(0)

        if self.training:
            vq_loss_first = torch.mean((z_q_first - z_normalized_first.detach()) ** 2) 
            commit_loss_first = self.beta * torch.mean((z_q_first.detach() - z_normalized_first) ** 2) 
            vq_loss_last = torch.mean((z_q_last - z_normalized_last.detach()) ** 2)

            entropy_loss =  torch.tensor(0)

        z_q = torch.cat([z_q_last,z_q_first], dim=-1)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q).contiguous()

        avg_probs_sem = torch.mean(encodings_mat_sem, dim=0) # prob of used codes 
        perplexity_sem = torch.exp(-torch.sum(avg_probs_sem * torch.log(avg_probs_sem + 1e-10))) # e^-H(p) entropy of distribution p
        avg_probs_rec = torch.mean(encodings_mat_rec, dim=0) # prob of used codes 
        perplexity_rec = torch.exp(-torch.sum(avg_probs_rec * torch.log(avg_probs_rec + 1e-10))) # e^-H(p) entropy of distribution p


        # z_q_after_compressed = torch.einsum('b h w c -> b c h w', z_q_after_compressed)
        ''' deadpoint reactivation'''
        # online clustered reinitialisation for unoptimized points
        if self.training and self.enable_resample:
            # calculate the average usage of code entries
            self.embed_prob_sem.mul_(self.decay).add_(avg_probs_sem, alpha= 1 - self.decay) # v * 0.99 + avg * 0.01
            self.embed_prob_rec.mul_(self.decay).add_(avg_probs_rec, alpha= 1 - self.decay) # v * 0.99 + avg * 0.01
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # probabilitical based random sampling
                if self.anchor == 'probrandom':
                    norm_distance_sem = F.softmax(d_sem.t(), dim=1)
                    if not torch.isfinite(norm_distance_sem).all():
                        print("Warning: probs contain NaN or Inf, replacing with uniform distribution")
                        print(f"d: {d_sem}, dmax:{d_sem.max()}, dmin: {d_sem.min()}")
                        norm_distance_sem = torch.ones_like(norm_distance_sem) / norm_distance_sem.numel()

                    norm_distance_sem = torch.clamp(norm_distance_sem, min=1e-10)
                    norm_distance_sem /= torch.sum(norm_distance_sem)  

                    prob_sem = torch.multinomial(norm_distance_sem, num_samples=1).view(-1)
                    random_feat_sem = z_normalized_last_flattened.detach()[prob_sem]
                    
                    norm_distance_rec = F.softmax(d_rec.t(), dim=1)

                    if not torch.isfinite(norm_distance_rec).all():
                        print("Warning: probs contain NaN or Inf, replacing with uniform distribution")
                        print(f"d: {d_rec}, dmax:{d_rec.max()}, dmin: {d_rec.min()}")
                        norm_distance_rec = torch.ones_like(norm_distance_rec) / norm_distance_rec.numel()

                    norm_distance_rec = torch.clamp(norm_distance_rec, min=1e-10)
                    norm_distance_rec /= torch.sum(norm_distance_rec)  

                    prob_rec = torch.multinomial(norm_distance_rec, num_samples=1).view(-1)
                    random_feat_rec = z_normalized_first_flattened.detach()[prob_rec]
                # decay parameter based on the average usage
                decay_sem = torch.exp(-(self.embed_prob_sem*self.n_e*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.e_dim_last)
                self.embedding_last.weight.data = self.embedding_last.weight.data * (1 - decay_sem) + random_feat_sem * decay_sem
                
                # decay parameter based on the average usage
                decay_rec = torch.exp(-(self.embed_prob_rec*self.n_e*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.e_dim_first)
                self.embedding_first.weight.data = self.embedding_first.weight.data * (1 - decay_rec) + random_feat_rec * decay_rec
                
           

        return z_q, ((vq_loss_last, vq_loss_first), (None, commit_loss_first), entropy_loss, (codebook_usage_last,codebook_usage_first)), (perplexity, min_encodings, (min_encoding_indices_first,min_encoding_indices_last))

    
class DINOTOK(nn.Module):
    def __init__(self, cfg: DINOTOKConfig):
        super().__init__()
        self.cfg = cfg
        self.codebook_size = cfg.codebook_size
        self.decoder = Decoder(cfg.decoder,cfg.enable_vq)
        self.quant_conv = nn.Sequential(
            nn.Conv2d(cfg.feature_channels, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(256, cfg.latent_channels_first, kernel_size=1),
            nn.GroupNorm(1, cfg.latent_channels_first),  # or nn.BatchNorm2d
        )

        if cfg.enable_vq:
            self.e_dim = 768
            self.codebook_anchor = torch.tensor(np.load(cfg.dinov3_status))
            self.pcs = self.pca_analyze_vocab(self.codebook_anchor.detach().cpu().numpy(), n=self.e_dim) # get all 768
            self.pc_sortids = self.pcs['sorted_idx_per_pc'][0]

            self.quantize = VectorQuantizer(cfg.codebook_size, cfg.latent_channels_first, cfg.latent_channels_last, 0.25, True, True)


    def pca_analyze_vocab(self, V: np.ndarray, n=32, standardize=True):
        # Optional standardization
        if standardize:
            scaler = StandardScaler(with_mean=True, with_std=True)
            X = scaler.fit_transform(V)
        else:
            # still center to get a proper covariance PCA
            X = V - V.mean(axis=0, keepdims=True)

        # Fit full PCA 
        pca_full = PCA(n_components=None, svd_solver="full", random_state=0).fit(X)
        comps_full   = pca_full.components_                  
        comps_n   = comps_full[:n, :]        
        sorted_idx_per_pc = [np.argsort(np.abs(comps_n[k]))[::-1] for k in range(n)]

        return {
            "sorted_idx_per_pc": sorted_idx_per_pc,       
        }
    

    def pca_reduce_dim_top(self, dinov2_latent, output_dim):

        pc_sortids_top = self.pc_sortids[:output_dim].copy()
        pc_sortids_top_tensor = torch.tensor(pc_sortids_top, dtype=torch.int32, device=dinov2_latent.device)
        dinov2_latent_reduced = dinov2_latent[:, pc_sortids_top_tensor,...].contiguous()

        return dinov2_latent_reduced


    def encode(self, h):
        h_first = h[0]
        h_last = h[-1]
        # b c h w
        h_first = self.quant_conv(h_first)

        h = torch.cat((h_last,h_first), dim=1)

        emb_loss = None
        info = None
        if self.cfg.enable_vq:
            h_last = self.pca_reduce_dim_top(h_last, output_dim=32)
            h = torch.cat((h_last,h_first), dim=1)
            h, emb_loss, info = self.quantize(h)

        return h, emb_loss, info, h_first



    def decode(self, x: torch.Tensor) -> torch.Tensor:
        
        x,x_dino = self.decoder(x) 
        return x,x_dino
    
    

    def forward(self, x, global_step: int):
        x, diff, q_indices, _ = self.encode(x)
        dec,x_dino = self.decode(x)
        return dec, diff, x, x_dino, q_indices


def dinov3_f16c768(name: str, pretrained_path: str, codebook_size: int, image_width: int, image_height: int, enable_vq: bool, dinov3_status: str = None) -> DINOTOKConfig:
    if name in ["dinov3-f16c768-in-1.0"]:
        if enable_vq:
            cfg_str = (
                "latent_channels=96 "
                "latent_channels_first=64 "
                "latent_channels_last=32 "
                "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
                "decoder.width_list=[128,256,512,512,1024,1024] decoder.depth_list=[0,5,10,3,2,2] "
                "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d] decoder.act=[relu,relu,relu,silu,silu,silu] "
                f"dinov3_status={dinov3_status} "
                f"codebook_size={codebook_size} "
                f"enable_vq={enable_vq} "
                f"decoder.image_height={image_height} "
                f"decoder.image_width={image_width}"
            )
        else:
            cfg_str = (
                "latent_channels=832 "
                "latent_channels_first=64 "
                "latent_channels_last=768 "
                "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
                "decoder.width_list=[128,256,512,512,1024,1024] decoder.depth_list=[0,5,10,2,2,2] "
                "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d] decoder.act=[relu,relu,relu,silu,silu,silu] "
                f"codebook_size={codebook_size} "
                f"enable_vq={enable_vq} "
                f"decoder.image_height={image_height} "
                f"decoder.image_width={image_width}"
            )
    else:
        raise NotImplementedError
    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: DINOTOKConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DINOTOKConfig), cfg))
    cfg.pretrained_path = pretrained_path
    return cfg



def dino_f14c768(name: str, pretrained_path: str, codebook_size: int, image_width: int, image_height: int, enable_vq: bool, dinov3_status: str = None) -> DINOTOKConfig:
    if name in ["dino-f14c768-in-1.0"]:
        if enable_vq:
            cfg_str = (
                "latent_channels=96 "
                "latent_channels_first=64 "
                "latent_channels_last=32 "
                "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
                "decoder.width_list=[128,256,512,512,1024,1024] decoder.depth_list=[0,5,10,3,2,2] "
                "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d] decoder.act=[relu,relu,relu,silu,silu,silu] "
                f"dinov3_status={dinov3_status} "
                f"codebook_size={codebook_size} "
                f"enable_vq={enable_vq} "
                f"decoder.image_height={image_height} "
                f"decoder.image_width={image_width}"
            )
        else:
            cfg_str = (
                "latent_channels=832 "
                "latent_channels_first=64 "
                "latent_channels_last=768 "
                "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
                "decoder.width_list=[128,256,512,512,1024,1024] decoder.depth_list=[0,5,10,2,2,2] "
                "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d] decoder.act=[relu,relu,relu,silu,silu,silu] "
                f"codebook_size={codebook_size} "
                f"enable_vq={enable_vq} "
                f"decoder.image_height={image_height} "
                f"decoder.image_width={image_width}"
            )
    else:
        raise NotImplementedError
    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: DINOTOKConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DINOTOKConfig), cfg))
    cfg.pretrained_path = pretrained_path
    return cfg

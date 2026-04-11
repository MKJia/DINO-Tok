from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


from tokenizer.dinotok.ViT_DINOv3 import vit_base
from tokenizer.dinotok.model_zoo import DINOTOK_HF


class DINOV3AEWrapper(nn.Module):
    """
    Adapter between the external DINOV3 AE and the Stage-1 interface expected by Stage 2 DiT training.
    """

    def __init__(
        self,
        encoder_ckpt: str,
        decoder_ckpt: str,
        img_size: int = 256,
        ds_rate: int = 16,
        enable_vq: bool = False,
        codebook_size: int = 16384
    ):
        super().__init__()
        self.img_size = img_size
        self.ds_rate = ds_rate
        self.patch_size = ds_rate

        self.encoder = vit_base(checkpoint=encoder_ckpt)
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        ae_args = SimpleNamespace(
            vq_model="dinov3-f16c768-in-1.0",
            codebook_size=codebook_size,
            image_width=img_size,
            image_height=img_size,
            enable_vq=enable_vq,
            dinov3_status=None,
        )
        self.decoder = DINOTOK_HF(ae_args)
        decoder_state = torch.load(decoder_ckpt, map_location="cpu", weights_only=False)
        self.decoder.load_state_dict(decoder_state["model"], strict=True)
        self.decoder.eval()
        self.decoder.requires_grad_(False)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean, persistent=False)
        self.register_buffer("img_std", std, persistent=False)
 
    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bicubic", align_corners=False)
        x = torch.clamp(x, 0.0, 1.0)
        x = (x - self.img_mean) / self.img_std
        return x

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        encoder = self.encoder.module if hasattr(self.encoder, "module") else self.encoder
        feats = encoder.get_intermediate_layers(x, reshape=False, norm=True)
        h = self.img_size // self.ds_rate
        w = h
        feats = tuple(rearrange(feat, "b (hw) c -> b c h w", h=h, w=w) for feat in feats)
        latents = self.decoder.encode(feats)[0]
        return latents

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        images, _ = self.decoder.decode(z)
        images = images * self.img_std + self.img_mean
        images = torch.clamp(images, 0.0, 1.0)
        return images

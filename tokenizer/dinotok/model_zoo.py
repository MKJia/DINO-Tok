from typing import Callable, Optional

from huggingface_hub import PyTorchModelHubMixin

from tokenizer.dinotok.models.tokenizer.dinotok import DINOTOK, DINOTOKConfig, dino_f14c768, dinov3_f16c768

__all__ = ["create_dinotok_model_cfg", "DINOTOK_HF"]


REGISTERED_DINOTOK_MODEL: dict[str, tuple[Callable, Optional[str]]] = {
    "dino-f14c768-in-1.0": (dino_f14c768, None),
    "dinov3-f16c768-in-1.0": (dinov3_f16c768, None),
}


def create_dinotok_model_cfg(name: str, codebook_size: int, image_height: int, image_width: int, enable_vq: bool, dinov3_status: str, pretrained_path: Optional[str] = None) -> DINOTOKConfig:
    assert name in REGISTERED_DINOTOK_MODEL, f"{name} is not supported"
    dinotok_cls, default_pt_path = REGISTERED_DINOTOK_MODEL[name]
    pretrained_path = default_pt_path if pretrained_path is None else pretrained_path
    model_cfg = dinotok_cls(name, pretrained_path, codebook_size, image_height, image_width, enable_vq, dinov3_status)
    return model_cfg


class DINOTOK_HF(DINOTOK, PyTorchModelHubMixin):
    def __init__(self, args):
        model_name = args.vq_model
        codebook_size = args.codebook_size
        image_width = args.image_width
        image_height = args.image_height
        enable_vq = args.enable_vq
        dinov3_status = getattr(args, "dinov3_status", None)
        cfg = create_dinotok_model_cfg(model_name, codebook_size, image_width, image_height, enable_vq, dinov3_status)
        DINOTOK.__init__(self, cfg)


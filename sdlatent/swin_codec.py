from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def psnr01(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.clip(a, 0.0, 1.0) - np.clip(b, 0.0, 1.0)) ** 2))
    if mse <= 1e-12:
        return 120.0
    return 10.0 * np.log10(1.0 / mse)


def build_swin(args):
    import torch
    import torch.nn as nn

    sys.path.insert(0, str(Path(args.swin_dir).resolve()))
    from net.network import SwinJSCC

    class DummyArgs:
        trainset = "DIV2K"
        testset = "kodak"
        distortion_metric = "MSE"
        model = args.model
        channel_type = "awgn"
        C = str(args.c)
        multiple_snr = str(args.encoder_snr)
        model_size = args.model_size

    class Config:
        seed = 42
        pass_channel = False
        CUDA = True
        device = torch.device(args.device)
        norm = False
        logger = None
        downsample = 4
        image_dims = (3, 256, 256)
        channel_number = int(args.c)
        encoder_kwargs = dict(
            model=args.model,
            img_size=(256, 256),
            patch_size=2,
            in_chans=3,
            embed_dims=[128, 192, 256, 320],
            depths=[2, 2, 6, 2],
            num_heads=[4, 6, 8, 10],
            C=channel_number,
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        decoder_kwargs = dict(
            model=args.model,
            img_size=(256, 256),
            embed_dims=[320, 256, 192, 128],
            depths=[2, 6, 2, 2],
            num_heads=[10, 8, 6, 4],
            C=channel_number,
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )

    device = torch.device(args.device)
    net = SwinJSCC(DummyArgs, Config).to(device)
    weights = torch.load(args.model_path, map_location=device)
    net.load_state_dict(weights, strict=True)
    net.eval()
    return net, device


def load_image_tensor(path: Path, device):
    from PIL import Image
    from torchvision import transforms

    image = Image.open(path).convert("RGB")
    width, height = image.size
    crop_w = width - width % 128
    crop_h = height - height % 128
    image = transforms.CenterCrop((crop_h, crop_w))(image)
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return tensor, (crop_h, crop_w)


def decode_latent(net, vec: np.ndarray, shape: tuple[int, int, int], snr_db: float, device, model: str) -> np.ndarray:
    import torch

    h, w, c = shape
    feature = torch.from_numpy(vec.reshape(1, h * w, c)).float().to(device)
    net.decoder.update_resolution(h, w)
    with torch.inference_mode():
        recon = net.decoder(feature, snr_db, model).clamp(0.0, 1.0)
    return recon.detach().cpu().numpy()[0]

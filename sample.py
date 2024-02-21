# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""

import argparse

from models import DiT_models
from diffusion import create_diffusion
from visualization import save_image
from experimental.models import AutoencoderKL

import mindspore as ms
from mindspore import ops


def main(args):
    # Setup MindSpore:
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_seed(args.seed)

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    state_dict = ms.load_checkpoint(args.ckpt)
    param_not_load, ckpt_not_load = ms.load_param_into_net(model, state_dict)
    if param_not_load:
        print(f"{param_not_load} in network is not loaded!")
    if ckpt_not_load:
        print(f"{ckpt_not_load} in checkpoint is not loaded!")
    model.set_train(False)  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    z = ops.randn(n, 4, latent_size, latent_size)
    y = ms.tensor(class_labels)

    # Setup classifier-free guidance:
    z = ops.cat([z, z], 0)
    y_null = ms.tensor([1000] * n)
    y = ops.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
    )
    samples, _ = samples.chunk(2, axis=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215)[0]

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a DiT checkpoint.")
    args = parser.parse_args()
    main(args)

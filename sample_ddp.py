# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import os
import math
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from models import DiT_models
from diffusion import create_diffusion
from experimental.models import AutoencoderKL

import mindspore as ms
from mindspore import ops
from mindspore.communication import get_group_size, get_local_rank, get_rank, init


def random_seed(seed=42, rank=0):
    ms.set_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def is_master(args):
    return args.rank == 0


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if args.distributed:
        init()
        args.local_rank = get_local_rank()
        args.world_size = get_group_size()
        args.rank = get_rank()
        ms.context.set_auto_parallel_context(
            device_num=args.world_size,
            global_rank=args.rank,
            parallel_mode="data_parallel",
            gradients_mean=True,
        )

    device = f"{ms.get_context('device_target')}:{ms.get_context('device_id')}"
    args.device = device
    return device


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    # Setup DDP:
    ms.set_context(mode=ms.GRAPH_MODE)
    device = init_distributed_device(args)
    random_seed(args.global_seed)
    print(f"Starting rank={args.rank}, seed={args.global_seed}, world_size={args.world_size}.")

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
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".ckpt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if args.rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * args.world_size
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if args.rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % args.world_size == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // args.world_size)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if args.rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = ops.randn(n, model.in_channels, latent_size, latent_size)
        y = ops.randint(0, args.num_classes, (n,))

        # Setup classifier-free guidance:
        if using_cfg:
            z = ops.cat([z, z], 0)
            y_null = ms.tensor([1000] * n)
            y = ops.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        else:
            model_kwargs = dict(y=y)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False
        )
        if using_cfg:
            samples, _ = samples.chunk(2, axis=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215)[0]
        samples = ops.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to(dtype=ms.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * args.world_size + args.rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    if args.rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a DiT checkpoint.")
    parser.add_argument("--distributed", default=False, action="store_true", help="Enable distributed sampling")
    args = parser.parse_args()
    main(args)

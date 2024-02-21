# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import random
from typing import Optional

import mindspore as ms
from mindspore import Parameter, ParameterTuple, Tensor, context, nn, ops
from mindspore.amp import DynamicLossScaler, StaticLossScaler, LossScaler, all_finite
from mindspore.dataset import vision, transforms, ImageFolderDataset
from mindspore.communication import get_group_size, get_local_rank, get_rank, init

from models import DiT_models
from diffusion import create_diffusion, SpacedDiffusion
from diffusion.diffusion_utils import discretized_gaussian_log_likelihood, normal_kl
from diffusion.gaussian_diffusion import mean_flat, _extract_into_tensor
from experimental.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
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


def create_logger(logging_dir, args):
    """
    Create a logger that writes to a log file and stdout.
    """
    if is_master(args):  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


_ema_op = ops.MultitypeFuncGraph("ema_op")


@_ema_op.register("Tensor", "Tensor", "Tensor")
def ema_op(factor, ema_weight, weight):
    return ops.assign(ema_weight, ema_weight * factor + weight * (1 - factor))


class TrainStep(nn.Cell):
    """Training step with loss scale.

    The steps of model optimization are performed in the following order:
        1. calculate grad
        2. allreduce grad
        3. clip grad [optional]
        4. call optimizer
        5. ema weights [optional]
    """

    def __init__(
        self,
        network: nn.Cell,
        vae: nn.Cell,
        diffusion: SpacedDiffusion,
        optimizer: nn.Optimizer,
        scaler: LossScaler,
        ema_decay: Optional[float] = 0.9999,
        grad_clip_norm: Optional[float] = None,
    ):
        super().__init__()
        self.network = network.set_grad()
        self.vae = vae
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.scaler = scaler
        if isinstance(self.scaler, StaticLossScaler):
            self.drop_overflow = False
        elif isinstance(self.scaler, DynamicLossScaler):
            self.drop_overflow = True
        else:
            raise NotImplementedError(f"Unsupported scaler: {type(self.scaler)}")
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode == context.ParallelMode.STAND_ALONE:
            self.grad_reducer = nn.Identity()
        elif self.parallel_mode in (context.ParallelMode.DATA_PARALLEL, context.ParallelMode.HYBRID_PARALLEL):
            self.grad_reducer = nn.DistributedGradReducer(self.weights)
        else:
            raise NotImplementedError(f"When creating reducer, Got Unsupported parallel mode: {self.parallel_mode}")
        self._jit_config_dict = network.jit_config_dict

        self.ema_decay = ema_decay
        self.state_dict = ParameterTuple(network.get_parameters())  # same as model.state_dict in torch
        if self.ema_decay:
            self.ema_state_dict = self.state_dict.clone(prefix="ema", init="same")
        else:
            self.ema_state_dict = ParameterTuple(())  # placeholder
        self.grad_clip_norm = grad_clip_norm
        self.hyper_map = ops.HyperMap()

        def forward_fn(x, y):
            t = ops.randint(0, diffusion.num_timesteps, (x.shape[0],))
            # model_kwargs = dict(y=y)
            # loss = diffusion.training_losses(network, x, t, model_kwargs)
            noise = ops.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise=noise)
            model_output = network(x_t, t, y)
            b, c = x_t.shape[:2]
            assert model_output.shape == (b, c * 2) + x_t.shape[2:]
            model_output, model_var_values = ops.split(model_output, c, axis=1)

            # Learn the variance using the variational bound, but don't let it affect our mean prediction.
            # _vb_terms_bpd(model=lambda *_: frozen_out, x_start=x, x_t=x_t, t=t, clip_denoised=False) begin
            true_mean, _, true_log_variance_clipped = diffusion.q_posterior_mean_variance(x_start=x, x_t=x_t, t=t)
            # p_mean_variance(model=lambda *_: frozen_out, x_t, t, clip_denoised=False) begin
            min_log = _extract_into_tensor(diffusion.posterior_log_variance_clipped, t, x_t.shape)
            max_log = _extract_into_tensor(np.log(diffusion.betas), t, x_t.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            pred_xstart = diffusion._predict_xstart_from_eps(x_t=x_t, t=t, eps=ops.stop_gradient(model_output))
            model_mean, _, _ = diffusion.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)
            assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
            # p_mean_variance end
            kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
            kl = mean_flat(kl) / 0.693147  # np.log(2.0)
            decoder_nll = -discretized_gaussian_log_likelihood(x, means=model_mean, log_scales=0.5 * model_log_variance)
            decoder_nll = mean_flat(decoder_nll) / 0.693147  # np.log(2.0)
            # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
            vb = ops.where((t == 0), decoder_nll, kl)
            # _vb_terms_bpd end

            loss = mean_flat((noise - model_output) ** 2) + vb
            loss = loss.mean()
            loss = scaler.scale(loss)
            return loss

        self.grad_fn = ops.value_and_grad(forward_fn, grad_position=None, weights=self.weights, has_aux=False)

    def update(self, loss, grads):
        if self.grad_clip_norm:
            loss = ops.depend(loss, self.optimizer(ops.clip_by_global_norm(grads, clip_norm=self.grad_clip_norm)))
        else:
            loss = ops.depend(loss, self.optimizer(grads))

        if self.ema_decay:
            loss = ops.depend(
                loss, self.hyper_map(ops.partial(_ema_op, Tensor(self.ema_decay, ms.float32)), self.ema_state_dict, self.state_dict)
            )
        return loss

    def construct(self, x, y):
        # Map input images to latent space + normalize latents:
        # x = self.vae.encode(x)[0].sample().mul(0.18215)
        x = self.vae.diag_gauss_dist.sample(self.vae.encode(x)[0]).mul(0.18215)
        loss, grads = self.grad_fn(x, y)
        grads = self.grad_reducer(grads)
        loss = self.scaler.unscale(loss)
        grads = self.scaler.unscale(grads)

        if self.drop_overflow:
            status = all_finite(grads)
            if status:
                loss = self.update(loss, grads)
            loss = ops.depend(loss, self.scaler.adjust(status))
        else:
            loss = self.update(loss, grads)
        return loss


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    # Setup DDP:
    ms.set_context(mode=ms.GRAPH_MODE)
    device = init_distributed_device(args)
    assert args.global_batch_size % args.world_size == 0, f"Batch size must be divisible by world size."
    random_seed(args.global_seed)
    print(f"Starting rank={args.rank}, seed={args.global_seed}, world_size={args.world_size}.")

    # Setup an experiment folder:
    if is_master(args):
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, args)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None, args)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Note that parameter initialization is done within the DiT constructor
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.trainable_params()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        vision.Decode(to_pil=True),
        lambda pil_image: center_crop_arr(pil_image, args.image_size),
        vision.RandomHorizontalFlip(),
        vision.ToTensor(),
        vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False)
    ])
    dataset = ImageFolderDataset(
        args.data_path,
        num_shards=args.world_size,
        shard_id=args.rank,
        shuffle=True,
        num_parallel_workers=args.num_workers,
    )
    loader = dataset.map(
        transform,
        input_columns="image",
    ).batch(
        batch_size=int(args.global_batch_size // args.world_size),
        drop_remainder=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    model.set_train(True)  # important! This enables embedding dropout for classifier-free guidance
    vae.set_train(False)
    scaler = StaticLossScaler(128)
    train_one_step = TrainStep(
        model,
        vae,
        diffusion,
        opt,
        scaler,
    )

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            loss = train_one_step(x, y)
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = running_loss / log_steps
                # dist.all_reduce(ms.Tensor(avg_loss), op=dist.ReduceOp.SUM)
                # avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if is_master(args):
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    ms.save_checkpoint(train_one_step, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.set_train(False)  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--distributed", default=False, action="store_true", help="Enable distributed training")
    args = parser.parse_args()
    main(args)

import os
import sys
import argparse
import json
import torch
from torch.optim.lr_scheduler import LambdaLR, ConstantLR, _LRScheduler
from audiolm_pytorch import SoundStream, SoundStreamTrainer

if os.path.exists(
    "/home/maxime/Documents/Code/Neural_Network/Pytorch/audiolm-pytorch/src"
):
    sys.path.insert(
        0, "/home/maxime/Documents/Code/Neural_Network/Pytorch/audiolm-pytorch/src"
    )
else:
    sys.path.insert(
        0, "/home/mjacquelin/Project/Neural_Network/Pytorch/audiolm-pytorch/src"
    )
from utils import AttrDict, build_env


def get_parser():
    parser = argparse.ArgumentParser(
        description="Quantize using K-means clustering over acoustic features."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="The json file with parameters for training the model",
    )
    return parser


def main(args):

    with open(args.config) as f:
        data = f.read()

    json_config = json.loads(data)
    config = AttrDict(json_config)
    build_env(args.config, "config.json", config.trainer["results_folder"])

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    device = torch.device(config.device)

    soundstream = SoundStream(
        channels=config.model["channels"],
        strides=config.model["strides"],
        channel_mults=config.model["channel_mults"],
        codebook_dim=config.model["codebook_dim"],
        codebook_size=config.model["codebook_size"],
        
        rq_num_quantizers=config.model["rq_num_quantizers"],
        # this paper proposes using multi-headed residual vector quantization -
        # https://arxiv.org/abs/2305.02765
        rq_groups=config.model["rq_groups"],
        # whether to use residual lookup free quantization - there are now reports of
        # successful usage of this unpublished technique
        use_lookup_free_quantizer=config.model["use_lookup_free_quantizer"],
        # whether to use residual finite scalar quantization
        use_finite_scalar_quantizer=config.model["use_finite_scalar_quantizer"],
        # local attention receptive field at bottleneck
        attn_window_size=config.model["attn_window_size"],
        use_local_attn=config.model["use_local_attn"],
        # 2 local attention transformer blocks - the soundstream folks were not experts
        # with attention, so i took the liberty to add some. encodec went with lstms,
        # but attention should be better
        attn_depth=config.model["attn_depth"],
        multi_spectral_recon_loss_weight=config.model["multi_spectral_recon_loss_weight"],
        target_sample_hz = config.model["target_sample_hz"],
    )
    if config.trainer["scheduler"]=="ConstantLR":
        scheduler = ConstantLR
    if config.trainer["discr_scheduler"]=="ConstantLR":
        discr_scheduler = ConstantLR

    trainer = SoundStreamTrainer(
        soundstream,
        folder=config.trainer["folder"],
        batch_size=config.trainer["batch_size"],
        lr=config.trainer["lr"],
        grad_accum_every=config.trainer["grad_accum_every"],
        data_max_length_seconds=config.trainer["data_max_length_seconds"],
        scheduler=scheduler,
        scheduler_kwargs=config.trainer["scheduler_kwargs"],
        discr_scheduler=discr_scheduler,
        discr_scheduler_kwargs=config.trainer["discr_scheduler_kwargs"],
        num_train_steps=config.trainer["num_train_steps"],
        save_model_every=config.trainer["save_model_every"],
        save_results_every=config.trainer["save_results_every"],
        results_folder=config.trainer["results_folder"],
        use_wandb_tracking = config.trainer["use_wandb_tracking"],
    ).to(device)

    with trainer.wandb_tracker(project = config.wandb_project, run = config.wandb_run):
        trainer.train()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

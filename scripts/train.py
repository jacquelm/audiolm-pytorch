import os
import sys
import argparse
import json
import torch
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
from utils import AttrDict


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

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    device = torch.device(config.device)

    soundstream = SoundStream(
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
        # 2 local attention transformer blocks - the soundstream folks were not experts
        # with attention, so i took the liberty to add some. encodec went with lstms,
        # but attention should be better
        attn_depth=config.model["attn_depth"],
    )

    trainer = SoundStreamTrainer(
        soundstream,
        folder=config.trainer["folder"],
        batch_size=config.trainer["batch_size"],
        grad_accum_every=config.trainer["grad_accum_every"],
        data_max_length_seconds=config.trainer["data_max_length_seconds"],
        num_train_steps=config.trainer["num_train_steps"],
        save_model_every=config.trainer["save_model_every"],
        save_results_every=config.trainer["save_results_every"],
    ).to(device)

    trainer.train()

    # after a lot of training, you can test the autoencoding as so

    soundstream.eval()  # your soundstream must be in eval mode, to avoid having the residual dropout of the residual VQ necessary for training

    audio = torch.randn(10080).cuda()
    recons = soundstream(audio, return_recons_only=True)  # (1, 10080) - 1 channel


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

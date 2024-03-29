import argparse
import json
import torch
from src import SoundStream, SoundStreamTrainer, AttrDict


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
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Features file path. You don't need to enter acoustic model details if you have dumped features",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="The device to run the code on",
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

    device = torch.device(args.device)

    soundstream = SoundStream(
        codebook_size=config.codebook_size,
        rq_num_quantizers=config.rq_num_quantizers,
        # this paper proposes using multi-headed residual vector quantization - 
        # https://arxiv.org/abs/2305.02765
        rq_groups=config.rq_groups,
        # whether to use residual lookup free quantization - there are now reports of 
        # successful usage of this unpublished technique
        use_lookup_free_quantizer=config.use_lookup_free_quantizer,  
        # whether to use residual finite scalar quantization
        use_finite_scalar_quantizer=config.use_finite_scalar_quantizer,  
        # local attention receptive field at bottleneck
        attn_window_size=config.attn_window_size,  
        # 2 local attention transformer blocks - the soundstream folks were not experts 
        # with attention, so i took the liberty to add some. encodec went with lstms, 
        # but attention should be better
        attn_depth=config.attn_depth,  
    )

    trainer = SoundStreamTrainer(
        soundstream,
        folder=config.data_path,
        batch_size=config.batch_size,
        grad_accum_every=8,  # effective batch size of 32
        data_max_length_seconds=2,  # train on 2 second audio
        num_train_steps=1_000_000,
    ).cuda()

    trainer.train()

    # after a lot of training, you can test the autoencoding as so

    soundstream.eval()  # your soundstream must be in eval mode, to avoid having the residual dropout of the residual VQ necessary for training

    audio = torch.randn(10080).cuda()
    recons = soundstream(audio, return_recons_only=True)  # (1, 10080) - 1 channel


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

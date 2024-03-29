import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    from einops._torch_specific import allow_ops_in_compiled_graph

    allow_ops_in_compiled_graph()

from .audiolm_pytorch import AudioLM
from .soundstream import SoundStream, AudioLMSoundStream, MusicLMSoundStream
from .encodec import EncodecWrapper

from .audiolm_pytorch import SemanticTransformer, CoarseTransformer, FineTransformer
from .audiolm_pytorch import (
    FineTransformerWrapper,
    CoarseTransformerWrapper,
    SemanticTransformerWrapper,
)

from .vq_wav2vec import FairseqVQWav2Vec
from .hubert_kmeans import HubertWithKmeans

from .trainer import (
    SoundStreamTrainer,
    SemanticTransformerTrainer,
    FineTransformerTrainer,
    CoarseTransformerTrainer,
)

from .audiolm_pytorch import get_embeds

from .utils import AttrDict

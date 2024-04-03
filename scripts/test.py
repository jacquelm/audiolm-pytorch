from audiolm_pytorch import SoundStream
import torchaudio
import torch
from einops import rearrange, repeat, reduce

path = 'results/soundstream.20000.pt' 
model = SoundStream.init_and_load_from(path).to("cuda")
model.eval()

audio = torch.randn(1, 512 * 320).to('cuda')

codes = model.tokenize(audio)

# you can now train anything with the codebook ids
recon_audio_from_codes = model.decode_from_codebook_indices(codes)

# sanity check

print(
    torch.allclose(recon_audio_from_codes, model(audio, return_recons_only=True))
)

x, sr = torchaudio.load('input.wav')
x, sr = torchaudio.functional.resample(x, sr, 16000), 16000

x = x.to("cuda")
print("x", x.shape)
with torch.no_grad():
    # y = model.tokenize(x)
    # print("y shape", y.shape)
    
    # z = model.decode_from_codebook_indices(y)
    # print("z", z.shape)
        
    # z_flat = rearrange(z, '1 1 n -> 1 n')
    # print("z_flat", z_flat.shape)
    
    # z_flat = z_flat.to("cpu")
    # torchaudio.save('output.wav',z_flat, sr)
    
    # 
    # try 2
    z = model(x, return_recons_only = True)
    print("z", z.shape)
    
    z_flat = rearrange(z, '1 1 n -> 1 n')
    print("z_flat", z_flat.shape)
    
    z_flat = z_flat.to("cpu")
    torchaudio.save('output.wav',z_flat, sr)
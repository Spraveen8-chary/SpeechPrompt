import torch
ckpt = torch.load(r"promptspeech_best.pt")
print(ckpt.keys())

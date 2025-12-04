import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import yaml, torch
from pathlib import Path
from src.training.train_prompt import UnitSeqDataset, PromptSpeechASRStub

# load config
cfg = yaml.safe_load(open("configs/train_asr.yaml"))
ds_cfg = cfg["datasets"][0]
ds = UnitSeqDataset(ds_cfg["path"], max_src_len=cfg.get("max_src_len",800), max_tgt_len=cfg.get("max_tgt_len",200))
print("Dataset size:", len(ds))

# load first sample
src, tgt = ds[0]
print("src shape:", src.shape, "tgt shape:", tgt.shape)
print("sample tgt first tokens:", tgt[:10])

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# model
asr = PromptSpeechASRStub(gslm_ckpt=cfg["gslm_ckpt"], prompt_len=cfg.get("prompt_len",50), hidden_dim=cfg.get("hidden_dim",1024), device=device)
asr.to(device)
asr.train()

# single forward
src = src.unsqueeze(0).to(device)   # [1, L]
pooled = asr.encode_with_prompt(src)   # [1, D]
print("pooled shape:", pooled.shape)

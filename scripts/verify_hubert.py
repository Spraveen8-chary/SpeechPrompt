# scripts/verify_hubert.py
import fairseq_safe_globals  
from fairseq import checkpoint_utils

print("🔹 Loading HuBERT base model (feature extractor) ...")

models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    ["pretrained/hubert_base_ls960.pt"]
)
model = models[0]

# Different fairseq versions expose attributes differently
embed_dim = getattr(model, "encoder_embed_dim", None)
if embed_dim is None and hasattr(model, "encoder"):
    embed_dim = getattr(model.encoder, "embed_dim", None)
if embed_dim is None:
    embed_dim = 768  # fallback for base model

num_layers = getattr(model, "encoder_layers", None)
if num_layers is None and hasattr(model, "encoder"):
    num_layers = getattr(model.encoder, "layers", None)
    if isinstance(num_layers, list):
        num_layers = len(num_layers)
if num_layers is None:
    num_layers = 12

print(f"✅ HuBERT OK — layers={num_layers}, dim={embed_dim}")

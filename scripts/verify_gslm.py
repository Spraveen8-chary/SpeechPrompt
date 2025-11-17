# scripts/verify_gslm.py
import fairseq_safe_globals  # register all safe globals first
from fairseq import checkpoint_utils

print("🔹 Loading GSLM Unit LM (hubert100_lm/checkpoint_best.pt) ...")

try:
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        ["pretrained/hubert100_lm/checkpoint_best_fixed.pt"]
    )
    model = models[0]
    num_layers = None
    embed_dim = None

    if hasattr(model, "decoder"):
        dec = model.decoder
        num_layers = getattr(dec, "decoder_layers", None) or getattr(dec, "layers", None)
        if isinstance(num_layers, list):
            num_layers = len(num_layers)
        embed_dim = getattr(dec, "embed_dim", None)

    print(f"✅ GSLM OK — layers={num_layers}, dim={embed_dim}")
except Exception as e:
    print(f"❌ GSLM failed to load: {e}")

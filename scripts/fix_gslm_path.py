# scripts/fix_gslm_path.py
"""
Final Fairseq GSLM checkpoint repair script.
Fixes BOTH 'args' and 'cfg' paths to point to your local pretrained directory.
"""

import torch
from pathlib import Path
from types import SimpleNamespace

ckpt_path = Path("pretrained/hubert100_lm/checkpoint_best.pt")
fixed_path = Path("pretrained/hubert100_lm/checkpoint_best_fixed.pt")

print(f"🔧 Loading checkpoint: {ckpt_path}")
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print("📦 Keys:", list(state.keys()))

# --------------------------------------------------------------------
# Local paths
# --------------------------------------------------------------------
local_data_dir = str(Path("pretrained/hubert100_lm").resolve())
local_dict_path = str(Path(local_data_dir) / "dict.txt")
print(f"🧩 Using local dict.txt: {local_dict_path}")

# --------------------------------------------------------------------
# Patch args (legacy)
# --------------------------------------------------------------------
if "args" in state:
    args = state["args"]
    if hasattr(args, "dict_path"):
        args.dict_path = local_dict_path
    if hasattr(args, "data"):
        args.data = local_data_dir
    if hasattr(args, "label_dir"):
        args.label_dir = local_data_dir
    print("✅ Patched legacy args fields.")
else:
    args = SimpleNamespace(
        data=local_data_dir,
        dict_path=local_dict_path,
        task="language_modeling",
        labels=["layer6.km100"],
        label_dir=local_data_dir,
        sample_rate=16000,
    )
    state["args"] = args
    print("⚙️ Created missing 'args' stub.")

# --------------------------------------------------------------------
# Patch cfg (new OmegaConf structure)
# --------------------------------------------------------------------
if "cfg" in state:
    cfg = state["cfg"]
    try:
        if hasattr(cfg, "task"):
            task = cfg.task
            if hasattr(task, "data"):
                print(f"🔁 cfg.task.data -> {local_data_dir}")
                task.data = local_data_dir
            if hasattr(task, "label_dir"):
                print(f"🔁 cfg.task.label_dir -> {local_data_dir}")
                task.label_dir = local_data_dir
            if hasattr(task, "labels"):
                task.labels = ["layer6.km100"]
            cfg.task = task
        state["cfg"] = cfg
        print("✅ Patched new Fairseq cfg.task fields.")
    except Exception as e:
        print(f"⚠️ Could not patch cfg: {e}")
else:
    print("⚠️ No cfg found; skipping cfg patch.")

# --------------------------------------------------------------------
# Save
# --------------------------------------------------------------------
torch.save(state, fixed_path)
print(f"✅ Saved fully patched checkpoint to: {fixed_path}")

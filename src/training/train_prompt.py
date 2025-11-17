"""
Unified PromptSpeech Trainer
=============================
Trains PromptSpeech model across one or more datasets (Speech Commands, LibriSpeech, LJSpeech, etc.)
Implements paper-faithful architecture: frozen GSLM + prompt tuning + learnable verbalizer.

Author: Sarampenta Praveen (PromptSpeech Implementation)
"""
import sys
from pathlib import Path
sys.path.append(str(Path("scripts").resolve()))
import fairseq_safe_globals  

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from fairseq import checkpoint_utils
import numpy as np
from tqdm import tqdm
import os, json
import random

import yaml

# ================================================================
# 🧩 Dataset Loader (Handles Single and Multi-Dataset)
# ================================================================
class UnitDataset(Dataset):
    def __init__(self, root_dir, max_len=200, dataset_name="unknown"):
        self.root_dir = Path(root_dir)
        self.files = list(self.root_dir.rglob("*.npy"))
        assert len(self.files) > 0, f"No .npy files found in {root_dir}"
        self.max_len = max_len
        self.dataset_name = dataset_name
        self.class_map = self._infer_class_map()
        self.labels = [self._label_from_path(f) for f in self.files]

    def _infer_class_map(self):
        folders = sorted({f.parent.name for f in self.files})
        return {cls: i for i, cls in enumerate(folders)}

    def _label_from_path(self, path):
        return self.class_map[path.parent.name]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        arr = arr[: self.max_len]
        pad_len = self.max_len - len(arr)
        if pad_len > 0:
            arr = np.pad(arr, (0, pad_len), constant_values=0)
        label = self._label_from_path(self.files[idx])
        return torch.tensor(arr, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ================================================================
# 🧠 Unified PromptSpeech Model
# ================================================================
class PromptSpeechModel(nn.Module):
    def __init__(self, gslm_ckpt, vocab_size=100, prompt_len=20, hidden_dim=1024, num_classes=10, device="cpu"):
        super().__init__()
        self.device = device
        print("🔹 Loading frozen GSLM model...")
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([gslm_ckpt])
        self.gslm = models[0].to(device).eval()
        for p in self.gslm.parameters():
            p.requires_grad = False

        # Prompt + Verbalizer
        self.hidden_dim = getattr(self.gslm.decoder, "embed_dim", hidden_dim)
        self.prompt = nn.Parameter(torch.randn(prompt_len, self.hidden_dim) * 0.02)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, input_ids):
        """
        Robust forward pass compatible with all Fairseq GSLM versions.
        Handles both single-return and tuple-return transformer layers.
        """
        B = input_ids.size(0)
        input_ids = input_ids.to(self.device)

        # 1️⃣ Embed tokens
        token_emb = self.gslm.decoder.embed_tokens(input_ids)  # [B, L, D]

        # 2️⃣ Prepend continuous prompts
        prompt_emb = self.prompt.unsqueeze(0).expand(B, -1, -1)  # [B, P, D]
        x = torch.cat([prompt_emb, token_emb], dim=1).transpose(0, 1)  # [L+P, B, D]

        # 3️⃣ Manual decoding through frozen layers (robust unpacking)
        for layer in self.gslm.decoder.layers:
            out = layer(
                x,
                self_attn_mask=None,
                self_attn_padding_mask=None,
                need_attn=False,
                need_head_weights=False,
            )
            # Some versions return tuple, some single tensor
            if isinstance(out, tuple):
                x = out[0]
            elif hasattr(out, "out"):  # new-style DecoderLayerOutput
                x = out.out
            else:
                x = out  # single tensor

        # 4️⃣ Apply layer norm if exists
        if getattr(self.gslm.decoder, "layer_norm", None) is not None:
            x = self.gslm.decoder.layer_norm(x)

        # 5️⃣ Pool + classify
        pooled = x.transpose(0, 1).mean(dim=1)  # [B, D]
        logits = self.classifier(pooled)
        return logits


# ================================================================
# 🏋️ Training + Validation Loop
# ================================================================
def train_promptspeech(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥 Using device: {device}")

    # ---- Load multiple datasets ----
    datasets = []
    all_class_maps = {}

    fast_debug = cfg.get("fast_debug", False)
    samples_per_class = cfg.get("samples_per_class", 30)  # default: 30 samples/class

    for d in cfg["datasets"]:
        path = d["path"]
        name = d["name"]

        print(f"\n📂 Loading dataset: {name}")
        full_ds = UnitDataset(path, max_len=cfg.get("max_len", 200), dataset_name=name)
        class_map = full_ds.class_map
        all_class_maps[name] = class_map

        print(f"   → Total samples found: {len(full_ds)}")
        print(f"   → Total classes: {len(class_map)}")

        # --------------------- FAST DEBUG MODE (balanced sampling) ----------------------
        if fast_debug:
            print(f"⚡ FastDebug enabled → Selecting {samples_per_class} samples per class")

            # Build class → sample index mapping
            class_indices = {cls: [] for cls in class_map}
            for idx, file_path in enumerate(full_ds.files):
                cls_name = file_path.parent.name
                class_indices[cls_name].append(idx)

            selected = []

            # For every class: shuffle + pick samples_per_class
            for cls_name, idx_list in class_indices.items():
                idx_list = idx_list.copy()
                random.shuffle(idx_list)  # RANDOM sampling

                take = min(samples_per_class, len(idx_list))
                chosen = idx_list[:take]
                selected.extend(chosen)

                print(f"     Class '{cls_name}' → {take} samples selected")

            # Final subset
            ds = torch.utils.data.Subset(full_ds, selected)
            print(f"🔥 Final subset size for {name}: {len(selected)} samples\n")
        else:
            ds = full_ds

        datasets.append(ds)

    # -------------------------- COMBINED DATASET -----------------------------
    full_dataset = ConcatDataset(datasets)
    print(f"📦 Total training samples across all datasets: {len(full_dataset)}\n")

    dataloader = DataLoader(full_dataset,
                            batch_size=cfg.get("batch_size", 8),
                            shuffle=True)

    # ---- Model ----
    model = PromptSpeechModel(
        gslm_ckpt=cfg["gslm_ckpt"],
        vocab_size=cfg.get("vocab_size", 100),
        prompt_len=cfg.get("prompt_len", 20),
        hidden_dim=cfg.get("hidden_dim", 1024),
        num_classes=sum(len(ds.dataset.class_map) for ds in datasets),
        device=device,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = float(cfg.get("lr", 5e-3))
    optimizer = torch.optim.Adam(
        [model.prompt] + list(model.classifier.parameters()), lr=lr
    )

    epochs = cfg.get("epochs", 20)

    best_acc = 0.0
    patience = cfg.get("patience", 5)
    no_improve = 0

    # ---- Training Loop ----
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, preds = logits.max(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        print(f"🎯 Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f} | Acc={acc*100:.2f}%")

        # ---- Early stopping ----
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            ckpt_path = Path("results/checkpoints/promptspeech_best.pt")
            torch.save({
                "epoch": epoch + 1,
                "prompt": model.prompt.detach().cpu(),
                "classifier": model.classifier.state_dict(),
                "acc": acc,
                "class_maps": all_class_maps,
            }, ckpt_path)
            print(f"💾 Saved best model → {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("⏹ Early stopping (no improvement).")
                break

    print(f"🏁 Training finished. Best accuracy: {best_acc*100:.2f}%")

# ================================================================
# 🧾 CLI ENTRY
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_multi.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_promptspeech(cfg)

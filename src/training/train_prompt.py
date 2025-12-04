"""
Unified PromptSpeech Trainer (corrected)
=======================================
- Handles speech_classification (existing behavior)
- Adds an explicit ASR branch that validates dataset format
  and provides clear instructions for ASR dataset layout.
Author: Updated for you by ChatGPT (Sarampenta Praveen project)
"""
import sys
from pathlib import Path
sys.path.append(str(Path("scripts").resolve()))
import fairseq_safe_globals  

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from fairseq import checkpoint_utils
import numpy as np
from tqdm import tqdm
import os, json, random, math
import yaml

# -------------------------
# Dataset classes
# -------------------------
class UnitDataset(Dataset):
    """Classification-style dataset: each utterance has a parent folder -> class label."""
    def __init__(self, root_dir, max_len=200, dataset_name="unknown"):
        self.root_dir = Path(root_dir)
        self.files = sorted(list(self.root_dir.rglob("*.npy")))
        assert len(self.files) > 0, f"No .npy files found in {root_dir}"
        self.max_len = max_len
        self.dataset_name = dataset_name
        self.class_map = self._infer_class_map()
        # labels precomputed
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


class UnitSeqDataset(Dataset):
    """
    Sequence dataset for ASR:
    Expects paired files with same basename:
      - <utt>.units.npy    <-- discretized units (input)
      - <utt>.trans.txt    <-- plain text transcription (one line)
    OR:
      - <utt>.units.npy
      - <utt>.target.npy   <-- integer token ids for transcript (if you pre-tokenized)
    The loader builds a char2id map if text transcripts are used.
    """
    def __init__(self, root_dir, max_src_len=800, max_tgt_len=200, dataset_name="asr"):
        self.root_dir = Path(root_dir)
        self.units_files = sorted(list(self.root_dir.rglob("*.npy")))
        # Filter to only keep those that *look like* source unit files (no guarantee)
        # We'll pair by basename.
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.dataset_name = dataset_name

        # Build pairs
        self.pairs = []
        for units_path in self.units_files:
            base = units_path.with_suffix("")  # remove last suffix
            # try .trans.txt
            txt_path = units_path.with_suffix(".trans.txt")
            target_npy = units_path.with_suffix(".target.npy")
            if txt_path.exists():
                self.pairs.append((units_path, txt_path))
            elif target_npy.exists():
                self.pairs.append((units_path, target_npy))
            else:
                # no pair found; skip
                continue

        if len(self.pairs) == 0:
            raise ValueError(
                f"No paired ASR files found in {root_dir}.\n"
                "ASR expects paired files: <utt>.units.npy + <utt>.trans.txt OR <utt>.target.npy"
            )

        # Build char vocab if trans.txt used
        self.use_token_ids = self.pairs and self.pairs[0][1].suffix == ".npy"
        self.char2id = {}
        self.id2char = {}
        if not self.use_token_ids:
            # construct char vocab from all transcriptions
            chars = set()
            for _, txt in self.pairs:
                s = txt.read_text(encoding="utf-8").strip()
                chars.update(list(s))
            # Add special <pad> and <eos>
            chars = sorted(chars)
            self.char2id = {c: i+2 for i, c in enumerate(chars)}  # reserve 0 pad, 1 eos
            self.char2id["<pad>"] = 0
            self.char2id["<eos>"] = 1
            self.id2char = {i:s for s,i in self.char2id.items()}

    def __len__(self):
        return len(self.pairs)

    def _load_src(self, units_path):
        arr = np.load(units_path)
        arr = arr[: self.max_src_len]
        pad_len = self.max_src_len - len(arr)
        if pad_len > 0:
            arr = np.pad(arr, (0, pad_len), constant_values=0)
        return torch.tensor(arr, dtype=torch.long)

    def _load_tgt_from_txt(self, txt_path):
        s = txt_path.read_text(encoding="utf-8").strip()
        ids = [self.char2id.get(c, 0) for c in s]  # unknown -> pad(0)
        ids = ids[: self.max_tgt_len - 1]  # reserve space for <eos>
        ids.append(self.char2id["<eos>"])
        pad_len = self.max_tgt_len - len(ids)
        if pad_len > 0:
            ids.extend([self.char2id["<pad>"]] * pad_len)
        return torch.tensor(ids, dtype=torch.long)

    def _load_tgt_from_npy(self, npy_path):
        arr = np.load(npy_path)
        arr = arr[: self.max_tgt_len]
        pad_len = self.max_tgt_len - len(arr)
        if pad_len > 0:
            arr = np.pad(arr, (0, pad_len), constant_values=0)
        return torch.tensor(arr, dtype=torch.long)

    def __getitem__(self, idx):
        src_path, tgt_path = self.pairs[idx]
        src = self._load_src(src_path)
        if self.use_token_ids:
            tgt = self._load_tgt_from_npy(tgt_path)
        else:
            tgt = self._load_tgt_from_txt(tgt_path)
        return src, tgt


# ================================================================
# Models
# ================================================================
class PromptSpeechModel(nn.Module):
    """Original classification-style PromptSpeech (keeps frozen GSLM decoder + prompts)"""
    def __init__(self, gslm_ckpt, vocab_size=100, prompt_len=20, hidden_dim=1024, num_classes=10, device="cpu"):
        super().__init__()
        self.device = device
        print("🔹 Loading frozen GSLM model...")
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([gslm_ckpt])
        self.gslm = models[0].to(device).eval()
        for p in self.gslm.parameters():
            p.requires_grad = False

        # Prompt + classifier
        self.hidden_dim = getattr(self.gslm.decoder, "embed_dim", hidden_dim)
        self.prompt = nn.Parameter(torch.randn(prompt_len, self.hidden_dim) * 0.02)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, input_ids):
        B = input_ids.size(0)
        input_ids = input_ids.to(self.device)

        # embed tokens
        token_emb = self.gslm.decoder.embed_tokens(input_ids)  # [B, L, D]
        # prepend prompt
        prompt_emb = self.prompt.unsqueeze(0).expand(B, -1, -1)  # [B, P, D]
        x = torch.cat([prompt_emb, token_emb], dim=1).transpose(0, 1)  # [L+P, B, D]

        for layer in self.gslm.decoder.layers:
            out = layer(
                x,
                self_attn_mask=None,
                self_attn_padding_mask=None,
                need_attn=False,
                need_head_weights=False,
            )
            if isinstance(out, tuple):
                x = out[0]
            elif hasattr(out, "out"):
                x = out.out
            else:
                x = out

        if getattr(self.gslm.decoder, "layer_norm", None) is not None:
            x = self.gslm.decoder.layer_norm(x)

        pooled = x.transpose(0, 1).mean(dim=1)  # [B, D]
        logits = self.classifier(pooled)
        return logits


# Note: ASR model implementation is task-specific and requires paired data + a verbalizer.
# We include a small stub model that loads the GSLM, prompt, and exposes a 'pool' vector
# that a downstream verbalizer can use to predict sequences. A full autoregressive
# training loop (teacher forcing on LM outputs) is more involved and depends on your
# exact target format. For safety we will validate inputs and provide clear steps.
class PromptSpeechASRStub(nn.Module):
    """
    ASR stub:
    - loads frozen GSLM
    - provides prompt parameter and a pooled representation
    - a separate verbalizer (linear layer) can map LM logits/probabilities to chars
    """
    def __init__(self, gslm_ckpt, prompt_len=50, hidden_dim=1024, device="cpu"):
        super().__init__()
        self.device = device
        print("🔹 Loading frozen GSLM (ASR stub)...")
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([gslm_ckpt])
        self.gslm = models[0].to(device).eval()
        for p in self.gslm.parameters():
            p.requires_grad = False

        self.hidden_dim = getattr(self.gslm.decoder, "embed_dim", hidden_dim)
        self.prompt = nn.Parameter(torch.randn(prompt_len, self.hidden_dim) * 0.02)

    def encode_with_prompt(self, input_ids):
        """Return the decoder-layer-pooled vector (B, D) given input_ids (B, L)"""
        B = input_ids.size(0)
        input_ids = input_ids.to(self.device)
        token_emb = self.gslm.decoder.embed_tokens(input_ids)  # [B, L, D]
        prompt_emb = self.prompt.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([prompt_emb, token_emb], dim=1).transpose(0, 1)
        for layer in self.gslm.decoder.layers:
            out = layer(
                x,
                self_attn_mask=None,
                self_attn_padding_mask=None,
                need_attn=False,
                need_head_weights=False,
            )
            if isinstance(out, tuple):
                x = out[0]
            elif hasattr(out, "out"):
                x = out.out
            else:
                x = out
        if getattr(self.gslm.decoder, "layer_norm", None) is not None:
            x = self.gslm.decoder.layer_norm(x)
        pooled = x.transpose(0, 1).mean(dim=1)  # [B, D]
        return pooled


# ================================================================
# Training orchestration
# ================================================================
def train_promptspeech(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥 Using device: {device}")

    task = cfg.get("task", "speech_classification")
    fast_debug = cfg.get("fast_debug", False)
    samples_per_class = cfg.get("samples_per_class", 30)

    datasets = []
    all_class_maps = {}

    # load datasets depending on the task
    for d in cfg["datasets"]:
        path = d["path"]
        name = d.get("name", "dataset")
        print(f"\n📂 Loading dataset: {name} ({path})")
        if task == "speech_classification":
            full_ds = UnitDataset(path, max_len=cfg.get("max_len", 200), dataset_name=name)
            class_map = full_ds.class_map
            all_class_maps[name] = class_map
            print(f"   → Total samples found: {len(full_ds)}")
            print(f"   → Total classes: {len(class_map)}")

            if fast_debug:
                print(f"⚡ FastDebug enabled → Selecting {samples_per_class} samples per class")
                # Build class -> indices
                class_indices = {cls: [] for cls in class_map}
                for idx, file_path in enumerate(full_ds.files):
                    cls_name = file_path.parent.name
                    class_indices[cls_name].append(idx)

                selected = []
                for cls_name, idx_list in class_indices.items():
                    random.shuffle(idx_list)
                    take = min(samples_per_class, len(idx_list))
                    chosen = idx_list[:take]
                    selected.extend(chosen)
                    print(f"     Class '{cls_name}' → {take} samples selected")
                ds = torch.utils.data.Subset(full_ds, selected)
                print(f"🔥 Final subset size for {name}: {len(selected)} samples\n")
            else:
                ds = full_ds

            datasets.append(ds)

        elif task == "asr":
            # For ASR we need paired data (units + transcript/targets).
            # UnitSeqDataset looks for <utt>.units.npy + <utt>.trans.txt OR <utt>.target.npy
            try:
                full_ds = UnitSeqDataset(path, max_src_len=cfg.get("max_src_len", 800),
                                        max_tgt_len=cfg.get("max_tgt_len", 200),
                                        dataset_name=name)
            except Exception as e:
                raise RuntimeError(
                    "ASR dataset loading failed. Please prepare your ASR dataset as PAIRS:\n"
                    " - <utt>.units.npy    (the discretized unit sequence for the audio input)\n"
                    " - <utt>.trans.txt    (plain transcription text)  OR  <utt>.target.npy (token ids)\n\n"
                    "Examples:\n"
                    "  data/units/libri/0001.units.npy\n"
                    "  data/units/libri/0001.trans.txt\n\n"
                    "If you already have transcripts in a single file, convert them to per-utt .trans.txt files\n"
                    "or generate .target.npy token id sequences beforehand.\n\n"
                    f"Original error: {e}"
                )
            print(f"   → Total paired utterances found: {len(full_ds)}")
            datasets.append(full_ds)

        else:
            raise ValueError(f"Unknown task: {task}. Supported: speech_classification, asr")

    if len(datasets) == 0:
        raise RuntimeError("No datasets were loaded. Check configs.")

    full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"\n📦 Total training samples across all datasets: {len(full_dataset)}\n")

    # DataLoader
    dataloader = DataLoader(full_dataset,
                            batch_size=cfg.get("batch_size", 8),
                            shuffle=True,
                            collate_fn=None,
                            num_workers=0,
                            pin_memory=False,
                            drop_last=False)

    # ------------------ Task-specific model creation ------------------
    save_dir = Path(cfg.get("save_dir", "results/checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "promptspeech_best.pt"

    if task == "speech_classification":
        # compute total classes across concatenated datasets
        total_classes = sum(len(ds.dataset.class_map) for ds in datasets)
        model = PromptSpeechModel(
            gslm_ckpt=cfg["gslm_ckpt"],
            vocab_size=cfg.get("vocab_size", 100),
            prompt_len=cfg.get("prompt_len", 20),
            hidden_dim=cfg.get("hidden_dim", 1024),
            num_classes=total_classes,
            device=device,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([model.prompt] + list(model.classifier.parameters()),
                                     lr=float(cfg.get("lr", 5e-3)))

        epochs = cfg.get("epochs", 20)
        best_acc = 0.0
        patience = cfg.get("patience", 5)
        no_improve = 0

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

            acc = correct / total if total > 0 else 0.0
            avg_loss = total_loss / total if total > 0 else 0.0
            print(f"🎯 Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f} | Acc={acc*100:.2f}%")

            if acc > best_acc:
                best_acc = acc
                no_improve = 0
                # Save task-specific checkpoint
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
        return

    elif task == "asr":
        # The ASR training flow is more involved. We'll perform a safety check and provide instructions.
        # Create ASR stub: pooled encoder + verbalizer placeholder
        # The actual training loop for ASR (autoregressive unit generation / verbalizer learning)
        # requires clear target format. Here we create a stub that will save prompt and a placeholder verbalizer.
        print("⚠️ ASR task requested. Performing dataset validation and preparing ASR stub.")
        # check whether dataset is UnitSeqDataset
        first_ds = datasets[0]
        if not isinstance(first_ds, UnitSeqDataset):
            raise RuntimeError("ASR training requires UnitSeqDataset (paired units + transcript).")

        # Build char/token vocab if using trans.txt
        use_token_ids = getattr(first_ds, "use_token_ids", False)
        if not use_token_ids:
            # we have char2id mapping
            char2id = first_ds.char2id
            id2char = {i:c for c,i in char2id.items()}
            num_tokens = len(char2id)
            print(f"   → Discovered {num_tokens} character tokens for ASR.")
        else:
            # If target token ids are already provided, we need to infer vocab size from data
            # (scan a few files)
            print("   → Targets are pre-tokenized (target .npy files). Using provided token ids.")
            # infer max id
            max_id = 0
            for _, tgt_path in first_ds.pairs[:200]:
                arr = np.load(tgt_path)
                if arr.size:
                    max_id = max(max_id, int(arr.max()))
            num_tokens = int(max_id) + 1
            print(f"   → Inferred target vocab size: {num_tokens}")

        # Build ASR stub model
        asr_model = PromptSpeechASRStub(
            gslm_ckpt=cfg["gslm_ckpt"],
            prompt_len=cfg.get("prompt_len", 50),
            hidden_dim=cfg.get("hidden_dim", 1024),
            device=device,
        ).to(device)

        # Create a verbalizer placeholder: maps pooled vector -> token distribution
        verbalizer = nn.Linear(asr_model.hidden_dim, num_tokens).to(device)

        # Optimizer: only train prompt + verbalizer for now
        optimizer = torch.optim.Adam([asr_model.prompt, *list(verbalizer.parameters())],
                                     lr=float(cfg.get("lr", 5e-4)))

        # NOTE: here we DO NOT implement a full autoregressive teacher-forcing training loop,
        # because the exact target format (units vs token ids vs char mapping) is data-dependent.
        # We provide a simple example supervised training using pooled representation -> token prediction
        # at a single timestep (this is a *placeholder* and NOT a full ASR).
        epochs = cfg.get("epochs", 1)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # assume 0 is pad

        print("\n⚠️ IMPORTANT: The ASR branch in this repository is a protected path that expects\n"
              "paired unit+transcription data. The current code will run a simple placeholder\n"
              "train that maps pooled representation -> token ids (one token) for demonstration.\n"
              "For real ASR sequence generation training (autoregressive over tokens/units),\n"
              "please ask me to implement it for your exact target format (char-level or token ids).\n")

        # quick demo train loop (placeholder)
        best_loss = math.inf
        for epoch in range(epochs):
            asr_model.train()
            total_loss = 0.0
            total_samples = 0
            for src, tgt in tqdm(dataloader, desc=f"ASR Epoch {epoch+1}/{epochs}"):

                print("DEBUG: batch loaded. src.shape=", getattr(src, "shape", None), "tgt.shape=", getattr(tgt, "shape", None))

                # src: [B, L_src], tgt: [B, L_tgt]
                src = src.to(device)
                tgt = tgt.to(device)
                pooled = asr_model.encode_with_prompt(src)  # [B, D]

                # Placeholder: predict first token of tgt (teacher forcing simplified)
                first_token = tgt[:, 0].long()  # [B]
                logits = verbalizer(pooled)  # [B, num_tokens]
                loss = criterion(logits, first_token)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * src.size(0)
                total_samples += src.size(0)

            avg_loss = total_loss / total_samples if total_samples>0 else 0.0
            print(f"ASR Epoch {epoch+1}/{epochs} | AvgLoss={avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                # Save placeholder ckpt (contains prompt + verbalizer)
                torch.save({
                    "epoch": epoch+1,
                    "prompt": asr_model.prompt.detach().cpu(),
                    "verbalizer": verbalizer.state_dict(),
                    "loss": best_loss,
                    "note": "PLACEHOLDER ASR ckpt. Implement full autoregressive training for production."
                }, ckpt_path)
                print(f"💾 Saved ASR stub checkpoint → {ckpt_path}")

        print("🏁 ASR placeholder training finished. Read the NOTE above for next steps.")
        return

    else:
        raise ValueError(f"Unsupported task: {task}")

# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_multi.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_promptspeech(cfg)

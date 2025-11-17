"""
models/gslm/gslm_wrapper.py

Paper-faithful GSLM wrapper for PromptSpeech reproduction.
Implements:
- Frozen pre-trained GSLM (decoder-only) from fairseq.
- Input Prompt Tuning and Deep Prefix Tuning (pI, pK, pV).
- No GPT or local substitute.

Requires:
  pip install fairseq
  and a pretrained GSLM checkpoint (e.g., facebook/gslm-unit-base)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils

class GSLMWrapper(nn.Module):
    def __init__(
        self,
        pretrained_path: str,
        prompt_len: int = 100,
        deep: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.deep = deep
        self.prompt_len = prompt_len

        # 1️⃣ Load pretrained GSLM model (frozen backbone)
        print(f"🔹 Loading pretrained GSLM from {pretrained_path} ...")
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([pretrained_path])
        self.gslm = models[0].eval().to(device)
        for p in self.gslm.parameters():
            p.requires_grad = False
        print("✅ Pretrained GSLM loaded and frozen.")

        # 2️⃣ Extract architecture info
        try:
            self.embed_dim = self.gslm.decoder.embed_dim
            self.n_layers = self.gslm.decoder.layers
        except AttributeError:
            self.embed_dim = getattr(self.gslm, "decoder_embed_dim", 512)
            self.n_layers = getattr(self.gslm, "decoder_layers", 12)

        # 3️⃣ Initialize prompts (trainable)
        self.input_prompts = nn.Parameter(
            torch.randn(prompt_len, self.embed_dim, device=device) * 0.02
        )

        if self.deep:
            self.key_prompts = nn.ParameterList([
                nn.Parameter(torch.randn(prompt_len, self.embed_dim, device=device) * 0.02)
                for _ in range(self.n_layers)
            ])
            self.value_prompts = nn.ParameterList([
                nn.Parameter(torch.randn(prompt_len, self.embed_dim, device=device) * 0.02)
                for _ in range(self.n_layers)
            ])
            print(f"✅ Deep prefix prompts initialized for {self.n_layers} layers.")

    def forward(self, input_ids: torch.LongTensor):
        """
        input_ids: (B, T) discrete unit tokens
        Returns:
            hidden_states: last-layer hidden states after prompt injection
        """
        B, T = input_ids.shape
        input_ids = input_ids.to(self.device)

        # 4️⃣ Forward pass through GSLM
        # Embed input units using frozen embeddings
        embed_tokens = self.gslm.decoder.embed_tokens(input_ids)
        # Prepend input prompts
        prompt_expand = self.input_prompts.unsqueeze(0).expand(B, -1, -1)
        inputs_embeds = torch.cat([prompt_expand, embed_tokens], dim=1)

        # Pass through layers with optional deep prompt tuning
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.gslm.decoder.layers):
            if self.deep:
                # Prefix tuning injection (add learned key/value prompts)
                kv_prompt_k = self.key_prompts[layer_idx].unsqueeze(0).expand(B, -1, -1)
                kv_prompt_v = self.value_prompts[layer_idx].unsqueeze(0).expand(B, -1, -1)

                # Pass with prompts as additional context
                hidden_states, _, _ = layer(
                    hidden_states,
                    self_attn_padding_mask=None,
                    self_attn_bias=None,
                    attn_mask=None,
                    encoder_out=None,
                    past_key_value=(kv_prompt_k, kv_prompt_v),
                )
            else:
                hidden_states, _, _ = layer(
                    hidden_states,
                    self_attn_padding_mask=None,
                    self_attn_bias=None,
                    attn_mask=None,
                    encoder_out=None,
                    past_key_value=None,
                )

        # Output projection via frozen LM head
        logits = self.gslm.decoder.output_layer(hidden_states)

        return {"logits": logits, "hidden_states": hidden_states}

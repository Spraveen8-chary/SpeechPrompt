"""
models/verbalizers/fixed_verbalizer.py

Implements the Fixed Verbalizer for PromptSpeech.
A Fixed Verbalizer maps model vocabulary tokens (discrete speech units)
to class labels using a predefined, static mapping.

Author: Sarampenta Praveen (PromptSpeech Reproduction)
"""

import torch
import torch.nn as nn


class FixedVerbalizer(nn.Module):
    """
    Fixed (non-trainable) verbalizer mapping discrete unit indices → class logits.

    Example verbalizer_map:
        {
            "yes": [1, 5, 9],
            "no": [2, 8, 11],
            "up": [3, 6, 12],
            "down": [4, 7, 10]
        }
    Each class is represented by a set of token IDs corresponding to discrete units.
    """

    def __init__(self, verbalizer_map: dict, vocab_size: int, num_classes: int):
        super().__init__()
        self.verbalizer_map = verbalizer_map
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, lm_logits: torch.Tensor):
        """
        Args:
            lm_logits: torch.Tensor of shape (B, T, V)
                Logits from GSLM (B=batch, T=sequence, V=vocab_size)

        Returns:
            class_logits: torch.Tensor of shape (B, num_classes)
                Class-level aggregated probabilities.
        """
        probs = self.softmax(lm_logits)  # (B, T, V)
        B, T, V = probs.size()
        class_logits = torch.zeros(B, self.num_classes, device=lm_logits.device)

        for class_idx, (_, token_ids) in enumerate(self.verbalizer_map.items()):
            token_ids = torch.tensor(token_ids, device=lm_logits.device)
            token_probs = probs[..., token_ids]  # (B, T, len(token_ids))
            class_mean = token_probs.mean(dim=(1, 2))  # mean across time & tokens
            class_logits[:, class_idx] = class_mean

        return class_logits


# ---------------------------------------------------------------------
# ✅ Quick self-test for debug or CI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    verbalizer_map = {
        "yes": [1, 5, 9],
        "no": [2, 8, 11],
        "up": [3, 6, 12],
        "down": [4, 7, 10],
    }

    vocab_size = 100
    num_classes = len(verbalizer_map)

    model = FixedVerbalizer(verbalizer_map, vocab_size, num_classes)
    lm_logits = torch.randn(2, 5, vocab_size)
    out = model(lm_logits)
    print("✅ Output shape:", out.shape)
    print("✅ Sample output:", out[0])

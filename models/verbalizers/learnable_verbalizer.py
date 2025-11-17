"""
models/verbalizers/learnable_verbalizer.py

Learnable Verbalizer with an automatic adapter that
maps incoming pooled hidden dims to the expected hidden_size
if they do not match. This makes training robust to
intermittent shape mismatches.

Author: Adapted for robust training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableVerbalizer(nn.Module):
    """
    Learnable Verbalizer with optional runtime adapter.

    Args:
        hidden_size (int): nominal expected hidden size (e.g. 512)
        num_classes (int): number of target classes
        temperature (float): scaling factor for logits
        pooling (str): 'mean' or 'last'
    """

    def __init__(self, hidden_size: int, num_classes: int, temperature: float = 0.01, pooling: str = "mean"):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.pooling = pooling

        # Main classifier W: expects input size == self.hidden_size
        self.W = nn.Linear(self.hidden_size, self.num_classes)
        nn.init.xavier_uniform_(self.W.weight)
        if self.W.bias is not None:
            nn.init.zeros_(self.W.bias)

        # Optional adapter: created lazily on first forward if needed
        self.adapter = None  # type: nn.Module | None
        self._adapter_in_dim = None

    def _build_adapter(self, in_dim, device):
        """
        Create a learned linear adapter from in_dim -> hidden_size.
        """
        self.adapter = nn.Linear(in_dim, self.hidden_size).to(device)
        nn.init.xavier_uniform_(self.adapter.weight)
        if self.adapter.bias is not None:
            nn.init.zeros_(self.adapter.bias)
        self._adapter_in_dim = int(in_dim)

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: either (B, T, H) or (B, H) depending on caller
        Returns:
            logits: (B, num_classes)
        """
        # Accept either pooled inputs (B,H) or raw sequence (B,T,H)
        if hidden_states.ndim == 3:
            if self.pooling == "mean":
                pooled = hidden_states.mean(dim=1)
            elif self.pooling == "last":
                pooled = hidden_states[:, -1, :]
            else:
                raise ValueError("pooling must be 'mean' or 'last'")
        elif hidden_states.ndim == 2:
            pooled = hidden_states
        else:
            raise ValueError(f"hidden_states must be 2D or 3D tensor, got ndim={hidden_states.ndim}")

        B, in_dim = pooled.shape

        # If incoming dim doesn't match expected hidden_size, lazily create adapter.
        if in_dim != self.hidden_size:
            # If adapter exists but input dim changed -> recreate
            if (self.adapter is None) or (self._adapter_in_dim != in_dim):
                # create adapter on same device as pooled
                device = pooled.device
                self._build_adapter(in_dim, device)
                # log a clear warning once
                print(f"⚠️ LearnableVerbalizer: adapter created to map {in_dim} -> {self.hidden_size}")

            # map to expected hidden_size
            pooled = self.adapter(pooled)

        # Now pooled is (B, hidden_size)
        logits = self.W(pooled) / self.temperature
        return logits


# Quick local test when run as script
if __name__ == "__main__":
    # Simulate two cases: in_dim == hidden_size and in_dim != hidden_size
    import torch

    # case 1: matching dims
    v = LearnableVerbalizer(hidden_size=512, num_classes=4)
    x = torch.randn(2, 512)
    print("case match ->", v(x).shape)

    # case 2: smaller incoming dim
    v2 = LearnableVerbalizer(hidden_size=512, num_classes=4)
    x2 = torch.randn(2, 16)
    out = v2(x2)
    print("case adapt ->", out.shape)

# scripts/fairseq_safe_globals.py
"""
Final robust safe-globals registration for Fairseq / OmegaConf checkpoints under PyTorch >= 2.6.

Covers:
 - builtins primitives and containers (int, float, bool, dict, list, tuple, set, bytes, bytearray, range, complex)
 - collections (defaultdict, OrderedDict, deque)
 - argparse.Namespace
 - typing.Any, Optional, Union, List, Dict, Tuple
 - omegaconf types (DictConfig, ListConfig, nodes.* including AnyNode, ValueNode, etc.)
 - fairseq.data.dictionary.Dictionary (if available)
 - a few common exception/utility classes

Usage:
    import scripts.fairseq_safe_globals
    # then load fairseq checkpoints
"""

import torch
import argparse
import importlib
import builtins
import collections
import typing

safe_types = set()

# --- builtins primitives & containers (explicit)
for t in (builtins.int, builtins.float, builtins.bool, builtins.str,
          builtins.bytes, builtins.bytearray, builtins.range, builtins.complex):
    safe_types.add(t)

# built-in containers
safe_types.update([builtins.dict, builtins.list, builtins.tuple, builtins.set, builtins.frozenset])

# --- collections
safe_types.update([collections.defaultdict, collections.OrderedDict, collections.deque])

# --- argparse
safe_types.add(argparse.Namespace)

# --- typing helpers
try:
    safe_types.add(typing.Any)
    for name in ("Optional", "Union", "List", "Dict", "Tuple"):
        if hasattr(typing, name):
            safe_types.add(getattr(typing, name))
except Exception:
    pass

# --- OmegaConf related classes (if available)
try:
    oc = importlib.import_module("omegaconf")
    if hasattr(oc, "dictconfig") and hasattr(oc.dictconfig, "DictConfig"):
        safe_types.add(oc.dictconfig.DictConfig)
    if hasattr(oc, "listconfig") and hasattr(oc.listconfig, "ListConfig"):
        safe_types.add(oc.listconfig.ListConfig)
    if getattr(oc, "base", None) is not None and getattr(oc.base, "ContainerMetadata", None) is not None:
        safe_types.add(oc.base.ContainerMetadata)
    if hasattr(oc, "nodes"):
        for name in dir(oc.nodes):
            try:
                attr = getattr(oc.nodes, name)
                if isinstance(attr, type):
                    safe_types.add(attr)
            except Exception:
                continue
except Exception:
    pass

# --- fairseq Dictionary (if present)
try:
    fd_mod = importlib.import_module("fairseq.data.dictionary")
    if hasattr(fd_mod, "Dictionary"):
        safe_types.add(fd_mod.Dictionary)
except Exception:
    pass

# --- common exceptions/utilities
try:
    safe_types.update([FileNotFoundError, StopIteration, MemoryError])
except Exception:
    pass

# --- finalize registration (deduplicate & deterministic ordering)
ordered = []
seen = set()
for t in safe_types:
    if t not in seen and t is not None:
        ordered.append(t)
        seen.add(t)

torch.serialization.add_safe_globals(ordered)

print("✅ Registered FINAL robust safe globals for Fairseq/OmegaConf checkpoint loading.")
print(f"   Registered {len(ordered)} types (examples): {[getattr(t,'__name__',str(t)) for t in ordered[:30]]}")

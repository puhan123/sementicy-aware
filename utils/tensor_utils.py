"""Utility helpers for tensor manipulation used across measurement workflows."""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def ensure_dense(tensor: torch.Tensor) -> torch.Tensor:
    """Return a dense view of ``tensor`` without modifying the input."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("ensure_dense expects a torch.Tensor input")
    if tensor.is_sparse:
        return tensor.to_dense()
    return tensor


def maybe_to_sparse(tensor: torch.Tensor, use_sparse: bool) -> torch.Tensor:
    """Convert ``tensor`` to sparse storage when ``use_sparse`` is True."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("maybe_to_sparse expects a torch.Tensor input")
    if use_sparse:
        return tensor if tensor.is_sparse else tensor.to_sparse()
    return tensor.coalesce() if tensor.is_sparse else tensor


def standardize_activations(
    activations: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    to_sparse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize activations and optionally return sparse storage.

    Args:
        activations: A tensor of shape ``(tokens, hidden)`` or ``(batch, seq, hidden)``.
        mean: Optional pre-computed mean per hidden dimension.
        std: Optional pre-computed std per hidden dimension.
        eps: Small epsilon to avoid divide-by-zero.
        to_sparse: Whether to convert the standardized activations into sparse storage.

    Returns:
        standardized: Standardized activations with optional sparse storage.
        mean: Mean used for standardization.
        std: Standard deviation used for standardization (clamped by ``eps``).
    """
    dense = ensure_dense(activations)
    if dense.dim() == 3:
        dense = dense.reshape(-1, dense.size(-1))
    elif dense.dim() == 1:
        dense = dense.unsqueeze(0)

    dense = dense.to(torch.float32)

    if mean is None or std is None:
        mean = dense.mean(dim=0)
        std = dense.std(dim=0, unbiased=False)
    else:
        mean = mean.to(dense.device, dtype=dense.dtype)
        std = std.to(dense.device, dtype=dense.dtype)

    std = std.clamp_min(eps)
    standardized = (dense - mean) / std
    standardized = maybe_to_sparse(standardized, to_sparse)
    return standardized, mean, std


__all__ = [
    "ensure_dense",
    "maybe_to_sparse",
    "standardize_activations",
]

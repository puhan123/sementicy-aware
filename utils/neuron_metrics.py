"""Utility functions for computing neuron-level metrics.

This module provides helpers to estimate per-neuron activity (phi),
contextual importance (c_i), and their combined scores (kappa).
The implementations are designed to work with tensors that contain
neuron activations or logit contributions collected across batches of
examples.  They intentionally avoid referencing any particular
experiment pipeline so that they can be reused by multiple scripts.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import torch

MetricReduction = Literal["mean", "sum", "none"]
PhiMethod = Literal["zscore", "attention_mean"]
CiMethod = Literal["lse", "target_logit", "logit_diff", "kl"]


def _canonical_dim(dim: int, ndim: int) -> int:
    """Convert a possibly negative dimension index to the canonical form."""
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(f"Dimension {dim} out of range for tensor with {ndim} dims")
    return dim


def compute_phi(
    activations: torch.Tensor,
    *,
    method: PhiMethod = "zscore",
    sample_dim: int = 0,
    reduction: MetricReduction = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Aggregate neuron activations into a φᵢ score.

    Args:
        activations: Tensor containing neuron activations. The dimension
            identified by ``sample_dim`` is treated as the sample axis.
        method: Either ``"zscore"`` or ``"attention_mean"``. ``"zscore"``
            normalises activations by subtracting the mean and dividing by
            the standard deviation along the sample dimension before
            applying the selected reduction. ``"attention_mean"`` simply
            averages the activations along the sample dimension.
        sample_dim: Dimension that indexes independent samples.
        reduction: How to reduce the normalised activations. ``"mean"``
            averages along the sample dimension, ``"sum"`` sums along it,
            and ``"none"`` keeps the per-sample scores.
        eps: Small constant to avoid division by zero.

    Returns:
        A tensor of φᵢ scores. When ``reduction`` is not ``"none"`` the
        sample dimension is removed; otherwise, its size matches the input.
    """

    if not isinstance(activations, torch.Tensor):
        raise TypeError("activations must be a torch.Tensor")

    sample_dim = _canonical_dim(sample_dim, activations.ndim)

    if method == "zscore":
        mean = activations.mean(dim=sample_dim, keepdim=True)
        std = activations.std(dim=sample_dim, correction=0, keepdim=True)
        std = std.clamp_min(eps)
        normalised = (activations - mean) / std
    elif method == "attention_mean":
        normalised = activations
    else:
        raise ValueError(f"Unsupported phi computation method: {method}")

    if reduction == "mean":
        return normalised.mean(dim=sample_dim)
    if reduction == "sum":
        return normalised.sum(dim=sample_dim)
    if reduction == "none":
        return normalised
    raise ValueError(f"Unsupported reduction: {reduction}")


def _topk_mask(values: torch.Tensor, top_k: Optional[int], dim: int) -> torch.Tensor:
    """Return a mask that keeps the largest ``top_k`` entries along ``dim``."""
    if top_k is None or top_k <= 0 or values.size(dim) <= top_k:
        return torch.ones_like(values, dtype=torch.bool)
    top_values, top_indices = torch.topk(values, top_k, dim=dim)
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask.scatter_(dim, top_indices, True)
    return mask


def compute_ci(
    clean_logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    *,
    method: CiMethod = "logit_diff",
    targets: Optional[torch.Tensor] = None,
    token_dim: int = -1,
    top_k: Optional[int] = None,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute contextual importance scores cᵢ.

    Args:
        clean_logits: Logits (or logit contributions) produced with a
            neuron retained.
        corrupted_logits: Logits (or logit contributions) from the
            counterfactual forward pass where the neuron is ablated or
            otherwise corrupted.
        method: One of ``"lse"``, ``"target_logit"``, ``"logit_diff"``, or
            ``"kl"``.
        targets: Token indices for the reference output. Required for
            ``"target_logit"`` and ``"logit_diff"``.
        token_dim: Dimension indexing vocabulary tokens.
        top_k: Optional number of competitor tokens to consider for the
            ``"logit_diff"`` and ``"kl"`` branches. ``None`` keeps all
            tokens.
        temperature: Softmax temperature used for weighting competitors in
            the ``"logit_diff"`` and ``"kl"`` branches.
        eps: Numerical stability constant.

    Returns:
        Tensor of contextual importance scores aggregated along the token
        dimension specified by ``token_dim``.
    """

    if not isinstance(clean_logits, torch.Tensor) or not isinstance(
        corrupted_logits, torch.Tensor
    ):
        raise TypeError("clean_logits and corrupted_logits must be torch tensors")
    if clean_logits.shape != corrupted_logits.shape:
        raise ValueError("clean_logits and corrupted_logits must share the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    token_dim = _canonical_dim(token_dim, clean_logits.ndim)

    if method == "lse":
        clean_score = torch.logsumexp(clean_logits, dim=token_dim)
        corrupt_score = torch.logsumexp(corrupted_logits, dim=token_dim)
        return clean_score - corrupt_score

    diff = clean_logits - corrupted_logits

    if method == "target_logit":
        if targets is None:
            raise ValueError("targets must be provided for target_logit computation")
        expanded_targets = targets.unsqueeze(token_dim)
        return torch.gather(diff, token_dim, expanded_targets).squeeze(token_dim)

    if method == "logit_diff":
        if targets is None:
            raise ValueError("targets must be provided for logit_diff computation")
        expanded_targets = targets.unsqueeze(token_dim)
        target_score = torch.gather(diff, token_dim, expanded_targets).squeeze(token_dim)

        competitor_mask = torch.ones_like(diff, dtype=torch.bool)
        competitor_mask.scatter_(token_dim, expanded_targets, False)
        if top_k is not None:
            topk_mask = _topk_mask(clean_logits.masked_fill(~competitor_mask, float("-inf")), top_k, token_dim)
            competitor_mask &= topk_mask

        competitor_logits = clean_logits / temperature
        competitor_logits = competitor_logits.masked_fill(~competitor_mask, float("-inf"))
        weights = torch.softmax(competitor_logits, dim=token_dim)
        weights = torch.where(competitor_mask, weights, torch.zeros_like(weights))
        weights_sum = weights.sum(dim=token_dim, keepdim=True).clamp_min(eps)
        weights = weights / weights_sum
        competitor_score = (weights * diff).sum(dim=token_dim)
        return target_score - competitor_score

    if method == "kl":
        mask = None
        if top_k is not None:
            mask = _topk_mask(clean_logits, top_k, token_dim)
        scaled_clean = clean_logits / temperature
        scaled_corrupt = corrupted_logits / temperature
        if mask is not None:
            scaled_clean = scaled_clean.masked_fill(~mask, float("-inf"))
            scaled_corrupt = scaled_corrupt.masked_fill(~mask, float("-inf"))

        p = torch.softmax(scaled_clean, dim=token_dim)
        q = torch.softmax(scaled_corrupt, dim=token_dim)
        if mask is not None:
            p = torch.where(mask, p, torch.zeros_like(p))
            q = torch.where(mask, q, torch.zeros_like(q))
            p = p / p.sum(dim=token_dim, keepdim=True).clamp_min(eps)
            q = q / q.sum(dim=token_dim, keepdim=True).clamp_min(eps)

        return (p * (p.clamp_min(eps).log() - q.clamp_min(eps).log())).sum(dim=token_dim)

    raise ValueError(f"Unsupported c_i computation method: {method}")


def compute_kappa(
    phi_scores: torch.Tensor,
    ci_scores: torch.Tensor,
    *,
    combine: Literal["product", "geometric_mean"] = "product",
    descending: bool = True,
) -> Tuple[torch.Tensor, List[List[Tuple[int, float]]]]:
    """Combine φᵢ and cᵢ scores into κᵢ rankings per layer.

    Args:
        phi_scores: Tensor containing per-neuron φᵢ values. The first
            dimension is interpreted as the layer dimension.
        ci_scores: Tensor with the same broadcastable shape containing the
            contextual importance scores.
        combine: How to combine φᵢ and cᵢ. ``"product"`` multiplies the two
            tensors, while ``"geometric_mean"`` uses the square root of
            their product to reduce the influence of large magnitudes.
        descending: Whether to sort neurons in descending order of κᵢ.

    Returns:
        A tuple consisting of the κᵢ tensor and a nested list containing the
        sorted (neuron_index, κᵢ value) pairs for each layer.
    """

    phi_scores, ci_scores = torch.broadcast_tensors(phi_scores, ci_scores)
    if phi_scores.ndim < 2:
        raise ValueError("phi_scores must have at least two dimensions (layers, neurons)")
    if phi_scores.shape[0] <= 0:
        raise ValueError("phi_scores must contain at least one layer")

    if combine == "product":
        kappa = phi_scores * ci_scores
    elif combine == "geometric_mean":
        kappa = torch.sqrt((phi_scores * ci_scores).clamp_min(0.0))
    else:
        raise ValueError(f"Unsupported combine strategy: {combine}")

    layer_rankings: List[List[Tuple[int, float]]] = []
    flattened = kappa.reshape(kappa.shape[0], -1)
    for layer_idx, layer_values in enumerate(flattened):
        sorted_values, sorted_indices = torch.sort(layer_values, descending=descending)
        layer_rankings.append(
            [(int(index.item()), float(value.item())) for index, value in zip(sorted_indices, sorted_values)]
        )

    return kappa, layer_rankings

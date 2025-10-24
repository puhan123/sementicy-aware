"""Utilities for capturing and persisting token-wise activation statistics."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .tensor_utils import ensure_dense, maybe_to_sparse, standardize_activations


def _to_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if isinstance(dtype_str, torch.dtype):
        return dtype_str
    key = dtype_str.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported activation dtype '{dtype_str}'")
    return mapping[key]


@dataclass
class TokenActivationStats:
    layer_name: str
    cache_root: str
    save_dtype: torch.dtype
    use_sparse: bool
    eps: float = 1e-6
    chunk_files: List[str] = field(default_factory=list)
    running_mean: Optional[torch.Tensor] = None
    running_M2: Optional[torch.Tensor] = None
    count: int = 0
    hidden_size: Optional[int] = None
    chunk_index: int = 0

    def __post_init__(self) -> None:
        safe_name = self.layer_name.replace("/", "_")
        self.layer_dir = os.path.join(self.cache_root, safe_name)
        os.makedirs(self.layer_dir, exist_ok=True)

    def _update_running_stats(self, batch: torch.Tensor) -> None:
        if batch.numel() == 0:
            return
        batch = batch.to(torch.float64)
        batch_count = batch.shape[0]
        batch_mean = batch.mean(dim=0)
        centered = batch - batch_mean
        batch_M2 = (centered * centered).sum(dim=0)
        if self.running_mean is None:
            self.running_mean = batch_mean
            self.running_M2 = batch_M2
            self.count = batch_count
            return
        total_count = self.count + batch_count
        delta = batch_mean - self.running_mean
        self.running_mean = self.running_mean + delta * batch_count / total_count
        self.running_M2 = (
            self.running_M2
            + batch_M2
            + delta.pow(2) * self.count * batch_count / total_count
        )
        self.count = total_count

    def update(self, activations: torch.Tensor, context: Dict[str, torch.Tensor]) -> None:
        if self.hidden_size is None:
            self.hidden_size = activations.shape[-1]
        dense = activations.detach().to("cpu")
        if dense.dim() == 2:
            dense = dense.unsqueeze(1)
        elif dense.dim() == 1:
            dense = dense.view(1, 1, -1)
        batch_size, seq_len, hidden = dense.shape
        if hidden != self.hidden_size:
            raise ValueError(
                f"Mismatched hidden sizes for {self.layer_name}: {hidden} vs {self.hidden_size}"
            )

        token_ids = context["token_ids"].to("cpu")
        sample_ids = context["sample_ids"].to("cpu")
        token_positions = context["token_positions"].to("cpu")
        attention_mask = context.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to("cpu")

        seq_len = min(seq_len, token_ids.size(1))
        dense = dense[:, :seq_len, :]
        token_ids = token_ids[:, :seq_len]
        token_positions = token_positions[:, :seq_len]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_len]

        flat_dense = dense.reshape(-1, hidden)
        flat_tokens = token_ids.reshape(-1)
        flat_positions = token_positions.reshape(-1)
        flat_sample_ids = sample_ids.view(-1, 1).expand(-1, seq_len).reshape(-1)
        if attention_mask is not None:
            mask = attention_mask.reshape(-1) > 0
            flat_dense = flat_dense[mask]
            flat_tokens = flat_tokens[mask]
            flat_positions = flat_positions[mask]
            flat_sample_ids = flat_sample_ids[mask]

        if flat_dense.numel() == 0:
            return

        self._update_running_stats(flat_dense)

        save_tensor = flat_dense.to(self.save_dtype)
        if self.use_sparse:
            save_tensor = save_tensor.to_sparse()

        chunk_path = os.path.join(self.layer_dir, f"chunk_{self.chunk_index:06d}.pt")
        torch.save(
            {
                "sample_ids": flat_sample_ids,
                "token_ids": flat_tokens,
                "token_positions": flat_positions,
                "activations": save_tensor,
            },
            chunk_path,
        )
        self.chunk_files.append(chunk_path)
        self.chunk_index += 1

    def finalize(self) -> Dict[str, object]:
        if self.count == 0 or self.running_mean is None:
            return {
                "layer": self.layer_name,
                "stats": None,
                "chunks": self.chunk_files,
                "count": 0,
            }
        variance = self.running_M2 / max(self.count - 1, 1)
        std = torch.sqrt(variance.clamp_min(self.eps))
        mean = self.running_mean
        stats_path = os.path.join(self.layer_dir, "stats.pt")
        torch.save(
            {
                "mean": mean.to(self.save_dtype),
                "std": std.to(self.save_dtype),
                "count": self.count,
            },
            stats_path,
        )

        for chunk_path in self.chunk_files:
            payload = torch.load(chunk_path)
            activations = ensure_dense(payload["activations"]).to(torch.float32)
            standardized, _, _ = standardize_activations(
                activations,
                mean=mean.to(torch.float32),
                std=std.to(torch.float32),
                eps=self.eps,
                to_sparse=self.use_sparse,
            )
            payload["activations"] = maybe_to_sparse(
                activations.to(self.save_dtype), self.use_sparse
            )
            if standardized.is_sparse:
                payload["z_scores"] = standardized.coalesce().to(self.save_dtype)
            else:
                payload["z_scores"] = maybe_to_sparse(
                    standardized.to(self.save_dtype), self.use_sparse
                )
            torch.save(payload, chunk_path)

        return {
            "layer": self.layer_name,
            "stats_path": stats_path,
            "chunks": self.chunk_files,
            "count": self.count,
            "hidden_size": self.hidden_size,
            "dtype": str(self.save_dtype),
            "use_sparse": self.use_sparse,
        }


class ActivationCollector:
    def __init__(
        self,
        model: nn.Module,
        cache_root: str,
        dtype: torch.dtype,
        use_sparse: bool,
        eps: float = 1e-6,
    ) -> None:
        self.model = model
        self.cache_root = cache_root
        self.dtype = dtype
        self.use_sparse = use_sparse
        self.eps = eps
        self.current_context: Optional[Dict[str, torch.Tensor]] = None
        self.layer_stats: Dict[str, TokenActivationStats] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        os.makedirs(self.cache_root, exist_ok=True)
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                stats = TokenActivationStats(
                    layer_name=name,
                    cache_root=self.cache_root,
                    save_dtype=self.dtype,
                    use_sparse=self.use_sparse,
                    eps=self.eps,
                )
                self.layer_stats[name] = stats
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if self.current_context is None:
                return
            if isinstance(output, tuple):
                output = output[0]
            if output is None:
                return
            self.layer_stats[name].update(output, self.current_context)

        return hook

    def set_context(self, context: Dict[str, torch.Tensor]) -> None:
        self.current_context = context

    def clear_context(self) -> None:
        self.current_context = None

    def collect(self, dataloader: DataLoader, device: str) -> None:
        self.model.eval()
        processed = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                batch_size, seq_len = input_ids.shape
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                token_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
                sample_ids = torch.arange(processed, processed + batch_size)
                processed += batch_size
                context = {
                    "sample_ids": sample_ids,
                    "token_ids": batch["input_ids"],
                    "token_positions": token_positions,
                }
                if attention_mask is not None:
                    context["attention_mask"] = batch["attention_mask"]
                self.set_context(context)
                inputs = {"input_ids": input_ids}
                if attention_mask is not None:
                    inputs["attention_mask"] = attention_mask
                _ = self.model(**inputs)
                self.clear_context()

    def finalize(self) -> Dict[str, object]:
        metadata = {
            "layers": {},
            "dtype": str(self.dtype),
            "use_sparse": self.use_sparse,
        }
        for name, stats in self.layer_stats.items():
            metadata["layers"][name] = stats.finalize()
        self.remove_hooks()
        return metadata

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def collect_activations_to_cache(
    model: nn.Module,
    dataset,
    cache_root: str,
    batch_size: int,
    device: str,
    dtype: str,
    use_sparse: bool,
    eps: float = 1e-6,
) -> Dict[str, object]:
    collector = ActivationCollector(
        model=model,
        cache_root=cache_root,
        dtype=_to_dtype(dtype),
        use_sparse=use_sparse,
        eps=eps,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    collector.collect(dataloader=dataloader, device=device)
    metadata = collector.finalize()
    manifest_path = os.path.join(cache_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
    return metadata


__all__ = [
    "ActivationCollector",
    "collect_activations_to_cache",
    "TokenActivationStats",
]

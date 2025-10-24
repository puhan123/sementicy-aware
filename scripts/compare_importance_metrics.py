#!/usr/bin/env python3
"""Compare classical importance metrics with κᵢ rankings on a toy batch.

This demonstration script loads a handful of GSM8K calibration samples,
constructs reproducible synthetic activations/logit deltas, and contrasts the
resulting φᵢ-only ordering with κᵢ rankings obtained from the combined
φᵢ·cᵢ scores.  The goal is to show how alternative contextual importance
metrics influence neuron prioritisation without depending on heavyweight model
checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sys

import torch

# Allow running the script directly via ``python scripts/...``.
REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.neuron_metrics import compute_ci, compute_kappa, compute_phi

DEFAULT_DATASET = Path("datasets_directory/GSM8k/data/gsm8k_train.jsonl")
DEFAULT_OUTPUT_DIR = Path("experiments/importance_demo")


def _load_samples(path: Path, limit: int) -> List[Dict[str, str]]:
    """Read ``limit`` JSONL entries from ``path``."""
    samples: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(samples) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples


def _hashed_bow(text: str, size: int) -> torch.Tensor:
    """Convert ``text`` into a deterministic hashed bag-of-words vector."""
    vector = torch.zeros(size, dtype=torch.float32)
    for token in text.lower().split():
        vector[hash(token) % size] += 1.0
    return vector


def _prepare_tensors(
    samples: Iterable[Dict[str, str]],
    *,
    n_layers: int,
    n_neurons: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct synthetic activations and logits for the provided samples."""

    question_features: List[torch.Tensor] = []
    answer_features: List[torch.Tensor] = []
    for example in samples:
        question_features.append(_hashed_bow(example["question"], n_neurons))
        answer_features.append(_hashed_bow(example["answer"], n_neurons))

    base_features = torch.stack(question_features) + 0.5 * torch.stack(answer_features)
    layer_offsets = torch.linspace(0.1, 1.0, n_layers).view(n_layers, 1, 1)
    neuron_scales = torch.linspace(0.8, 1.2, n_neurons).view(1, 1, n_neurons)

    # Shape: (layers, samples, neurons)
    activations = base_features.unsqueeze(0) * neuron_scales
    activations = activations * layer_offsets + torch.sin(activations * 0.15)

    vocab_positions = torch.linspace(-1.5, 1.5, vocab_size).view(1, 1, 1, vocab_size)
    clean_logits = activations.unsqueeze(-1) * vocab_positions
    corrupted_logits = clean_logits - 0.4 * activations.unsqueeze(-1) * vocab_positions / layer_offsets

    target_indices = torch.tensor(
        [abs(hash(example["answer"])) % vocab_size for example in samples],
        dtype=torch.long,
    )
    targets = target_indices.view(1, -1, 1).expand(n_layers, -1, n_neurons)
    return activations, clean_logits, corrupted_logits, targets


def _format_topk(values: torch.Tensor, k: int) -> List[Tuple[int, float]]:
    top_values, top_indices = torch.topk(values, k)
    return [(idx.item(), score.item()) for score, idx in zip(top_values, top_indices)]


def run_demo(args: argparse.Namespace) -> Dict[str, object]:
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(dataset_path, args.num_samples)

    (
        activations,
        clean_logits,
        corrupted_logits,
        targets,
    ) = _prepare_tensors(
        samples,
        n_layers=args.n_layers,
        n_neurons=args.n_neurons,
        vocab_size=args.vocab_size,
    )

    phi_scores = compute_phi(activations, method=args.phi_method, sample_dim=1, reduction="mean")
    ci_scores = compute_ci(
        clean_logits,
        corrupted_logits,
        method=args.ci_method,
        targets=targets,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    if ci_scores.ndim > 2:
        ci_scores = ci_scores.mean(dim=1)
    kappa_scores, kappa_rankings = compute_kappa(phi_scores, ci_scores, combine=args.combine)

    summary: Dict[str, object] = {
        "config": {
            "dataset_path": str(dataset_path),
            "num_samples": args.num_samples,
            "n_layers": args.n_layers,
            "n_neurons": args.n_neurons,
            "vocab_size": args.vocab_size,
            "phi_method": args.phi_method,
            "ci_method": args.ci_method,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "combine": args.combine,
        },
        "phi_rankings": {},
        "kappa_rankings": {},
        "target_logit_deltas": {},
    }

    logit_delta = (clean_logits - corrupted_logits)
    for layer_idx in range(args.n_layers):
        phi_layer = phi_scores[layer_idx]
        kappa_layer = kappa_scores[layer_idx]
        summary["phi_rankings"][f"layer_{layer_idx}"] = _format_topk(phi_layer, args.top_k_report)
        summary["kappa_rankings"][f"layer_{layer_idx}"] = kappa_rankings[layer_idx][: args.top_k_report]

        sample_logit_deltas = []
        for sample_idx in range(args.num_samples):
            target_id = targets[layer_idx, sample_idx, 0].item()
            neuron_changes = logit_delta[layer_idx, sample_idx, :, target_id]
            top_neuron_changes = _format_topk(neuron_changes, args.top_k_report)
            sample_logit_deltas.append(
                {
                    "sample_index": sample_idx,
                    "target_token": target_id,
                    "top_neuron_deltas": top_neuron_changes,
                }
            )
        summary["target_logit_deltas"][f"layer_{layer_idx}"] = sample_logit_deltas

        print(f"Layer {layer_idx}")
        print("  φᵢ top neurons:")
        for idx, score in summary["phi_rankings"][f"layer_{layer_idx}"]:
            print(f"    neuron {idx:2d}: {score: .4f}")
        print("  κᵢ top neurons:")
        for idx, score in summary["kappa_rankings"][f"layer_{layer_idx}"]:
            print(f"    neuron {idx:2d}: {score: .4f}")
        print()

    output_path = output_dir / "importance_metrics_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary to {output_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET, type=str)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=str)
    parser.add_argument("--num-samples", default=4, type=int)
    parser.add_argument("--n-layers", default=3, type=int)
    parser.add_argument("--n-neurons", default=12, type=int)
    parser.add_argument("--vocab-size", default=32, type=int)
    parser.add_argument("--phi-method", default="attention_mean", choices=["zscore", "attention_mean"])
    parser.add_argument("--ci-method", default="logit_diff", choices=["lse", "target_logit", "logit_diff", "kl"])
    parser.add_argument("--top-k", default=16, type=int, help="Competitor tokens kept for contextual metrics")
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument(
        "--combine",
        default="product",
        choices=["product", "geometric_mean"],
        help="κᵢ combination rule",
    )
    parser.add_argument("--top-k-report", default=5, type=int, help="How many neurons to include in the printed summary")
    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(0)
    run_demo(parse_args())

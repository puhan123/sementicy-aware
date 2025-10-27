import json
import os
from collections import defaultdict
import math
from statistics import mean, median
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from matplotlib import pyplot as plt
import numpy as np
import torch

from datasets_directory.MMLU.MMLU_utils import MMLU_N_Shot_Dataset

from datasets_directory.MMLU import categories

from datasets_directory.GSM8k.GSM8k_utils import GSM8k_N_Shot_Dataset

from datasets_directory.Spider.Spider_utils import Spider_N_Shot_Dataset

from datasets_directory.PretrainDatasets.wikitext2_utils import Wikitext2_Dataset

from datasets_directory.PretrainDatasets.c4_utils import C4_New_Dataset

KAPPA_SCORING_METHODS = {"mean_abs", "sum_abs", "l2"}
KAPPA_CI_TYPES = {"none", "bootstrap"}
KAPPA_MODULE_REDUCTIONS = {"mean", "max", "median"}

"""Dataset Loading"""
def preprocess_calibration_datasets(args, tokenizer, indices_for_choices, n_calibration_points=128, seqlen=2048):
    """Args must contain these attributes: dataset, model, logger, serial_number, ntrain, eval_start_p, train_end_p, """
    tokenizer.padding_side = "left"  # VERY IMPORTANT in order to properly mask loss.
    tokenizer.pad_token = tokenizer.eos_token
    if args.dataset == "MMLU_MCQA":
        subjects_to_use = "all"
        train_dataset = MMLU_N_Shot_Dataset(model_name=args.model, tokenizer=tokenizer, use_train_split=True, ntrain=args.ntrain, eval_start_p=args.eval_start_p, train_end_p=args.train_end_p, verbose=False, subjects_to_use=subjects_to_use)
    elif args.dataset == "MMLU_STEM":
        subjects_to_use = categories.categories_to_subjects["STEM"]
        train_dataset = MMLU_N_Shot_Dataset(model_name=args.model, tokenizer=tokenizer, use_train_split=True, ntrain=args.ntrain, eval_start_p=args.eval_start_p, train_end_p=args.train_end_p, verbose=False, subjects_to_use=subjects_to_use)
    elif args.dataset == "MMLU_social_sciences":
        subjects_to_use = categories.categories_to_subjects["social sciences"]
        train_dataset = MMLU_N_Shot_Dataset(model_name=args.model, tokenizer=tokenizer, use_train_split=True, ntrain=args.ntrain, eval_start_p=args.eval_start_p, train_end_p=args.train_end_p, verbose=False, subjects_to_use=subjects_to_use)
    elif args.dataset == "MMLU_humanities":
        subjects_to_use = categories.categories_to_subjects["humanities"]
        train_dataset = MMLU_N_Shot_Dataset(model_name=args.model, tokenizer=tokenizer, use_train_split=True, ntrain=args.ntrain, eval_start_p=args.eval_start_p, train_end_p=args.train_end_p, verbose=False, subjects_to_use=subjects_to_use)
    elif args.dataset == "GSM8k":
        train_dataset = GSM8k_N_Shot_Dataset(model_name=args.model, tokenizer=tokenizer, \
                use_train_split=True, n_shot=8, cot_flag=True, make_data_wrong=False
                )
    elif args.dataset == "Spider":
        train_dataset = Spider_N_Shot_Dataset(model_name=args.model, tokenizer=tokenizer, \
                use_train_split=True, make_data_wrong=False
                )
    elif args.dataset == "wikitext2":
        train_dataset = Wikitext2_Dataset(seed=args.serial_number, use_train_split=True, tokenizer_name=tokenizer.name_or_path, verbose=False, nsamples=n_calibration_points, seqlen=seqlen, device="cuda")
    elif args.dataset == "c4":
        raise ValueError(f"Dataset c4 has been deprecated, use c4_new")
    elif args.dataset == "c4_new":
        train_dataset = C4_New_Dataset(seed=args.serial_number, use_train_split=True, tokenizer_name=tokenizer.name_or_path, verbose=False, nsamples=n_calibration_points, seqlen=seqlen, device="cuda")
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    train_dataset.shuffle(seed=args.serial_number)
    train_dataset.truncate_to_seqlen_n_samples(seqlen, n_calibration_points)
    return train_dataset

"""Saving"""
def save_accumulated_importances(args, accumulated_gradient, save_full_gradients=False, file_name="per_module_saved_importances", save_path=None, dtype=torch.float16):
    """Save an importance dictionary where the keys are strings and values are pytorch tensors in dtype."""
    # Save accumulated_gradient
    if save_full_gradients:
        if not save_path:
            save_path = os.path.join(args.results_dir, args.run_name, f"{file_name}_{args.serial_number}.pt")
        with open(save_path, "wb") as f:
            torch.cuda.empty_cache()
            save_accumulated_gradient = {}
            for i in accumulated_gradient:
                save_accumulated_gradient[i] = torch.clone(accumulated_gradient[i].to(dtype))
            for key, value in save_accumulated_gradient.items():
                if isinstance(value, torch.nn.Parameter):
                    save_accumulated_gradient[key] = value.data
            torch.save(save_accumulated_gradient, f)

    # Save per module gradients
    else:
        if not save_path:
            save_path = os.path.join(args.results_dir, args.run_name, f"{file_name}_{args.serial_number}.json")
        with open(save_path, "w") as f:
            torch.cuda.empty_cache()
            save_accumulated_gradient = {}
            for i in accumulated_gradient:
                save_accumulated_gradient[i] = torch.mean(torch.abs(torch.clone(accumulated_gradient[i].detach().to(dtype).to("cpu")))).item()
            json.dump(save_accumulated_gradient, f)

"""Filtering"""
def filter_importances_dict(importances, configuration="mlp_atten_only"):
    if "mlp_atten_only":
        importances = {k:v for k, v in importances.items() if ".mlp." in k or ".self_attn." in k}
    elif "linear_only":
        raise Exception("Not implemented")
    return importances


def _get_kappa_scoring_function(scoring_method: str) -> Callable[[torch.Tensor], torch.Tensor]:
    scoring_method = scoring_method.lower()
    if scoring_method == "mean_abs":
        return lambda weights: weights.abs().mean()
    if scoring_method == "sum_abs":
        return lambda weights: weights.abs().sum()
    if scoring_method == "l2":
        return lambda weights: torch.linalg.vector_norm(weights, ord=2)
    raise ValueError(f"Unsupported scoring method: {scoring_method}. Expected one of {sorted(KAPPA_SCORING_METHODS)}")


def _bootstrap_confidence_interval(
    values: torch.Tensor,
    scoring_fn: Callable[[torch.Tensor], torch.Tensor],
    alpha: float,
    num_samples: int,
    random_state: Optional[np.random.Generator],
) -> Optional[Dict[str, float]]:
    flat_values = values.detach().cpu().reshape(-1)
    if flat_values.numel() == 0:
        return None
    rng = random_state or np.random.default_rng()
    bootstrap_scores = []
    values_np = flat_values.numpy()
    for _ in range(num_samples):
        resampled = rng.choice(values_np, size=values_np.shape[0], replace=True)
        resampled_tensor = torch.from_numpy(resampled).to(flat_values.dtype)
        bootstrap_scores.append(float(scoring_fn(resampled_tensor)))
    lower_q = alpha / 2
    upper_q = 1 - lower_q
    low = float(np.quantile(bootstrap_scores, lower_q))
    high = float(np.quantile(bootstrap_scores, upper_q))
    return {"type": "bootstrap", "low": low, "high": high}


def compute_kappa_scores(
    importances: Dict[str, torch.Tensor],
    scoring_method: str = "mean_abs",
    ci_type: str = "none",
    ci_alpha: float = 0.05,
    ci_samples: int = 1000,
    ci_seed: Optional[int] = None,
) -> List[Dict[str, object]]:
    if scoring_method.lower() not in KAPPA_SCORING_METHODS:
        raise ValueError(f"Unsupported scoring method: {scoring_method}. Expected one of {sorted(KAPPA_SCORING_METHODS)}")
    if ci_type.lower() not in KAPPA_CI_TYPES:
        raise ValueError(f"Unsupported ci_type: {ci_type}. Expected one of {sorted(KAPPA_CI_TYPES)}")

    scoring_fn = _get_kappa_scoring_function(scoring_method)
    rng = None
    if ci_type.lower() == "bootstrap":
        if not 0 < ci_alpha < 1:
            raise ValueError("ci_alpha must be between 0 and 1 when using bootstrap confidence intervals")
        if ci_samples <= 0:
            raise ValueError("ci_samples must be a positive integer when using bootstrap confidence intervals")
        if ci_seed is not None:
            rng = np.random.default_rng(ci_seed)

    scores: List[Dict[str, object]] = []
    for module_name, tensor in importances.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim < 2:
            # Skip tensors that cannot be interpreted as neuron weight matrices
            continue
        flattened = tensor.detach().to(torch.float32).cpu().reshape(tensor.shape[0], -1)
        for neuron_idx in range(flattened.shape[0]):
            neuron_weights = flattened[neuron_idx]
            score_value = float(scoring_fn(neuron_weights))
            ci: Optional[Dict[str, float]] = None
            if ci_type.lower() == "bootstrap":
                ci = _bootstrap_confidence_interval(
                    neuron_weights,
                    scoring_fn=scoring_fn,
                    alpha=ci_alpha,
                    num_samples=ci_samples,
                    random_state=rng,
                )
            scores.append(
                {
                    "module": module_name,
                    "neuron_index": int(neuron_idx),
                    "score": score_value,
                    "ci": ci,
                    "scoring_method": scoring_method,
                }
            )
    scores.sort(key=lambda entry: entry["score"], reverse=True)
    return scores


def save_kappa_scores(
    args,
    kappa_scores: List[Dict[str, object]],
    save_path: Optional[str] = None,
) -> str:
    if not save_path:
        save_path = os.path.join(args.results_dir, args.run_name, f"kappa_scores_{args.serial_number}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    serializable_scores = []
    for entry in kappa_scores:
        serialized_entry = {
            "module": entry["module"],
            "neuron_index": entry["neuron_index"],
            "score": float(entry["score"]),
            "scoring_method": entry.get("scoring_method"),
        }
        ci = entry.get("ci")
        if ci is not None:
            serialized_entry["ci"] = {
                "type": ci.get("type"),
                "low": float(ci.get("low")),
                "high": float(ci.get("high")),
            }
        serializable_scores.append(serialized_entry)
    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(serializable_scores, fp, indent=2)
    return save_path


def load_kappa_scores_from_json(path: str) -> List[Dict[str, object]]:
    """Load previously persisted κᵢ scores from ``path``.

    The loader performs minimal validation to ensure that the JSON payload is
    a list of mapping objects containing at least ``module`` and ``score``
    fields.  Entries missing either attribute are silently discarded to keep
    the downstream ranking logic resilient to partial writes.
    """

    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, list):
        raise ValueError("kappa score payload must be a JSON list")
    normalized_entries: List[Dict[str, object]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        module = entry.get("module")
        score = entry.get("score")
        if module is None or score is None:
            continue
        normalized_entries.append({"module": str(module), "score": float(score)})
    return normalized_entries


def _get_module_aggregator(reduction: str) -> Callable[[Sequence[float]], float]:
    reduction = reduction.lower()
    if reduction == "mean":
        return mean
    if reduction == "max":
        return max
    if reduction == "median":
        return median
    raise ValueError(
        f"Unsupported module reduction: {reduction}. Expected one of {sorted(KAPPA_MODULE_REDUCTIONS)}"
    )


def aggregate_module_kappa_scores(
    kappa_scores: Iterable[Dict[str, object]],
    reduction: str = "mean",
) -> List[Tuple[str, float]]:
    """Aggregate neuron-level κᵢ scores into per-module statistics."""

    if reduction.lower() not in KAPPA_MODULE_REDUCTIONS:
        raise ValueError(
            f"Unsupported module reduction: {reduction}. Expected one of {sorted(KAPPA_MODULE_REDUCTIONS)}"
        )
    aggregator = _get_module_aggregator(reduction)
    module_buckets: Dict[str, List[float]] = defaultdict(list)
    for entry in kappa_scores:
        module = entry.get("module")
        score = entry.get("score")
        if module is None or score is None:
            continue
        module_buckets[str(module)].append(float(score))

    ranked_modules: List[Tuple[str, float]] = []
    for module_name, scores in module_buckets.items():
        if not scores:
            continue
        ranked_modules.append((module_name, float(aggregator(scores))))
    ranked_modules.sort(key=lambda item: item[1], reverse=True)
    return ranked_modules


def select_top_modules_by_kappa(
    kappa_scores: Iterable[Dict[str, object]],
    fraction: float,
    reduction: str = "mean",
) -> List[str]:
    """Select the top fraction of modules based on aggregated κᵢ scores."""

    try:
        fraction_value = float(fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("fraction must be a real number") from exc
    if math.isnan(fraction_value):
        raise ValueError("fraction must be a real number")
    fraction_value = max(0.0, min(1.0, fraction_value))
    aggregated = aggregate_module_kappa_scores(kappa_scores, reduction=reduction)
    if not aggregated or fraction_value <= 0:
        return []
    top_count = max(1, math.ceil(len(aggregated) * fraction_value))
    top_count = min(top_count, len(aggregated))
    return [module for module, _ in aggregated[:top_count]]

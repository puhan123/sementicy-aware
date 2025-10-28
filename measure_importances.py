# Default
import gc
import os
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import yaml
from dotenv import load_dotenv
import datetime
load_dotenv()
# ML / Data
import numpy as np
import torch
import torch.nn as nn
import huggingface_hub
# token = os.getenv('HUGGINGFACE_TOKEN')
# huggingface_hub.login(token=token)
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from utils.model_utils import load_model
from utils.gradient_attributors import grad_attributor, sample_abs, weight_prod_contrastive_postprocess
from utils.measurement_utils import filter_importances_dict, preprocess_calibration_datasets, save_accumulated_importances
from utils.neuron_metrics import compute_ci, compute_kappa, compute_phi


def _build_model_inputs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Prepare the minimal set of inputs expected by causal LM models."""

    required_keys = {"input_ids"}
    optional_keys = {"attention_mask", "position_ids", "token_type_ids"}
    model_inputs: Dict[str, torch.Tensor] = {}

    missing = required_keys - batch.keys()
    if missing:
        raise KeyError(f"Missing required model inputs: {missing}")

    for key in required_keys | optional_keys:
        if key in batch:
            model_inputs[key] = batch[key]
    return model_inputs


def _register_activation_hooks(
    model: torch.nn.Module,
) -> Tuple[Dict[str, List[torch.Tensor]], List[RemovableHandle]]:
    """Attach hooks to all linear submodules within MLP or attention blocks."""

    activations: Dict[str, List[torch.Tensor]] = defaultdict(list)
    handles: List[RemovableHandle] = []

    def make_hook(name: str):
        def hook(_: torch.nn.Module, __, output: torch.Tensor):
            if isinstance(output, tuple):
                output = output[0]
            if not torch.is_tensor(output):
                return
            act = output.detach().to("cpu")
            if act.ndim == 1:
                act = act.unsqueeze(0)
            elif act.ndim >= 3:
                # Average across the sequence dimension to obtain per-neuron scores.
                act = act.mean(dim=-2)
            act = act.reshape(act.shape[0], -1).to(torch.float32)
            activations[name].append(act)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (".mlp." in name or ".self_attn." in name):
            handles.append(module.register_forward_hook(make_hook(name)))

    return activations, handles


def _is_accelerate_dispatched(model: torch.nn.Module) -> bool:
    """Return True when the model relies on Accelerate hooks for device placement."""

    hook = getattr(model, "_hf_hook", None)
    if hook is None:
        return False
    try:
        from accelerate.hooks import BaseHook  # type: ignore

        return isinstance(hook, BaseHook)
    except Exception:
        # If accelerate is unavailable, fall back to checking the presence of the hook.
        return hook is not None


def _coerce_device(device_like: str) -> torch.device:
    if device_like == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(device_like)
    except (TypeError, ValueError):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_execution_device(model: torch.nn.Module, requested_device: str) -> torch.device:
    candidates: List[torch.device] = []

    device_attr = getattr(model, "device", None)
    if isinstance(device_attr, torch.device):
        candidates.append(device_attr)
    elif isinstance(device_attr, str):
        try:
            candidates.append(torch.device(device_attr))
        except (TypeError, ValueError):
            pass

    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict) and hf_device_map:
        first_device = next(iter(hf_device_map.values()))
        try:
            candidates.append(torch.device(first_device))
        except (TypeError, ValueError):
            pass

    for device in candidates:
        if device.type != "meta" and device.type == "cuda":
            return device

    for device in candidates:
        if device.type != "meta":
            return device

    return _coerce_device(requested_device)


def _ensure_model_on_device(model: torch.nn.Module, device: str) -> torch.nn.Module:
    if _is_accelerate_dispatched(model):
        return model
    target = _coerce_device(device)
    return model.to(target)


def _compute_phi_ci_kappa_importances(args, clean_model: torch.nn.Module, dataset) -> Dict[str, torch.Tensor]:
    """Compute neuron importances via φ, cᵢ, and κ statistics."""

    if not args.corrupt_model:
        raise ValueError("phi_ci_kappa selector requires --corrupt_model to be provided")

    clean_parameter_shapes: Dict[str, torch.Size] = {}
    for name, param in clean_model.named_parameters():
        if not name.endswith("weight"):
            continue
        if ".mlp." not in name and ".self_attn." not in name:
            continue
        clean_parameter_shapes[name] = param.shape

    clean_model = _ensure_model_on_device(clean_model, args.device)
    clean_model.eval()

    clean_execution_device = _resolve_execution_device(clean_model, args.device)

    activation_storage, handles = _register_activation_hooks(clean_model)

    clean_logits: List[torch.Tensor] = []
    corrupt_logits: List[torch.Tensor] = []
    target_tokens: List[torch.Tensor] = []
    attention_masks: List[torch.Tensor] = []

    clean_dataloader = DataLoader(dataset, batch_size=args.dataloader_batch_size, shuffle=False)

    with torch.no_grad():
        for batch in clean_dataloader:
            batch = {
                key: value.to(clean_execution_device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            model_inputs = _build_model_inputs(batch)

            clean_outputs = clean_model(**model_inputs)

            clean_shifted = clean_outputs.logits[..., :-1, :].contiguous().detach().cpu()
            targets = model_inputs["input_ids"][..., 1:].contiguous().detach().cpu()

            clean_logits.append(clean_shifted)
            target_tokens.append(targets)

            if "attention_mask" in model_inputs:
                shifted_mask = model_inputs["attention_mask"][..., 1:].contiguous().detach().cpu()
                attention_masks.append(shifted_mask)

    for handle in handles:
        handle.remove()

    del clean_dataloader

    clean_is_accelerate = _is_accelerate_dispatched(clean_model)
    if not clean_is_accelerate:
        clean_model = clean_model.to("cpu")

    del clean_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    corrupt_model = load_model(
        args.corrupt_model,
        checkpoints_dir=args.checkpoints_dir,
        full_32_precision=False,
        brainfloat=False,
    )["model"]

    corrupt_model = _ensure_model_on_device(corrupt_model, args.device)
    corrupt_model.eval()

    corrupt_execution_device = _resolve_execution_device(corrupt_model, args.device)

    corrupt_dataloader = DataLoader(dataset, batch_size=args.dataloader_batch_size, shuffle=False)

    with torch.no_grad():
        for batch in corrupt_dataloader:
            batch = {
                key: value.to(corrupt_execution_device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            model_inputs = _build_model_inputs(batch)

            corrupt_outputs = corrupt_model(**model_inputs)

            corrupt_shifted = corrupt_outputs.logits[..., :-1, :].contiguous().detach().cpu()

            corrupt_logits.append(corrupt_shifted)

    clean_concat = torch.cat(clean_logits, dim=0) if clean_logits else torch.empty(0)
    corrupt_concat = torch.cat(corrupt_logits, dim=0) if corrupt_logits else torch.empty(0)
    targets_concat = torch.cat(target_tokens, dim=0) if target_tokens else torch.empty(0, dtype=torch.long)

    if clean_concat.numel() == 0:
        raise RuntimeError("No logits were captured during phi_ci_kappa computation")

    ci_scores = compute_ci(
        clean_concat,
        corrupt_concat,
        method=args.ci_method,
        targets=targets_concat,
        token_dim=-1,
        top_k=args.ci_top_k,
        temperature=args.ci_temperature,
    )

    if attention_masks:
        mask_concat = torch.cat(attention_masks, dim=0).to(ci_scores.dtype)
        token_counts = mask_concat.sum(dim=-1).clamp_min(1.0)
        ci_per_sample = (ci_scores * mask_concat).sum(dim=-1) / token_counts
    else:
        ci_per_sample = ci_scores.mean(dim=-1)

    importances: Dict[str, torch.Tensor] = {}
    ci_expanded_cache: Dict[int, torch.Tensor] = {}

    for module_name, activations in activation_storage.items():
        if not activations:
            continue
        stacked = torch.cat(activations, dim=0)
        phi_values = compute_phi(stacked, sample_dim=0, reduction="none")
        if phi_values.shape[0] != ci_per_sample.shape[0]:
            raise ValueError(
                f"Mismatch between phi samples ({phi_values.shape[0]}) and ci samples ({ci_per_sample.shape[0]})"
            )

        feature_dim = phi_values.shape[1]
        if feature_dim not in ci_expanded_cache:
            ci_expanded_cache[feature_dim] = ci_per_sample.view(-1, 1).expand(-1, feature_dim)
        ci_tensor = ci_expanded_cache[feature_dim]

        kappa_tensor, _ = compute_kappa(phi_values, ci_tensor, combine=args.kappa_combine)
        # Average across samples to obtain a single importance score per output neuron.
        importances[module_name] = kappa_tensor.mean(dim=0)

    del corrupt_model, corrupt_dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    clean_importances: Dict[str, torch.Tensor] = {}
    for name, shape in clean_parameter_shapes.items():
        module_name = name.rsplit(".", 1)[0]
        if module_name not in importances:
            continue
        layer_scores = importances[module_name].to(torch.float32)
        if layer_scores.numel() != shape[0]:
            raise ValueError(
                f"Kappa scores for {module_name} have shape {layer_scores.shape}, expected {shape[0]} entries"
            )
        view_shape = (layer_scores.shape[0],) + (1,) * (len(shape) - 1)
        expanded = layer_scores.view(view_shape).expand(shape).contiguous().clone()
        clean_importances[name] = expanded

    return clean_importances

def main(args):
    print(f"{datetime.datetime.now()=}")
    # Save args
    args_dict = vars(args)  # Convert Namespace to dictionary
    with open(os.path.join(args.results_dir, args.run_name, 'args.yaml'), 'w' if args.override_args_yaml else 'a') as f:
        yaml.dump(args_dict, f, default_flow_style=False)

    # If args.save_importances_pt_path already exists, return
    if os.path.exists(args.save_importances_pt_path) and not args.force_recompute:
        print(f"Importances already exists at {args.save_importances_pt_path}")
        return {"run_name": args.run_name}

    print("Capturing Importances")
    if args.gradient_dtype == "bfloat16":
        full_32_precision = False
        brainfloat = True
    elif args.gradient_dtype == "float32":
        full_32_precision = True
        brainfloat = False
    elif args.gradient_dtype == "float16":
        full_32_precision = False
        brainfloat = False
    model_info = load_model(args.model, checkpoints_dir=args.checkpoints_dir, full_32_precision=full_32_precision, brainfloat=brainfloat)
    model, tokenizer = model_info["model"], model_info["tokenizer"]
    dataset = preprocess_calibration_datasets(args, tokenizer=tokenizer, indices_for_choices=None, n_calibration_points=args.n_calibration_points)
    importances = None
    if args.selector_type == "sample_abs_weight_prod_contrastive":
        del model, model_info
        importances = grad_attributor(args, args.model, args.corrupt_model, dataset, checkpoints_dir=args.checkpoints_dir, attributor_function=sample_abs, postprocess_function=weight_prod_contrastive_postprocess)
    elif args.selector_type == "sample_abs_weight_prod_contrastive_sm16bit":
        del model, model_info
        importances = grad_attributor(args, args.model, args.corrupt_model, dataset, checkpoints_dir=args.checkpoints_dir, attributor_function=sample_abs, postprocess_function=weight_prod_contrastive_postprocess, record_memory_history=False, backward_in_full_32_precision=False)
    elif args.selector_type == "phi_ci_kappa":
        importances = _compute_phi_ci_kappa_importances(args, model, dataset)
    else:
        raise Exception(f"Selector type {args.selector_type} not supported")
    importances = filter_importances_dict(importances, configuration="mlp_atten_only")
    save_accumulated_importances(args, accumulated_gradient=importances, save_full_gradients=args.save_full_gradients, save_path=args.save_importances_pt_path, dtype = torch.float16 if args.save_in_float16 else torch.float32)
    print(f"{datetime.datetime.now()=}")
    return {"run_name": args.run_name}

# Obtain command line arguments, call main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--serial_number", type=int, default=0, required=True)
    parser.add_argument("--save_importances_pt_path", type=str, default=None, required=True)
    parser.add_argument("--dataset", type=str, default="MMLU", required=True) 
    parser.add_argument("--selector_type", type=str, default="grad", required=True)
    parser.add_argument("--model", type=str, default="Meta-Llama-3-8B", required=True) 
    # Optional arguments
    parser.add_argument("--save_full_gradients", action="store_true")
    parser.add_argument("--corrupt_model", type=str, default=None, help="The name for the corrupt model")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--plot_importances", action="store_true", help="Flag for debugging purposes, behavior has been deprecated.")
    parser.add_argument("--override_args_yaml", action="store_true")
    parser.add_argument("--gradient_dtype", type=str, default="float32")
    parser.add_argument("--save_in_float16", action="store_true")  # For debugging and replication only
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_start_p", type=float, default=.75)
    parser.add_argument("--train_end_p", type=float, default=.75)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--n_calibration_points", type=int, default=128)
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--ci_method", type=str, default="logit_diff")
    parser.add_argument("--ci_top_k", type=int, default=None)
    parser.add_argument("--ci_temperature", type=float, default=1.0)
    parser.add_argument("--kappa_combine", type=str, default="product")
    parser.add_argument("--dataloader_batch_size", type=int, default=1)
    args = parser.parse_args()
    args.unsupervised = True
    random.seed(int(args.serial_number))
    np.random.seed(int(args.serial_number))
    torch.manual_seed(int(args.serial_number))
    torch.cuda.manual_seed(int(args.serial_number))

    os.makedirs(os.path.join(args.results_dir, args.run_name), exist_ok=True)
    if args.testing:
        args.n_calibration_points = 3
    main(args)
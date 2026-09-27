"""Diagnostics for controlled activation-initialization perturbations.

The helpers in this file deliberately perturb *only the initial activation
quantizer scale*.  They do not leave a clipping hook enabled during BRECQ or
sampling, so an observed difference can be attributed to the finite-budget
reconstruction trajectory rather than to a changed reconstruction objective.
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ddim.functions.denoising import compute_alpha
from qdiff.quant_layer import UniformAffineQuantizer

logger = logging.getLogger(__name__)


def _matching_stat(name: str, max_avg_stats: dict) -> Optional[float]:
    """Return the max-avg clipping magnitude belonging to a quantizer name."""
    for layer_name, value in max_avg_stats.items():
        if layer_name in name:
            return value
    return None


def iter_activation_quantizers(qnn: nn.Module) -> Iterable[Tuple[str, UniformAffineQuantizer]]:
    """Yield learnable activation quantizers, excluding fixed weight quantizers."""
    for name, module in qnn.named_modules():
        if isinstance(module, UniformAffineQuantizer) and getattr(module, "leaf_param", False):
            yield name, module


def snapshot_activation_scales(qnn: nn.Module) -> Dict[str, dict]:
    """Capture activation deltas and their clipping magnitudes on CPU."""
    out = {}
    for name, q in iter_activation_quantizers(qnn):
        if q.delta is None:
            continue
        delta = q.delta.detach().float().cpu().clone()
        out[name] = {
            "delta": delta,
            "zero_point": None if q.zero_point is None else torch.as_tensor(q.zero_point).detach().float().cpu().clone(),
            "clip": delta * float(q.n_levels),
            "symmetric": bool(q.sym),
        }
    return out


def initialize_tmma_interpolation(
    qnn: nn.Module,
    max_avg_stats: dict,
    rho: float,
) -> Tuple[Dict[str, dict], dict]:
    """Interpolate Min--Max and TMMA initial scales in-place.

    The ordinary first forward of ``qnn`` must have run before this function,
    so that its current deltas are the exact Min--Max initialization used by
    the baseline.  For matched symmetric activation quantizers, TMMA uses the
    saved group-wise max-avg magnitude divided by the integer range.  The
    applied scale is ``delta_tmma + rho * (delta_minmax - delta_tmma)``.
    """
    minmax = snapshot_activation_scales(qnn)
    tmma = {name: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in item.items()}
            for name, item in minmax.items()}
    changed, sq_ref, sq_delta = 0, 0.0, 0.0

    stats_by_name = {name: value for name, value in max_avg_stats.items()}
    for name, q in iter_activation_quantizers(qnn):
        if name not in minmax:
            continue
        threshold = _matching_stat(name, stats_by_name)
        if threshold is None or threshold <= 0:
            continue
        if not q.sym:
            raise ValueError(
                "TMMA initialization requires --a_sym: the max-average "
                "statistic is a symmetric activation magnitude."
            )
        delta_tmma = torch.full_like(q.delta.detach(), float(threshold) / float(q.n_levels))
        delta_minmax = q.delta.detach()
        delta = torch.clamp(delta_tmma + float(rho) * (delta_minmax - delta_tmma), min=1e-8)
        q.delta.data.copy_(delta)
        q.zero_point = 0
        tmma[name]["delta"] = delta_tmma.detach().float().cpu().clone()
        tmma[name]["clip"] = tmma[name]["delta"] * float(q.n_levels)
        changed += 1
        sq_ref += float(delta_tmma.detach().pow(2).sum().cpu())
        sq_delta += float((delta - delta_tmma).detach().pow(2).sum().cpu())

    summary = {
        "rho": float(rho),
        "matched_quantizers": int(changed),
        "relative_l2_to_tmma": float((sq_delta ** 0.5) / max(sq_ref ** 0.5, 1e-12)),
    }
    logger.info(
        "Initialization perturbation rho=%.3f: interpolated %d activation quantizers "
        "(relative scale distance to TMMA=%.5f)",
        rho, changed, summary["relative_l2_to_tmma"],
    )
    return tmma, summary


def initialize_tmma_scales(qnn: nn.Module, max_avg_stats: dict) -> dict:
    """Initialize learnable activation scales from TMMA statistics.

    The activation quantizers must first be materialized by an ordinary
    forward pass.  This function then replaces only their initial ``delta``
    values with ``S_tmma / (2^(q-1)-1)``.  It does not install clipping hooks
    or change the full-precision reconstruction target.
    """
    _tmma, summary = initialize_tmma_interpolation(qnn, max_avg_stats, rho=0.0)
    if summary["matched_quantizers"] == 0:
        raise RuntimeError(
            "TMMA statistics did not match any learnable activation quantizer"
        )
    logger.info(
        "TMMA initialized %d learnable activation quantizers before BRECQ",
        summary["matched_quantizers"],
    )
    return summary


@torch.no_grad()
def activation_quantization_probe(
    qnn: nn.Module,
    cali_data: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    batch_size: int = 32,
    max_batches: int = 4,
) -> dict:
    """Measure element-weighted activation quantization MSE and clipping rate."""
    sums = defaultdict(lambda: {"sq": 0.0, "n": 0, "clip": 0})
    handles = []

    def make_hook(name, quantizer):
        def hook(_module, inputs, output):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            x = inputs[0].detach()
            y = output.detach()
            rec = sums[name]
            rec["sq"] += float((x - y).float().pow(2).sum().cpu())
            rec["n"] += int(x.numel())
            if quantizer.delta is not None:
                clip = quantizer.delta.detach().abs() * float(quantizer.n_levels)
                rec["clip"] += int((x.abs() > clip).sum().item())
        return hook

    for name, q in iter_activation_quantizers(qnn):
        handles.append(q.register_forward_hook(make_hook(name, q)))
    xs, ts = cali_data
    old_state = None
    if hasattr(qnn, "set_quant_state"):
        qnn.set_quant_state(weight_quant=True, act_quant=True)
    qnn.eval()
    for start in range(0, min(xs.size(0), batch_size * max_batches), batch_size):
        _ = qnn(xs[start:start + batch_size].to(device), ts[start:start + batch_size].to(device))
    for handle in handles:
        handle.remove()
    per_layer = {
        name: {
            "mse": rec["sq"] / max(rec["n"], 1),
            "clip_rate": rec["clip"] / max(rec["n"], 1),
            "elements": rec["n"],
        }
        for name, rec in sums.items()
    }
    total_n = sum(v["elements"] for v in per_layer.values())
    return {
        "mean_layer_mse": float(np.mean([v["mse"] for v in per_layer.values()])) if per_layer else float("nan"),
        "element_weighted_mse": float(sum(v["mse"] * v["elements"] for v in per_layer.values()) / max(total_n, 1)),
        "element_weighted_clip_rate": float(sum(v["clip_rate"] * v["elements"] for v in per_layer.values()) / max(total_n, 1)),
        "per_layer": per_layer,
    }


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def two_stage_trajectory_probe(
    fp_model: nn.Module,
    stage1_qnn: nn.Module,
    stage2_qnn: nn.Module,
    betas: torch.Tensor,
    seq: List[int],
    split_idx: int,
    device: torch.device,
    n_trajectories: int,
    channels: int,
    image_size: int,
    seed: int,
    output_prefix: str,
) -> dict:
    """Run matched FP/quantized DDIM paths and save direct and latent errors.

    ``stage1_qnn`` is used for the first ``split_idx`` reverse steps and
    ``stage2_qnn`` afterwards, exactly matching two-stage sampling.  The same
    initial noise is used for FP and quantized paths.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    xq = torch.randn(n_trajectories, channels, image_size, image_size, generator=generator).to(device)
    xfp = xq.clone()
    fp_model.eval().to(device)
    stage1_qnn.eval().to("cpu")
    stage2_qnn.eval().to("cpu")
    seq_next = [-1] + list(seq[:-1])
    rows = []
    active = None

    for step_idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
        qnn = stage1_qnn if step_idx < split_idx else stage2_qnn
        if active is not qnn:
            if active is not None:
                active.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            qnn.to(device)
            qnn.set_quant_state(weight_quant=True, act_quant=True)
            active = qnn
        t = torch.full((n_trajectories,), float(i), device=device)
        t_next = torch.full((n_trajectories,), float(j), device=device)
        at, at_next = compute_alpha(betas, t.long()), compute_alpha(betas, t_next.long())
        eps_q = qnn(xq, t)
        eps_fp_at_q = fp_model(xq, t)
        eps_fp = fp_model(xfp, t)
        direct = (eps_q - eps_fp_at_q).float().pow(2).flatten(1).mean(1)
        x0_q = (xq - eps_q * (1.0 - at).sqrt()) / at.sqrt()
        x0_fp = (xfp - eps_fp * (1.0 - at).sqrt()) / at.sqrt()
        xq = at_next.sqrt() * x0_q + (1.0 - at_next).sqrt() * eps_q
        xfp = at_next.sqrt() * x0_fp + (1.0 - at_next).sqrt() * eps_fp
        err = (xq - xfp).float()
        mse = err.pow(2).flatten(1).mean(1)
        rel_l2 = err.flatten(1).norm(dim=1) / xfp.float().flatten(1).norm(dim=1).clamp(min=1e-12)
        rows.append({
            "step_index": int(step_idx), "t": int(i), "next_t": int(j),
            "stage": 0 if step_idx < split_idx else 1,
            "direct_eps_mse": float(direct.mean().cpu()),
            "latent_mse": float(mse.mean().cpu()),
            "latent_rel_l2": float(rel_l2.mean().cpu()),
        })

    final_mse = (xq - xfp).float().pow(2).flatten(1).mean(1).cpu().numpy()
    np.savez_compressed(
        output_prefix + "_samples.npz",
        final_mse=final_mse,
        final_error=(xq - xfp).float().cpu().numpy(),
    )
    payload = {
        "n_trajectories": int(n_trajectories),
        "seed": int(seed),
        "final_latent_mse_mean": float(final_mse.mean()),
        "final_latent_mse_std": float(final_mse.std(ddof=1)) if len(final_mse) > 1 else 0.0,
        "steps": rows,
        "sample_file": os.path.basename(output_prefix + "_samples.npz"),
    }
    save_json(output_prefix + ".json", payload)
    logger.info("Saved matched two-stage trajectory diagnostics to %s.json", output_prefix)
    return payload

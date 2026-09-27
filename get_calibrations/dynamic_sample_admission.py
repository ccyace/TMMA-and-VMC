"""
动态校准集构造（论文 Algorithm 1 及 Stage 1–3）

本脚本用于 PTQ（如 BRECQ）前的校准数据筛选：从大规模候选池中有目的地选出
budget 个 (x_t, t) 对，使校准集在激活分布上更具代表性。

================================================================================
三阶段流程
================================================================================
  Stage 1 — 时间步敏感度 S(t)
      对每个 DDIM 子步 t，在 batch 上估计 S(t) = E_x[ Var(A^(t)(x)) ]，
      其中 A 为所有 Conv2d 层激活拼接后的向量；再归一化得采样权重 w(t)。

  Stage 2 — 层敏感度 S_l 与敏感层集合 L_sen
      对每个卷积层 l，S_l = (1/T) Σ_t (1/N) Σ_i Var(A_l^(t)(x_i))；
      取 S_l 最高的前 α% 层作为敏感层 L_sen（Stage 3 仅在这些层上做准入判断）。

  Stage 3 — Algorithm 1 动态准入主循环
      按 w(t) 从候选池随机提议 (x_t, t)；对每个敏感层 l 计算若接受该样本后
      激活池方差的增量 Δ_l；若超过半数敏感层 Δ_l > 0 则接受，否则以概率
      (1 - discard_p) 随机接受；接受后更新在线池统计。

================================================================================
输入 / 输出
================================================================================
  候选池（二选一）：
    - --pool_pt：预生成 allst，如 cifar_sd1236_sample2048_allst.pt
      xs [T, N, C, H, W]、ts [T, N]，与 DDIM quad 100 子步对齐
    - 不指定 pool_pt：在线用 FP 模型跑 DDIM 生成小规模轨迹

  输出（与 cali_st=1 兼容，可直接作为 --cali_data_path）：
    { 'xs': [B, C, H, W], 'ts': [B], 'meta': {...} }

  两阶段量化对齐：
    --output_g0 / --output_g1 分别对 reversed(DDIM seq) 的前半/后半子步
    各跑一遍 Stage 1–3，对应 two_stage_newcalibrations.py 的 group0/group1。
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from pytorch_lightning import seed_everything

# ---------------------------------------------------------------------------
# 路径：将项目根目录加入 sys.path，以便 import ddim / qdiff 等模块
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ddim.functions.ckpt_util import get_ckpt_path
from ddim.functions.denoising import compute_alpha
from ddim.models.diffusion import Model

logger = logging.getLogger(__name__)


# =============================================================================
# 配置与扩散日程工具
# =============================================================================

def dict2namespace(config):
    """将 YAML 嵌套 dict 递归转为 argparse.Namespace，供 Model(config) 使用。"""
    import argparse as ap

    namespace = ap.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    构造扩散过程的 β 序列（与 ddim 采样脚本一致）。
    quad：在 sqrt 空间线性插值再平方，低噪声段步长更密。
    """
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def build_ddim_seq(num_train_timesteps: int, timesteps: int, skip_type: str) -> List[int]:
    """
    构造 DDIM 采样的离散时间索引序列 seq（升序，如 [0, 1, 4, 9, ...]）。

    注意：实际去噪时 generalized_steps 会对 seq 做 reversed()，
    因此 traj[k] 的第 k 维对应的是「去噪顺序」下的第 k 步子步。
    """
    if skip_type == "uniform":
        skip = num_train_timesteps // timesteps
        return list(range(0, num_train_timesteps, skip))
    if skip_type == "quad":
        seq = (
            np.linspace(0, np.sqrt(num_train_timesteps * 0.8), timesteps) ** 2
        )
        return [int(s) for s in list(seq)]
    raise NotImplementedError(skip_type)


def ddim_reversed_step_indices(timesteps: int) -> Tuple[List[int], List[int]]:
    """
    两阶段量化分组：与 cali_max_stat / two_stage_newcalibrations 一致。

    traj 第一维 k 已按去噪顺序排列（k=0 对应 reversed(seq) 的第一步，噪声最大）：
      - 组 0 (g0)：k ∈ [0, T/2)       — 高噪声半段，用于 stage1 BRECQ
      - 组 1 (g1)：k ∈ [T/2, T)       — 低噪声半段，用于 stage2 BRECQ

    Returns
    -------
    g0, g1 : 两组在 traj 第一维上的索引列表
    """
    half = timesteps // 2
    g0 = list(range(0, half))
    g1 = list(range(half, timesteps))
    return g0, g1


# =============================================================================
# 候选池加载与在线轨迹生成
# =============================================================================

def load_allst_candidate_pool(
    path: str,
    timesteps: int,
    skip_type: str,
    num_train_timesteps: int,
    strict_ts_check: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 export_ddim_*_allst.pt 加载预生成候选池。

    数据约定
    ---------
    xs : [T 或 T+1, N, C, H, W]  — 每步 N 张图的 UNet 输入状态 x_{t_k}
    ts : [T, N]                  — ts[k, :] 应为常数 t_k（同一子步共享时间索引）

    返回
    ----
    traj        : [T, N, C, H, W] float32，取 xs 前 T 帧
    ts_per_step : [T] float32，每步一个标量时间索引
    """
    d = torch.load(path, map_location="cpu")
    if not isinstance(d, dict) or "xs" not in d or "ts" not in d:
        raise ValueError("pool .pt 需为含键 xs、ts 的字典")
    xs, ts = d["xs"], d["ts"]
    if not (torch.is_tensor(xs) and torch.is_tensor(ts)):
        raise TypeError("xs / ts 须为 Tensor")
    if ts.ndim != 2:
        raise ValueError(f"ts 形状应为 [T,N]，当前 {tuple(ts.shape)}")
    T, N = int(ts.shape[0]), int(ts.shape[1])
    if T != timesteps:
        raise ValueError(f"pool 时间维 T={T} 与 --timesteps={timesteps} 不一致，请对齐后再跑。")
    if xs.shape[0] < T:
        raise ValueError(f"xs 第一维应 >= T，当前 xs.shape={tuple(xs.shape)}")
    if xs.shape[1] != N:
        logger.warning("xs N=%d 与 ts N=%d 不一致，将截断为 min。", xs.shape[1], N)
        n_min = min(int(xs.shape[1]), N)
        xs = xs[:, :n_min]
        ts = ts[:, :n_min]
        N = n_min
    traj = xs[:T].float().contiguous()
    # 每行 ts[k,:] 应相同，取 [:,0] 作为该子步的标量时间
    ts_per_step = ts.float()[:, 0].contiguous()
    if strict_ts_check:
        seq = build_ddim_seq(num_train_timesteps, timesteps, skip_type)
        # 去噪顺序下的时间索引 = reversed(升序 seq)
        expected = torch.tensor(list(reversed(seq)), dtype=torch.float32)
        diff = (ts_per_step - expected).abs().max().item()
        if diff > 0.51:
            logger.warning(
                "pool 内 ts[:,0] 与当前 config 下 reversed(DDIM seq) 最大偏差=%.3f；"
                "若该 .pt 由不同 timesteps/skip 生成，请用 --skip_ts_check 或换池文件。",
                diff,
            )
    logger.info("已加载候选池 %s traj=%s N=%d", path, tuple(traj.shape), N)
    return traj, ts_per_step


@torch.no_grad()
def ddim_trajectory_states(
    x_init: torch.Tensor,
    seq: List[int],
    model: nn.Module,
    betas: torch.Tensor,
    device: torch.device,
    eta: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在线生成 DDIM 轨迹：从 x_init（纯噪声）出发，逐步去噪并记录每步 UNet 输入。

    与 generalized_steps 的 DDIM 更新公式一致，但：
      - 设备可配置（不硬编码 cuda）
      - 每步前保存 cur 作为 states_before[k]（即该步网络看到的 x_{t_k}）

    Parameters
    ----------
    x_init : [B, C, H, W] 随机噪声起点
    seq    : 升序 DDIM 时间索引列表
    eta    : DDIM 随机性，0 为确定性 DDIM

    Returns
    -------
    states : [S, B, C, H, W]  S=len(seq)，第 k 步的 UNet 输入
    ts     : [S] 每步对应的离散时间 t_k（去噪顺序）
    """
    n = x_init.size(0)
    # seq_next[i] = seq[i-1]，末步 next 为 -1（表示 x_0）
    seq_next = [-1] + list(seq[:-1])
    rev = list(reversed(seq))
    rev_next = list(reversed(seq_next))
    states_before: List[torch.Tensor] = []
    ts_list: List[float] = []
    cur = x_init.to(device)
    betas = betas.to(device)

    for i, j in zip(rev, rev_next):
        # 在去噪到时间 i 之前，先记录当前状态（即 x_{t_i}）
        states_before.append(cur.detach().cpu().clone())
        ts_list.append(float(i))
        t = torch.ones(n, device=device) * i
        next_t = torch.ones(n, device=device) * j
        at = compute_alpha(betas, t.long())
        at_next = compute_alpha(betas, next_t.long())
        et = model(cur, t)
        # 预测 x_0
        x0_t = (cur - et * (1 - at).sqrt()) / at.sqrt()
        # DDIM 更新：x_{t-1} = sqrt(α_{t-1}) x_0 + c1·ε + c2·ε_θ
        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1**2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(cur) + c2 * et
        cur = xt_next

    return torch.stack(states_before, dim=0), torch.tensor(ts_list, dtype=torch.float32)


# =============================================================================
# 激活统计与 Hook 工具
# =============================================================================

def var_over_elements_per_sample(x: torch.Tensor) -> torch.Tensor:
    """
    对单张样本的激活张量，在除 batch 维外所有元素上求方差。

    x : [B, ...]  →  返回 [B]，每个元素为 Var(A(x_i))。
    Stage 2 逐层使用；Stage 1 则先拼接多层再算标量方差。
    """
    if x.ndim == 1:
        return x.unsqueeze(0).var(unbiased=False, dim=-1)
    return x.flatten(1).var(unbiased=False, dim=1)


def list_conv2d_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """枚举 UNet 中所有 Conv2d 层（含名字），用于注册 hook 与敏感度统计。"""
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            out.append((name, m))
    return out


class ActivationHook:
    """
    在 Conv2d forward 时捕获输出激活，供 Stage 1/2/3 使用。

    用法：
        hook = ActivationHook(conv_names)
        hook.register(model, name_subset=sensitive_names)  # 仅 hook 指定层
        _ = model(x, t)
        acts = hook._acts  # {layer_name: Tensor[B,C,H,W]}
        hook.remove()
    """

    def __init__(self, names: List[str]):
        self.names = names
        self._acts: Dict[str, torch.Tensor] = {}
        self._handles = []

    def _make_hook(self, name: str):
        def hook(_m, _inp, output):
            if isinstance(output, torch.Tensor):
                self._acts[name] = output.detach()

        return hook

    def register(self, model: nn.Module, name_subset: Optional[List[str]] = None):
        """注册 hook；name_subset 非空时只 hook 列表中的层（Stage 3 仅用敏感层）。"""
        self.remove()
        self.clear()
        subset = set(name_subset) if name_subset is not None else None
        for name, m in model.named_modules():
            if not isinstance(m, nn.Conv2d):
                continue
            if subset is not None and name not in subset:
                continue
            h = m.register_forward_hook(self._make_hook(name))
            self._handles.append(h)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self):
        """清空前一次 forward 缓存的激活（每次 model 前向前应 clear）。"""
        self._acts.clear()

    def values_ordered(self, names: List[str]) -> List[torch.Tensor]:
        return [self._acts[n] for n in names if n in self._acts]


def mean_layerwise_element_variance(
    hook: ActivationHook, ordered_names: List[str]
) -> torch.Tensor:
    """Compute Eq. (24)'s equal-weight layer average for one batch.

    Each layer first contributes one element-wise activation variance per
    sample.  The result is then averaged over layers, so layers with larger
    activation tensors do not receive a larger implicit weight.

    Returns
    -------
    Tensor[B]
        The mean layer-wise variance for every sample in the batch.
    """
    layer_vars = [
        var_over_elements_per_sample(hook._acts[name])
        for name in ordered_names
        if name in hook._acts
    ]
    if not layer_vars:
        raise RuntimeError("No activation tensors were captured for timestep scoring")
    return torch.stack(layer_vars, dim=0).mean(dim=0)

# =============================================================================
# Stage 1：时间步敏感度 S(t) → 权重 w(t)
# =============================================================================

@torch.no_grad()
def estimate_timestep_sensitivity(
    model: nn.Module,
    traj: torch.Tensor,
    ts_per_step: torch.Tensor,
    conv_names: List[str],
    device: torch.device,
    hook: ActivationHook,
    batches_per_step: int = 2,
    batch_size: int = 8,
    rng_seed: int = 0,
) -> torch.Tensor:
    """
    Stage 1：估计每个 DDIM 子步的时间敏感度 S(t)。

    对每个子步 k（固定 x 来自 traj[k, :]）：
      1. 随机抽 batch_size 张图，固定时间 t = ts_per_step[k]
      2. FP 前向，hook 所有 Conv2d
      3. 对每张图、每层分别求 Var_elem(A_l(x))
      4. 对层等权平均，再对 batch 和多轮 batches_per_step 求均值 → S(t_k)

    直觉：某时间步若激活在空间/通道上波动大，量化误差影响也大，应提高抽样权重。

    Parameters
    ----------
    traj : [S, N_pool, C, H, W] on CPU，候选池

    Returns
    -------
    S_out : [S] float64，每个子步的 S(t)；末尾 +1e-12 防除零
    """
    S_steps, n_pool = traj.shape[0], traj.shape[1]
    S_out = torch.zeros(S_steps, dtype=torch.float64)
    rng = np.random.RandomState(rng_seed)

    for k in range(S_steps):
        vals = []
        t_scalar = ts_per_step[k].item()
        for _ in range(batches_per_step):
            idx = rng.choice(n_pool, size=min(batch_size, n_pool), replace=False)
            x = traj[k, idx].to(device)
            t = torch.full((x.size(0),), t_scalar, device=device, dtype=torch.float32)
            hook.clear()
            hook.register(model, conv_names)
            _ = model(x, t)
            # Eq. (24): first compute Var_elem independently for every layer,
            # then average layers with equal weight and finally average D_t.
            per_sample = mean_layerwise_element_variance(hook, conv_names)
            vals.append(float(per_sample.mean().item()))
        hook.remove()
        S_out[k] = float(np.mean(vals)) + 1e-12
    return S_out


# =============================================================================
# Stage 2：层敏感度 S_l → 敏感层 L_sen
# =============================================================================

@torch.no_grad()
def estimate_layer_sensitivity(
    model: nn.Module,
    traj: torch.Tensor,
    ts_per_step: torch.Tensor,
    conv_names: List[str],
    device: torch.device,
    hook: ActivationHook,
    max_samples_per_step: int = 16,
    rng_seed: int = 1,
) -> Dict[str, float]:
    """
    Stage 2：按论文公式 (17) 估计每层敏感度。

        S_l = (1/T) Σ_t (1/N) Σ_i Var(A_l^(t)(x_i))

    实现：对每个子步 k 随机抽最多 max_samples_per_step 张图前向，
    逐层算 var_over_elements_per_sample 的 batch mean，累加后除以 T。

    Returns
    -------
    dict {layer_name: S_l}
    """
    S_steps, n_pool = traj.shape[0], traj.shape[1]
    rng = np.random.RandomState(rng_seed)
    acc = {n: 0.0 for n in conv_names}
    for k in range(S_steps):
        t_scalar = ts_per_step[k].item()
        idx = rng.choice(n_pool, size=min(max_samples_per_step, n_pool), replace=False)
        x = traj[k, idx].to(device)
        t = torch.full((x.size(0),), t_scalar, device=device, dtype=torch.float32)
        hook.clear()
        hook.register(model, conv_names)
        _ = model(x, t)
        for name in conv_names:
            if name not in hook._acts:
                continue
            v = var_over_elements_per_sample(hook._acts[name]).mean().item()
            acc[name] += v
        hook.remove()
    inv_t = 1.0 / float(S_steps)
    return {n: acc[n] * inv_t for n in conv_names}


def select_top_fraction(names_scores: List[Tuple[str, float]], alpha_percent: float) -> List[str]:
    """
    按 S_l 降序取前 α% 层作为 L_sen。

    alpha_percent=30 表示取 ceil(0.3 * num_layers) 层，至少 1 层。
    """
    names_scores = sorted(names_scores, key=lambda x: -x[1])
    n = max(1, int(math.ceil(len(names_scores) * alpha_percent / 100.0)))
    return [n for n, _ in names_scores[:n]]


# =============================================================================
# Stage 3：在线池统计与 Algorithm 1 主循环
# =============================================================================

def pool_stats_merge(
    N: float, S1: float, S2: float, act_flat: torch.Tensor
) -> Tuple[float, float, float, float]:
    """
    将新样本的激活合并进已有池，更新充分统计量并计算新方差。

    维护 (N, S1, S2) 其中：
      S1 = Σ x_i
      S2 = Σ x_i²
      Var = S2/N - (S1/N)²

    对应论文式 (19)-(21)。Stage 3 用 Var' - Var 作为接受判据的 Δ_l。
    """
    n = float(act_flat.numel())
    s1 = float(act_flat.sum().item())
    s2 = float((act_flat * act_flat).sum().item())
    Np = N + n
    S1p = S1 + s1
    S2p = S2 + s2
    if Np <= 0:
        return Np, S1p, S2p, 0.0
    varp = S2p / Np - (S1p / Np) ** 2
    return Np, S1p, S2p, varp


def run_algorithm(
    model: nn.Module,
    traj: torch.Tensor,
    ts_per_step: torch.Tensor,
    conv_names: List[str],
    sensitive_names: List[str],
    w: np.ndarray,
    device: torch.device,
    budget: int,
    discard_p: float,
    max_proposals: int,
    rng_seed: int = 12345,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stage 3 / Algorithm 1 主循环：动态准入构造最终校准集。

    每次提议 (proposal)：
      1. 按剩余候选质量重归一化后的 w 加权随机选子步 k，
         再从该子步的未使用图像索引中均匀抽取 img_i
      2. 取 x1 = traj[k, img_i]，t = ts_per_step[k]
      3. FP 前向，仅 hook 敏感层 L_sen
      4. 对每层 l：假设接受 x1，用 pool_stats_merge 算合并后方差 Var'
         delta_l = Var' - Var；若 delta_l > 0 则 m += 1
      5. 接受规则：
           - m > |L_sen|/2  → 必接受（多数敏感层认为有信息增益）
           - 否则           → u ~ U(0,1)，u > discard_p 则接受（默认 10% 接受率）
      6. 接受则更新 pool 统计，并加入 accepted_x / accepted_t

    Parameters
    ----------
    budget        : 目标校准集大小 B
    discard_p     : 弱提议的丢弃概率 p（论文伪代码）
    max_proposals : 最大提议次数，防止 discard_p 过大导致死循环

    Returns
    -------
    cali_xs : [B, C, H, W]
    cali_ts : [B] float，与 two_stage_newcalibrations / BRECQ 脚本一致
    """
    S_steps, n_pool = traj.shape[0], traj.shape[1]
    hook = ActivationHook(conv_names)

    # 每个敏感层维护独立的在线激活池统计（仅在接受样本后更新）
    pool_N = {n: 0.0 for n in sensitive_names}
    pool_S1 = {n: 0.0 for n in sensitive_names}
    pool_S2 = {n: 0.0 for n in sensitive_names}
    pool_Var = {n: 0.0 for n in sensitive_names}

    accepted_x: List[torch.Tensor] = []
    accepted_t: List[float] = []

    proposals = 0
    rng = np.random.default_rng(rng_seed)

    # Algorithm 1 retrieves an unused candidate on every proposal.  Keep a
    # swap-pop pool per timestep so each (step, image) pair is proposed at
    # most once without repeatedly allocating or scanning a global set.
    unused_by_step: List[List[int]] = [
        list(range(n_pool)) for _ in range(S_steps)
    ]

    while len(accepted_x) < budget and proposals < max_proposals:
        available = np.fromiter(
            (len(indices) > 0 for indices in unused_by_step),
            dtype=np.float64,
            count=S_steps,
        )
        if available.sum() == 0:
            logger.warning(
                "候选池已耗尽：所有 %d 个 (timestep, image) pair 均已提议。",
                S_steps * n_pool,
            )
            break

        proposals += 1
        # 已耗尽的 timestep 权重置零，再对剩余 timestep 重归一化。
        remaining_w = np.asarray(w, dtype=np.float64) * available
        remaining_sum = float(remaining_w.sum())
        if remaining_sum <= 0.0 or not np.isfinite(remaining_sum):
            remaining_w = available / available.sum()
        else:
            remaining_w /= remaining_sum
        k = int(rng.choice(S_steps, p=remaining_w))

        # 从 timestep k 的未使用索引中均匀抽取，并用 swap-pop O(1) 删除。
        candidates_k = unused_by_step[k]
        pos = int(rng.integers(0, len(candidates_k)))
        img_i = int(candidates_k[pos])
        candidates_k[pos] = candidates_k[-1]
        candidates_k.pop()
        x1 = traj[k, img_i : img_i + 1].to(device)
        t_scalar = float(ts_per_step[k].item())
        t = torch.full((1,), t_scalar, device=device, dtype=torch.float32)

        hook.clear()
        hook.register(model, sensitive_names)
        _ = model(x1, t)

        m = 0  # 方差增大的敏感层计数
        hat = {}  # 每层预计算的合并后统计 (Np, S1p, S2p, Var_p, delta)
        for name in sensitive_names:
            if name not in hook._acts:
                hat[name] = (pool_N[name], pool_S1[name], pool_S2[name], pool_Var[name], 0.0)
                continue
            flat = hook._acts[name].detach().float().reshape(-1)
            N, S1, S2, Var = pool_N[name], pool_S1[name], pool_S2[name], pool_Var[name]
            Np, S1p, S2p, Var_p = pool_stats_merge(N, S1, S2, flat)
            delta = Var_p - Var
            hat[name] = (Np, S1p, S2p, Var_p, delta)
            if delta > 0:
                m += 1

        n_sen = len(sensitive_names)
        # Algorithm 1 initializes the pool with the first unused candidate.
        # Besides matching the paper, this guarantees a valid non-empty pool
        # even when every captured activation tensor is constant.
        if proposals == 1:
            accept = True
        elif m > n_sen / 2.0:
            accept = True
        else:
            # u > p 则接受；discard_p=0.9 时约 10% 随机接受，增加多样性
            accept = float(rng.random()) > discard_p

        if accept:
            accepted_x.append(x1.cpu().squeeze(0).clone())
            accepted_t.append(t_scalar)
            for name in sensitive_names:
                Np, S1p, S2p, Var_p, _ = hat[name]
                pool_N[name] = Np
                pool_S1[name] = S1p
                pool_S2[name] = S2p
                pool_Var[name] = Var_p

    hook.remove()

    if len(accepted_x) < budget:
        logger.warning(
            "预算未满：接受 %d / %d（已提议 %d 个不重复候选，max_proposals=%d）。"
            "可扩大候选池、调大 max_proposals 或减小 discard_p。",
            len(accepted_x),
            budget,
            proposals,
            max_proposals,
        )

    cali_xs = torch.stack(accepted_x, dim=0)
    cali_ts = torch.tensor(accepted_t, dtype=torch.float32)
    return cali_xs, cali_ts


# =============================================================================
# Stage 1–3 串联与保存
# =============================================================================

def run_stages_and_save(
    model: nn.Module,
    traj: torch.Tensor,
    ts_per_step: torch.Tensor,
    conv_names: List[str],
    device: torch.device,
    args: argparse.Namespace,
    output_path: str,
    rng_offset: int,
    meta_extra: Dict,
) -> None:
    """
    对给定 traj 子集（可为全 T 步或 g0/g1 半段）依次执行 Stage1→2→3 并保存 .pt。

    rng_offset：分组模式下 g0/g1 使用不同随机种子偏移，保证两次运行独立可复现。
    meta_extra：并入输出 meta（如 ddim_group、pool_source 等）。
    """
    hook = ActivationHook(conv_names)
    base_seed = int(args.seed) + int(rng_offset)

    # --- Stage 1 ---
    S_vec = estimate_timestep_sensitivity(
        model,
        traj,
        ts_per_step,
        conv_names,
        device,
        hook,
        batches_per_step=args.s1_batches,
        batch_size=args.s1_batch_size,
        rng_seed=base_seed,
    )
    w = (S_vec / S_vec.sum()).numpy().astype(np.float64)
    logger.info(
        "Stage1 完成 (rng_off=%s)，S 子步=%d 范围 [%.4e, %.4e]",
        rng_offset,
        traj.shape[0],
        float(S_vec.min()),
        float(S_vec.max()),
    )

    # --- Stage 2 ---
    S_layer = estimate_layer_sensitivity(
        model,
        traj,
        ts_per_step,
        conv_names,
        device,
        hook,
        max_samples_per_step=args.s2_max_samples,
        rng_seed=base_seed + 1,
    )
    ranked = [(n, S_layer[n]) for n in conv_names]
    sensitive = select_top_fraction(ranked, args.alpha)
    logger.info("Stage2 |L_sen|=%d alpha=%.2f%%", len(sensitive), args.alpha)

    # --- Stage 3 ---
    cali_xs, cali_ts = run_algorithm(
        model,
        traj,
        ts_per_step,
        conv_names,
        sensitive,
        w,
        device,
        budget=args.budget,
        discard_p=args.discard_p,
        max_proposals=args.max_proposals,
        rng_seed=base_seed + 9000,
    )
    logger.info("Stage3 校准集 xs=%s ts=%s", tuple(cali_xs.shape), tuple(cali_ts.shape))

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    meta = {
        "algorithm": "dynamic_sample_admission_var_max",
        "budget": args.budget,
        "alpha_percent": args.alpha,
        "discard_p": args.discard_p,
        "ddim_timesteps": args.timesteps,
        "skip_type": args.skip_type,
        "sensitive_layers": sensitive,
        "w_per_ddim_step": w.tolist(),
        "S_timestep": S_vec.tolist(),
    }
    meta.update(meta_extra)
    torch.save({"xs": cali_xs, "ts": cali_ts, "meta": meta}, output_path)
    logger.info("已保存: %s", os.path.abspath(output_path))


def parse_args():
    p = argparse.ArgumentParser(description="Algorithm 1 动态校准集构造")
    p.add_argument("--config", type=str, default="configs/cifar10.yml")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--timesteps", type=int, default=100, help="DDIM 子步数 T_eff")
    p.add_argument("--skip_type", type=str, default="quad", choices=["quad", "uniform"])
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--pool_images", type=int, default=128, help="候选池轨迹条数（噪声起点数）")
    p.add_argument("--pool_batch", type=int, default=16, help="生成轨迹时每批噪声图像数")
    p.add_argument("--budget", type=int, default=256, help="B：最终校准集大小")
    p.add_argument("--alpha", type=float, default=30.0, help="敏感层比例 α（百分数，如 30 表示前 30%%）")
    p.add_argument("--discard_p", type=float, default=0.9, help="伪代码中的丢弃概率 p（u>p 则接受）")
    p.add_argument("--s1_batches", type=int, default=2, help="Stage1 每步估计 S(t) 的 batch 轮数")
    p.add_argument("--s1_batch_size", type=int, default=8)
    p.add_argument("--s2_max_samples", type=int, default=16, help="Stage2 每步最多抽样图像数 N")
    p.add_argument("--max_proposals", type=int, default=500000, help="主循环最大提议次数防死循环")
    p.add_argument("--output", type=str, default="out/dynamic_cali/cali_dynamic.pt")
    p.add_argument(
        "--pool_pt",
        type=str,
        default="",
        help="预生成 allst 候选池 .pt（如 cifar_sd1236_sample2048_allst.pt），"
        "需满足 ts.shape[0]==timesteps，xs 第一维至少为 timesteps（常用 xs[:T] 与 ts 对齐）",
    )
    p.add_argument(
        "--output_g0",
        type=str,
        default="",
        help="与 --output_g1 同时指定时：仅对 DDIM reversed(seq) 前一半子步建动态校准集并保存",
    )
    p.add_argument(
        "--output_g1",
        type=str,
        default="",
        help="与 --output_g0 同时指定时：仅对 reversed(seq) 后一半子步建动态校准集并保存",
    )
    p.add_argument(
        "--skip_ts_check",
        action="store_true",
        help="加载 pool_pt 时不校验 ts[:,0] 与当前 quad/uniform 序列是否一致",
    )
    args = p.parse_args()
    if bool(args.output_g0) ^ bool(args.output_g1):
        p.error("分组输出须同时提供 --output_g0 与 --output_g1，或两者都不提供而仅用 --output。")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    seed_everything(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，改用 CPU。")

    # ---------- 加载 config 与扩散 β 日程 ----------
    with open(args.config, "r", encoding="utf-8") as f:
        config = dict2namespace(yaml.safe_load(f))
    config.split_shortcut = getattr(config, "split_shortcut", False)

    betas_np = get_beta_schedule(
        config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas_np).float()

    seq = build_ddim_seq(config.diffusion.num_diffusion_timesteps, args.timesteps, args.skip_type)
    logger.info("DDIM 序列长度 |seq|=%d，首末时间步: %s -> %s", len(seq), seq[0], seq[-1])

    # ---------- 加载 FP UNet（用于前向估计敏感度，不参与量化） ----------
    model = Model(config)
    if str(config.data.dataset).upper() == "CIFAR10":
        ckpt_name = "cifar10"
    elif str(config.data.dataset).upper() == "LSUN":
        ckpt_name = f"lsun_{config.data.category}"
    else:
        raise ValueError(config.data.dataset)
    ckpt = get_ckpt_path(f"ema_{ckpt_name}")
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.to(device)
    model.eval()

    conv_layers = list_conv2d_modules(model)
    conv_names = [n for n, _ in conv_layers]
    logger.info("卷积层数量: %d（用于 Stage1/2 与敏感层筛选）", len(conv_names))

    # ---------- 候选池：预生成 allst 或在线 DDIM 轨迹 ----------
    if args.pool_pt:
        traj, ts_per_step = load_allst_candidate_pool(
            args.pool_pt,
            args.timesteps,
            args.skip_type,
            config.diffusion.num_diffusion_timesteps,
            strict_ts_check=not args.skip_ts_check,
        )
        pool_meta = {"pool_source": "allst_file", "pool_pt": os.path.abspath(args.pool_pt)}
    else:
        # 无 pool_pt 时：用 FP 模型在线跑 DDIM，生成 pool_images 条轨迹
        n_total = args.pool_images
        traj_chunks: List[torch.Tensor] = []
        ts_ref: Optional[torch.Tensor] = None
        n_done = 0
        C, H, W = config.data.channels, config.data.image_size, config.data.image_size
        while n_done < n_total:
            bs = min(args.pool_batch, n_total - n_done)
            x0 = torch.randn(bs, C, H, W, device=device)
            states, ts_step = ddim_trajectory_states(
                x0, seq, model, betas, device, eta=args.eta
            )
            if ts_ref is None:
                ts_ref = ts_step
            else:
                if not torch.allclose(ts_ref, ts_step):
                    raise RuntimeError("DDIM 时间步子序列不一致")
            # states: [S, bs, C, H, W] → 沿 N 维拼接
            traj_chunks.append(states.cpu())
            n_done += bs
            logger.info("轨迹生成进度: %d / %d", n_done, n_total)

        traj = torch.cat(traj_chunks, dim=1)[:, :n_total]
        assert ts_ref is not None
        ts_per_step = ts_ref
        pool_meta = {"pool_source": "synthetic_ddim", "pool_images": n_total}

    logger.info("候选池 traj=%s ts_per_step=%s", tuple(traj.shape), tuple(ts_per_step.shape))

    # ---------- 全量 or 分组 (g0/g1) 运行 Stage 1–3 ----------
    split_mode = bool(args.output_g0) and bool(args.output_g1)
    if split_mode:
        idx0, idx1 = ddim_reversed_step_indices(args.timesteps)
        logger.info(
            "分组量化对齐：组0 子步索引 %d..%d（共 %d），组1 %d..%d（共 %d），"
            "与 reversed(quad seq) 前半/后半一致。",
            idx0[0],
            idx0[-1],
            len(idx0),
            idx1[0],
            idx1[-1],
            len(idx1),
        )
        # rng_offset 不同 → g0/g1 随机抽样独立；g1 用 +100000 避免与 g0 撞种子
        runs = [
            ("g0", idx0, args.output_g0, 0),
            ("g1", idx1, args.output_g1, 100000),
        ]
        for tag, indices, out_path, rng_off in runs:
            traj_s = traj[indices].contiguous()
            ts_s = ts_per_step[indices].contiguous()
            logger.info("[%s] traj=%s", tag, tuple(traj_s.shape))
            meta_extra = {
                **pool_meta,
                "ddim_group": tag,
                "reversed_step_indices": list(indices),
                "description": (
                    "reversed(seq) 前半子步（高噪声半段）"
                    if tag == "g0"
                    else "reversed(seq) 后半子步（低噪声半段）"
                ),
            }
            run_stages_and_save(
                model,
                traj_s,
                ts_s,
                conv_names,
                device,
                args,
                out_path,
                rng_offset=rng_off,
                meta_extra=meta_extra,
            )
    else:
        meta_extra = {**pool_meta, "ddim_group": "full", "reversed_step_indices": list(range(args.timesteps))}
        run_stages_and_save(
            model,
            traj,
            ts_per_step,
            conv_names,
            device,
            args,
            args.output,
            rng_offset=0,
            meta_extra=meta_extra,
        )


if __name__ == "__main__":
    main()

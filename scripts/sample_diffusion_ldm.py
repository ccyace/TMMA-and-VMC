import argparse, os, sys, gc, glob, datetime, yaml
import logging
import time
import numpy as np
from tqdm import trange, tqdm
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from PIL import Image
import torchvision.utils as tvu
sys.path.append(r'C:\Users\wsh\Desktop\q-diffusion-master')
import torch
import torch.nn as nn
import sys
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.util import instantiate_from_config
from ddim.functions.denoising import generalized_steps
from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock, 
    block_reconstruction, layer_reconstruction,
)
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_block import QuantQKMatMul, QuantSMVMatMul
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import resume_cali_model, get_train_samples
from qdiff.max_avg_truncation import (
    load_max_avg_statistics,
)
from qdiff.init_perturbation import initialize_tmma_scales

logger = logging.getLogger(__name__)


rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def convsample_dpm(model, steps, shape, eta=1.0
                    ):
    dpm = DPMSolverSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = dpm.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0, dpm=False):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    # with model.ema_scope("Plotting"):
    t0 = time.time()
    if vanilla:
        sample, progrow = convsample(model, shape,
                                        make_prog_row=True)
    elif dpm:
        logger.info(f'Using DPM sampling with {custom_steps} sampling steps and eta={eta}')
        sample, intermediates = convsample_dpm(model,  steps=custom_steps, shape=shape,
                                                eta=eta)
    else:
        sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                eta=eta)

    t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    logger.info(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, 
    n_samples=50000, nplog=None, dpm=False):
    if vanilla:
        logger.info(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        logger.info(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        logger.info(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta, dpm=dpm)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                logger.info(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    logger.info(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume_base",
        type=str,
        nargs="?",
        help="load fp32 base model from logdir or checkpoint in logdir (will deprecate after direct quantized model loading implemented)",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=10000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "--seed",
        type=int,
        # default=42,
        required=True,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fast dpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    # linear quantization configs
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument(
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
    )
    parser.add_argument(
        "--weight_bit",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--quant_mode", type=str, default="qdiff", 
        choices=["qdiff"], 
        help="quantization mode to use"
    )
    # qdiff specific configs
    parser.add_argument(
        "--cali_st", type=int, default=1, 
        help="number of timesteps used for calibration; two-stage reads this many steps from each slice"
    )
    parser.add_argument(
        "--cali_batch_size", type=int, default=32, 
        help="batch size for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_n", type=int, default=1024, 
        help="samples kept at each selected timestep; two-stage total per group is cali_st * cali_n"
    )
    parser.add_argument(
        "--cali_iters", type=int, default=20000, 
        help="number of iterations for each qdiff reconstruction"
    )
    parser.add_argument('--cali_iters_a', default=5000, type=int, 
        help='number of iteration for LSQ')
    parser.add_argument('--cali_lr', default=4e-4, type=float, 
        help='learning rate for LSQ')
    parser.add_argument('--cali_p', default=2.4, type=float, 
        help='L_p norm minimization for LSQ')
    parser.add_argument(
        "--cali_ckpt", type=str,
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--cali_data_path", type=str, default="sd_coco_sample1024_allst.pt",
        help="单阶段 PTQ 的校准集路径",
    )
    parser.add_argument(
        "--cali_data_path_group0",
        type=str,
        default="",
        help="两阶段组0（前半段高噪声）扁平校准集，须与 group1 同时给出",
    )
    parser.add_argument(
        "--cali_data_path_group1",
        type=str,
        default="",
        help="两阶段组1（后半段低噪声）扁平校准集，须与 group0 同时给出",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the calibrated qdiff model"
    )
    parser.add_argument(
        "--resume_w", action="store_true",
        help="resume the calibrated qdiff model weights only"
    )
    parser.add_argument(
        "--cond", action="store_true",
        help="whether to use conditional guidance"
    )
    parser.add_argument(
        "--a_sym", action="store_true",
        help="act quantizers use symmetric quantization (empirically helpful in some cases)"
    )
    parser.add_argument(
        "--a_min_max", action="store_true",
        help="act quantizers initialize with min-max (empirically helpful in some cases)"
    )
    parser.add_argument(
        "--running_stat", action="store_true",
        help="use running statistics for act quantizers"
    )
    parser.add_argument(
        "--rs_sm_only", action="store_true",
        help="use running statistics only for softmax act quantizers"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument(
        "--dpm", action="store_true",
        help="use dpm solver for sampling"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    parser.add_argument(
        "--init_debug_log", action="store_true",
        help="print progress logs during weight quantizer initialization"
    )
    parser.add_argument(
        "--init_log_interval", type=int, default=50,
        help="log interval (in visited QuantModule count) for init debug progress"
    )
    parser.add_argument(
        "--two_stage",
        action="store_true",
        help="两阶段 TMMA：组0 BRECQ+前半采样，组1 BRECQ+后半采样（需 --max_avg_json）",
    )
    parser.add_argument(
        "--recon_only",
        action="store_true",
        help="两阶段只做 BRECQ 并保存/合并权重，跳过前后半采样和 VAE 解码",
    )
    parser.add_argument(
        "--max_avg_json",
        type=str,
        default="",
        help="分组 TMMA max_avg 统计 JSON（两阶段必填；单阶段可选一组作尺度初始化）",
    )
    parser.add_argument(
        "--max_avg_group",
        type=int,
        default=0,
        choices=[0, 1],
        help="单阶段激活 BRECQ 使用的时间组（两阶段模式下忽略）",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="quad",
        choices=["quad", "uniform"],
        help="两阶段半段采样用的 DDIM 步序列类型",
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        logger.info(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


def _clone_ldm_unet(ldm_model, config):
    """从当前 LDM 拷贝一份独立 U-Net（供两阶段各自 QuantModel 使用）。"""
    unet = instantiate_from_config(config.model.params.unet_config)
    unet.load_state_dict(ldm_model.model.diffusion_model.state_dict(), strict=False)
    unet.cuda()
    unet.eval()
    return unet


def _build_ddim_seq(num_timesteps, custom_steps, skip_type):
    total_timesteps = num_timesteps
    if skip_type == "uniform":
        skip = total_timesteps // custom_steps
        seq = list(range(0, total_timesteps, skip))
    elif skip_type == "quad":
        seq = (
            np.linspace(0, np.sqrt(total_timesteps * 0.8), custom_steps) ** 2
        )
        seq = [int(s) for s in list(seq)]
        seq = sorted(list(set(seq)))
    else:
        raise ValueError(f"不支持的 skip_type: {skip_type}")
    return seq


def _half_ddim_seq(num_timesteps, custom_steps, skip_type, first_half=True):
    seq = _build_ddim_seq(num_timesteps, custom_steps, skip_type)
    reversed_seq = list(reversed(seq))
    half_steps = len(reversed_seq) // 2
    if first_half:
        return reversed_seq[:half_steps]
    return reversed_seq[half_steps:]


def _pick_exact_steps(nsteps, num_st):
    """在一组的时间步里均匀抽出正好 num_st 个下标（含两端）。"""
    num_st = int(num_st)
    if num_st <= 0 or num_st >= nsteps:
        return list(range(nsteps))
    if num_st == 1:
        return [0]
    picked = []
    used = set()
    for i in range(num_st):
        j = int(round(i * (nsteps - 1) / (num_st - 1)))
        if j in used:
            for delta in range(1, nsteps):
                if j + delta < nsteps and (j + delta) not in used:
                    j = j + delta
                    break
                if j - delta >= 0 and (j - delta) not in used:
                    j = j - delta
                    break
        used.add(j)
        picked.append(j)
    picked.sort()
    return picked


def _subset_row_index(ts, cali_st, cali_n):
    """按连续相同时间步分块，只给出要拷贝的行号，不碰 xs。"""
    ts = ts.detach().reshape(-1)
    if ts.numel() == 0:
        raise ValueError("校准集 ts 为空")
    changed = torch.ones(ts.numel(), dtype=torch.bool)
    changed[1:] = ts[1:] != ts[:-1]
    starts = torch.nonzero(changed, as_tuple=False).flatten()
    ends = torch.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = ts.numel()
    step_ids = _pick_exact_steps(int(starts.numel()), cali_st)
    take = int(cali_n)
    rows = []
    for s in step_ids:
        a = int(starts[s])
        b = int(ends[s])
        n = min(take, b - a)
        rows.append(torch.arange(a, a + n, dtype=torch.long))
    return torch.cat(rows), len(step_ids), int(starts.numel())


def _load_group_slice(path, stage_tag, group_id, cali_st, cali_n):
    """内存映射打开切片，只把选中的行拷进内存。"""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"[{stage_tag}] 找不到校准集: {path}\n"
            "请同时传入 --cali_data_path_group0 与 --cali_data_path_group1"
        )
    logger.info("[%s] 加载校准数据: %s (group_id=%d)", stage_tag, path, group_id)
    data = torch.load(path, map_location="cpu", mmap=True)
    xs, ts = data["xs"], data["ts"]
    if not torch.is_tensor(xs):
        raise ValueError(f"[{stage_tag}] xs 不是 Tensor: {path}")
    if not torch.is_tensor(ts):
        ts = torch.as_tensor(ts)
    if xs.shape[0] != ts.reshape(-1).shape[0]:
        raise ValueError(
            f"[{stage_tag}] xs/ts 数量不一致: {xs.shape[0]} vs {ts.reshape(-1).shape[0]}"
        )
    row_index, n_picked, n_steps = _subset_row_index(ts, cali_st, cali_n)
    xs_out = xs.index_select(0, row_index).clone().float()
    ts_out = ts.reshape(-1).index_select(0, row_index).clone().float()
    del data, xs, ts, row_index
    gc.collect()
    logger.info(
        "[%s] 切片共 %d 个时间步，抽取 %d 步 × 每步最多 %d = %d 条，形状 %s %s，时间步 [%.1f, %.1f]",
        stage_tag,
        n_steps,
        n_picked,
        int(cali_n),
        int(xs_out.shape[0]),
        tuple(xs_out.shape),
        tuple(ts_out.shape),
        float(ts_out.min()),
        float(ts_out.max()),
    )
    return xs_out, ts_out


def _resolve_two_stage_cali_paths(opt):
    g0 = (getattr(opt, "cali_data_path_group0", None) or "").strip()
    g1 = (getattr(opt, "cali_data_path_group1", None) or "").strip()
    if not g0 or not g1:
        raise ValueError("两阶段须同时提供 --cali_data_path_group0 与 --cali_data_path_group1")
    return g0, g1


def _skip_memory_heavy_attn_recon(qnn, x, t, min_spatial_side=16):
    """
    跳过空间边长 >= min_spatial_side 的注意力相关层的 BRECQ。
    这些层缓存 5120 条激活会撑爆主机内存（qkv≈12GB，QK≈160GB）。
    仍保留 min-max / 一次前向得到的量化刻度，只是不做 AdaRound / 激活重建。
    """
    token_thr = int(min_spatial_side) ** 2
    skipped = []
    handles = []

    def _mark(mod, name, tokens, kind):
        if tokens < token_thr or getattr(mod, "ignore_reconstruction", False):
            return
        mod.ignore_reconstruction = True
        skipped.append((name, kind, int(tokens)))

    for name, mod in qnn.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(mod, (QuantQKMatMul, QuantSMVMatMul)):
            def hook(module, inp, out, n=name):
                if torch.is_tensor(out) and out.ndim == 3:
                    # QK: [BH,T,T]；SMV: [BH,C,T]
                    tokens = int(out.shape[-1])
                    kind = "qk" if out.shape[-2] == out.shape[-1] else "smv"
                    _mark(module, n, tokens, kind)
                elif inp and torch.is_tensor(inp[0]) and inp[0].ndim == 3:
                    tokens = int(inp[0].shape[-1])
                    _mark(module, n, tokens, "attn_in")

            handles.append(mod.register_forward_hook(hook))
        elif isinstance(mod, QuantModule) and leaf in ("qkv", "proj_out"):
            def hook(module, inp, out, n=name, leaf=leaf):
                if torch.is_tensor(out) and out.ndim == 3:
                    _mark(module, n, int(out.shape[-1]), leaf)
                elif inp and torch.is_tensor(inp[0]) and inp[0].ndim == 3:
                    _mark(module, n, int(inp[0].shape[-1]), leaf)

            handles.append(mod.register_forward_hook(hook))

    with torch.no_grad():
        _ = qnn(x, t)
    for h in handles:
        h.remove()

    if skipped:
        logger.info(
            "跳过 %d 个大内存注意力层的重建（token>=%d，约 >=%dx%d）：",
            len(skipped),
            token_thr,
            min_spatial_side,
            min_spatial_side,
        )
        for name, kind, tokens in skipped:
            side = int(round(tokens ** 0.5))
            logger.info("  - %s (%s, tokens=%d ≈ %dx%d)", name, kind, tokens, side, side)
    else:
        logger.info("未发现需跳过的大内存注意力层")
    return skipped


def _walk_brecq(qnn, root, kwargs):
    for name, module in root.named_children():
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction:
                logger.info("跳过 QuantModule 重建: %s", name)
                continue
            layer_reconstruction(qnn, module, **kwargs)
        elif isinstance(module, BaseQuantBlock):
            if module.ignore_reconstruction:
                logger.info("跳过 BaseQuantBlock 重建: %s", name)
                continue
            block_reconstruction(qnn, module, **kwargs)
        else:
            _walk_brecq(qnn, module, kwargs)


def _finalize_qnn_params(qnn, quant_act):
    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta = nn.Parameter(m.delta)
        elif isinstance(m, UniformAffineQuantizer) and quant_act:
            if m.zero_point is not None:
                if not torch.is_tensor(m.zero_point):
                    m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                else:
                    m.zero_point = nn.Parameter(m.zero_point)


def _cpu_state_dict(qnn):
    return {
        k: v.detach().cpu() if torch.is_tensor(v) else v
        for k, v in qnn.state_dict().items()
    }


def _save_stage_ckpt(qnn, quant_act, path, stage_tag):
    """固化量化参数后保存单阶段 state_dict（对齐 CIFAR ckpt_stage*.pth）。"""
    _finalize_qnn_params(qnn, quant_act)
    torch.save(_cpu_state_dict(qnn), path)
    logger.info("[%s] 保存量化模型: %s", stage_tag, path)


def _save_combined_two_stage_ckpt(opt, stage1_path, stage2_path, stage2_state):
    """将两阶段权重合并保存为 ckpt_two_stage.pth。"""
    stage1_sd = torch.load(stage1_path, map_location="cpu")
    stage2_sd = {
        k: v.detach().cpu() if torch.is_tensor(v) else v
        for k, v in stage2_state.items()
    }
    g0_path = getattr(opt, "cali_data_path_group0", None)
    g1_path = getattr(opt, "cali_data_path_group1", None)
    combined = {
        "stage1_group0": stage1_sd,
        "stage2_group1": stage2_sd,
        "meta": {
            "weight_bit": opt.weight_bit,
            "act_bit": opt.act_bit,
            "quant_act": bool(opt.quant_act),
            "a_sym": bool(opt.a_sym),
            "split": bool(getattr(opt, "split", False)),
            "cali_data_path_group0": g0_path,
            "cali_data_path_group1": g1_path,
            "cali_split": "allst_first_half_g0_second_half_g1",
            "max_avg_json": getattr(opt, "max_avg_json", None),
            "stage1_ckpt": os.path.basename(stage1_path),
            "stage2_ckpt": os.path.basename(stage2_path),
        },
    }
    out = os.path.join(os.path.dirname(stage1_path), "ckpt_two_stage.pth")
    torch.save(combined, out)
    logger.info(
        "已合并保存两阶段权重: %s (keys: stage1_group0, stage2_group1, meta)", out
    )
    return out


def _cali_init_bs(n, preferred=1):
    """权重量化 / 激活初始化用的安全 batch（8GB 上避免硬编码 8/64 OOM）。"""
    return max(1, min(int(preferred), int(n)))


def _ldm_stage_brecq(fp_unet, opt, cali_data, cali_xs, cali_ts, max_avg_stats, group_id):
    """单阶段：权重 BRECQ + 组 group_id 激活 BRECQ；返回 (qnn, fp_unet)。"""
    a_scale_method = "mse" if not opt.a_min_max else "max"
    wq_params = {"n_bits": opt.weight_bit, "channel_wise": True, "scale_method": "max"}
    aq_params = {
        "n_bits": opt.act_bit,
        "symmetric": opt.a_sym,
        "channel_wise": False,
        "scale_method": a_scale_method,
        "leaf_param": opt.quant_act,
    }
    qnn = QuantModel(
        model=fp_unet,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        sm_abit=opt.sm_abit,
    )
    qnn.cuda()
    qnn.eval()

    cali_bs = max(1, int(opt.cali_batch_size))
    init_bs = _cali_init_bs(cali_xs.size(0), preferred=min(cali_bs, 1))
    act_init_bs = _cali_init_bs(cali_xs.size(0), preferred=cali_bs)

    kwargs_w = dict(
        cali_data=cali_data,
        batch_size=cali_bs,
        iters=opt.cali_iters,
        weight=0.01,
        asym=True,
        b_range=(20, 2),
        warmup=0.2,
        act_quant=False,
        opt_mode="mse",
    )
    logger.info("组 %d：权重量化初始化 (bs=%d)", group_id, init_bs)
    qnn.set_quant_state(True, False)
    _skip_memory_heavy_attn_recon(
        qnn,
        cali_xs[:init_bs].cuda(),
        cali_ts[:init_bs].cuda(),
        min_spatial_side=16,
    )
    logger.info("组 %d：权重 BRECQ (cali_batch_size=%d)", group_id, cali_bs)
    _walk_brecq(qnn, qnn, kwargs_w)
    qnn.set_quant_state(weight_quant=True, act_quant=False)

    if opt.quant_act:
        logger.info("组 %d：激活量化（TMMA 初始化, init_bs=%d）", group_id, act_init_bs)
        qnn.set_quant_state(True, True)
        with torch.no_grad():
            _ = qnn(cali_xs[:act_init_bs].cuda(), cali_ts[:act_init_bs].cuda())
        if opt.running_stat:
            qnn.set_running_stat(True)
            for i in range(int(cali_xs.size(0) / act_init_bs)):
                _ = qnn(
                    cali_xs[i * act_init_bs : (i + 1) * act_init_bs].cuda(),
                    cali_ts[i * act_init_bs : (i + 1) * act_init_bs].cuda(),
                )
            qnn.set_running_stat(False)
        group_stats = {
            layer: groups.get(group_id, None)
            for layer, groups in max_avg_stats.items()
        }
        init_summary = initialize_tmma_scales(qnn, group_stats)
        logger.info(
            "组 %d：激活 BRECQ 前 TMMA 初始化 %d 个量化器",
            group_id,
            init_summary["matched_quantizers"],
        )
        kwargs_a = dict(
            cali_data=cali_data,
            batch_size=cali_bs,
            iters=opt.cali_iters_a,
            act_quant=True,
            opt_mode="mse",
            lr=opt.cali_lr,
            p=opt.cali_p,
        )
        logger.info("组 %d：激活 BRECQ", group_id)
        _walk_brecq(qnn, qnn, kwargs_a)
        qnn.set_quant_state(weight_quant=True, act_quant=True)

    return qnn


@torch.no_grad()
def _first_half_sampling_ldm(
    qnn, num_samples, betas, num_timesteps, custom_steps, skip_type, eta,
    channels, image_size, device, batch_size,
):
    half_seq = _half_ddim_seq(num_timesteps, custom_steps, skip_type, first_half=True)
    logger.info("前半段 DDIM 子序列: %s -> %s（%d 步）", half_seq[0], half_seq[-1], len(half_seq))
    chunks = []
    for i in tqdm(range(0, num_samples, batch_size), desc="前半段采样"):
        bs = min(batch_size, num_samples - i)
        x = torch.randn(bs, channels, image_size, image_size, device=device)
        xs, _ = generalized_steps(x, half_seq, qnn, betas, eta=eta)
        chunks.append(xs[-1].cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _second_half_sampling_ldm(
    qnn, intermediate, betas, num_timesteps, custom_steps, skip_type, eta, device, batch_size,
):
    half_seq = _half_ddim_seq(num_timesteps, custom_steps, skip_type, first_half=False)
    logger.info("后半段 DDIM 子序列: %s -> %s（%d 步）", half_seq[0], half_seq[-1], len(half_seq))
    chunks = []
    for i in tqdm(range(0, intermediate.shape[0], batch_size), desc="后半段采样"):
        end = min(i + batch_size, intermediate.shape[0])
        x = intermediate[i:end].to(device)
        xs, _ = generalized_steps(x, half_seq, qnn, betas, eta=eta)
        chunks.append(xs[-1].cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _save_ldm_decoded_png(ldm_model, latents, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(latents.shape[0]):
        z = latents[i : i + 1].cuda()
        img = ldm_model.decode_first_stage(z)
        tvu.save_image(img, os.path.join(out_dir, f"{i}.png"))


def two_stage_ldm_quantization(ldm_model, opt, config, logdir, imglogdir):
    """
    两阶段 TMMA：组0 BRECQ 存 ckpt → 前半采样 → 组1 BRECQ 存 ckpt →
    合并 ckpt_two_stage.pth → 后半采样 → VAE 解码。
    权重保存对齐 two_stage_quantized_sampling.py。
    """
    if not (opt.max_avg_json and str(opt.max_avg_json).strip()):
        raise ValueError("两阶段模式需要 --max_avg_json")
    if opt.resume or opt.resume_w:
        raise ValueError("两阶段模式不支持 --resume / --resume_w")

    recon_only = bool(getattr(opt, "recon_only", False))
    if recon_only:
        logger.info("开始 LDM 两阶段重建（--recon_only：不采样、不解码）")
    else:
        logger.info("开始 LDM 两阶段量化采样")
    max_avg_stats = load_max_avg_statistics(opt.max_avg_json.strip())
    g0_path, g1_path = _resolve_two_stage_cali_paths(opt)
    opt.cali_data_path_group0 = g0_path
    opt.cali_data_path_group1 = g1_path
    cali_g0 = _load_group_slice(g0_path, "stage1_group0", 0, opt.cali_st, opt.cali_n)

    betas = ldm_model.betas.float().cuda()
    num_timesteps = int(ldm_model.num_timesteps)
    channels = config.model.params.channels
    image_size = config.model.params.image_size
    device = torch.device("cuda")
    skip_type = getattr(opt, "skip_type", "quad")
    batch_size = opt.batch_size

    # ---------- 阶段 1：组 0 ----------
    logger.info("=" * 75)
    logger.info(
        "第一阶段：组 0 max_avg + BRECQ%s",
        "" if recon_only else " + 前半采样",
    )
    logger.info("=" * 75)
    fp_unet1 = _clone_ldm_unet(ldm_model, config)
    qnn_stage1 = _ldm_stage_brecq(
        fp_unet1, opt, cali_g0, cali_g0[0], cali_g0[1], max_avg_stats, 0
    )
    stage1_ckpt = os.path.join(logdir, "ckpt_stage1_group0.pth")
    _save_stage_ckpt(qnn_stage1, opt.quant_act, stage1_ckpt, "stage1_group0")
    intermediate = None
    if not recon_only:
        intermediate = _first_half_sampling_ldm(
            qnn_stage1,
            opt.n_samples,
            betas,
            num_timesteps,
            opt.custom_steps,
            skip_type,
            opt.eta,
            channels,
            image_size,
            device,
            batch_size,
        )
        torch.save(intermediate, os.path.join(logdir, "intermediate_noise_images.pt"))
    del qnn_stage1, fp_unet1, cali_g0
    torch.cuda.empty_cache()
    gc.collect()

    # ---------- 阶段 2：组 1 ----------
    logger.info("=" * 75)
    logger.info(
        "第二阶段：组 1 max_avg + BRECQ%s",
        "" if recon_only else " + 后半采样",
    )
    logger.info("=" * 75)
    cali_g1 = _load_group_slice(g1_path, "stage2_group1", 1, opt.cali_st, opt.cali_n)
    fp_unet2 = _clone_ldm_unet(ldm_model, config)
    qnn_stage2 = _ldm_stage_brecq(
        fp_unet2, opt, cali_g1, cali_g1[0], cali_g1[1], max_avg_stats, 1
    )
    stage2_ckpt = os.path.join(logdir, "ckpt_stage2_group1.pth")
    _save_stage_ckpt(qnn_stage2, opt.quant_act, stage2_ckpt, "stage2_group1")
    _save_combined_two_stage_ckpt(
        opt, stage1_ckpt, stage2_ckpt, qnn_stage2.state_dict()
    )
    if not recon_only:
        final_latents = _second_half_sampling_ldm(
            qnn_stage2,
            intermediate,
            betas,
            num_timesteps,
            opt.custom_steps,
            skip_type,
            opt.eta,
            device,
            batch_size,
        )
        torch.save(final_latents, os.path.join(logdir, "final_latents.pt"))
        _save_ldm_decoded_png(ldm_model, final_latents, imglogdir)
    else:
        logger.info("已跳过前后半采样与 VAE 解码")

    del fp_unet2
    return qnn_stage2


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    ckpt = None

    # fix random seed
    seed_everything(opt.seed)

    if not os.path.exists(opt.resume_base):
        raise ValueError("Cannot find {}".format(opt.resume_base))
    if os.path.isfile(opt.resume_base):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume_base.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume_base.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume_base
    else:
        assert os.path.isdir(opt.resume_base), f"{opt.resume_base} is not a directory"
        logdir = opt.resume_base.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    logdir = os.path.join(logdir, "samples", now)
    os.makedirs(logdir)
    log_path = os.path.join(logdir, "run.log")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    print(config)

    # logger.info(75 * "=")
    # logger.info(f"Host {os.uname()[1]}")
    # logger.info("logging to:")
    # imglogdir = os.path.join(logdir, "img")
    # numpylogdir = os.path.join(logdir, "numpy")
    import os
    import platform

    logger.info(75 * "=")

    # 获取主机名，不同操作系统使用不同方法
    if platform.system() == "Windows":
        host = platform.node()  # Windows 上使用 platform.node() 获取主机名
    else:
        host = os.uname()[1]  # 类 Unix 系统上使用 os.uname()

    logger.info(f"Host {host}")
    logger.info("logging to:")

    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    logger.info(logdir)
    logger.info(75 * "=")

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    logger.info(f"global step: {global_step}")
    logger.info("Switched to EMA weights")
    model.model_ema.store(model.model.parameters())
    model.model_ema.copy_to(model.model)

    # print(model.model)
    assert(not opt.cond)
    two_stage_done = False
    if opt.ptq:
        if opt.quant_mode == 'qdiff' and getattr(opt, "two_stage", False):
            qnn_final = two_stage_ldm_quantization(model, opt, config, logdir, imglogdir)
            model.model.diffusion_model = qnn_final
            two_stage_done = True
        elif opt.quant_mode == 'qdiff':
            a_scale_method = 'mse' if not opt.a_min_max else 'max'
            wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'max'}
            aq_params = {
                'n_bits': opt.act_bit, 'symmetric': opt.a_sym, 'channel_wise': False, 
                'scale_method': a_scale_method, 'leaf_param': opt.quant_act
            }
            if opt.resume:
                logger.info('Load with min-max quick initialization')
                wq_params['scale_method'] = 'max'
                aq_params['scale_method'] = 'max'
            if opt.resume_w:
                wq_params['scale_method'] = 'max'
            # with model.ema_scope("Quantizing", restore=False):
            qnn = QuantModel(
                model=model.model.diffusion_model, weight_quant_params=wq_params, act_quant_params=aq_params,
                sm_abit=opt.sm_abit)
            qnn.cuda()
            qnn.eval()

            if opt.resume:
                image_size = config.model.params.image_size
                channels = config.model.params.channels
                cali_data = (torch.randn(1, channels, image_size, image_size), torch.randint(0, 1000, (1,)))
                resume_cali_model(qnn, opt.cali_ckpt, cali_data, opt.quant_act, "qdiff", cond=False)
            else:
                logger.info(f"Sampling data from {opt.cali_st} timesteps for calibration")
                sample_data = torch.load(opt.cali_data_path)
                cali_data = get_train_samples(opt, sample_data)
                del(sample_data)
                gc.collect()
                logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape}")
                
                cali_xs, cali_ts = cali_data
                if opt.resume_w:
                    resume_cali_model(qnn, opt.cali_ckpt, cali_data, False, cond=False)
                else:
                    logger.info("Initializing weight quantization parameters")
                    qnn.set_quant_state(True, False) # enable weight quantization, disable act quantization
                    init_start = time.time()
                    quant_modules = [m for m in qnn.modules() if isinstance(m, QuantModule)]
                    logger.info(f"QuantModule count before init: {len(quant_modules)}")

                    hooks = []
                    visited_counter = {'count': 0}

                    if opt.init_debug_log:
                        interval = max(1, opt.init_log_interval)

                        def _make_hook(module_name):
                            def _hook(_module, _inputs):
                                visited_counter['count'] += 1
                                if visited_counter['count'] % interval == 0:
                                    logger.info(
                                        f"[init debug] visited QuantModule {visited_counter['count']} / {len(quant_modules)} "
                                        f"(latest: {module_name})"
                                    )
                            return _hook

                        for name, module in qnn.named_modules():
                            if isinstance(module, QuantModule):
                                hooks.append(module.register_forward_pre_hook(_make_hook(name)))

                    with torch.no_grad():
                        init_bs = _cali_init_bs(cali_xs.size(0), preferred=min(max(1, int(opt.cali_batch_size)), 1))
                        _ = qnn(cali_xs[:init_bs].cuda(), cali_ts[:init_bs].cuda())
                    for hook in hooks:
                        hook.remove()
                    inited_weight_quantizers = sum(
                        int(getattr(m.weight_quantizer, "inited", False))
                        for m in quant_modules
                    )
                    logger.info(
                        f"Weight init completed in {time.time() - init_start:.2f}s; "
                        f"initialized quantizers: {inited_weight_quantizers}/{len(quant_modules)} "
                        f"(init_bs={init_bs})"
                    )
                    logger.info("Initializing has done!")

                # Kwargs for weight rounding calibration
                kwargs = dict(cali_data=cali_data, batch_size=opt.cali_batch_size, 
                            iters=opt.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                            warmup=0.2, act_quant=False, opt_mode='mse')

                def recon_model(model):
                    """
                    Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
                    """
                    for name, module in model.named_children():
                        logger.info(f"{name} {isinstance(module, BaseQuantBlock)}")
                        if isinstance(module, QuantModule):
                            if module.ignore_reconstruction is True:
                                logger.info('Ignore reconstruction of layer {}'.format(name))
                                continue
                            else:
                                logger.info('Reconstruction for layer {}'.format(name))
                                layer_reconstruction(qnn, module, **kwargs)
                        elif isinstance(module, BaseQuantBlock):
                            if module.ignore_reconstruction is True:
                                logger.info('Ignore reconstruction of block {}'.format(name))
                                continue
                            else:
                                logger.info('Reconstruction for block {}'.format(name))
                                block_reconstruction(qnn, module, **kwargs)
                        else:
                            recon_model(module)

                if not opt.resume_w:
                    logger.info("Doing weight calibration")
                    recon_model(qnn)
                    qnn.set_quant_state(weight_quant=True, act_quant=False)
                if opt.quant_act:
                    logger.info("UNet model")
                    logger.info(model.model)
                    logger.info("Doing activation calibration")
                    qnn.set_quant_state(True, True)
                    max_avg_stats = None
                    act_init_bs = _cali_init_bs(
                        cali_xs.size(0), preferred=max(1, int(opt.cali_batch_size))
                    )
                    if opt.max_avg_json and str(opt.max_avg_json).strip():
                        logger.info(
                            "激活 BRECQ：TMMA 初始化 group=%d",
                            opt.max_avg_group,
                        )
                        max_avg_stats = load_max_avg_statistics(opt.max_avg_json.strip())
                    with torch.no_grad():
                        _ = qnn(cali_xs[:act_init_bs].cuda(), cali_ts[:act_init_bs].cuda())
                        if opt.running_stat:
                            logger.info('Running stat for activation quantization')
                            qnn.set_running_stat(True)
                            for i in trange(int(cali_xs.size(0) / act_init_bs)):
                                _ = qnn(
                                    cali_xs[i * act_init_bs:(i + 1) * act_init_bs].cuda(),
                                    cali_ts[i * act_init_bs:(i + 1) * act_init_bs].cuda(),
                                )
                            qnn.set_running_stat(False)
                        if max_avg_stats is not None:
                            group_stats = {
                                layer: groups.get(opt.max_avg_group, None)
                                for layer, groups in max_avg_stats.items()
                            }
                            init_summary = initialize_tmma_scales(qnn, group_stats)
                            logger.info(
                                "激活 BRECQ 前：组 %d TMMA 初始化 %d 个量化器",
                                opt.max_avg_group,
                                init_summary["matched_quantizers"],
                            )
                    kwargs = dict(
                        cali_data=cali_data, batch_size=opt.cali_batch_size,
                        iters=opt.cali_iters_a, act_quant=True,
                        opt_mode='mse', lr=opt.cali_lr, p=opt.cali_p)
                    recon_model(qnn)
                    qnn.set_quant_state(weight_quant=True, act_quant=True)   

                logger.info("Saving calibrated quantized UNet model")
                for m in qnn.model.modules():
                    if isinstance(m, AdaRoundQuantizer):
                        m.zero_point = nn.Parameter(m.zero_point)
                        m.delta = nn.Parameter(m.delta)
                    elif isinstance(m, UniformAffineQuantizer) and opt.quant_act:
                        if m.zero_point is not None:
                            if not torch.is_tensor(m.zero_point):
                                m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                            else:
                                m.zero_point = nn.Parameter(m.zero_point)
                torch.save(qnn.state_dict(), os.path.join(logdir, "ckpt.pth"))         

            model.model.diffusion_model = qnn


    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    if opt.verbose:
        print(sampling_conf)
        logger.info("first_stage_model")
        logger.info(model.first_stage_model)
        logger.info("UNet model")
        logger.info(model.model)


    if not two_stage_done:
        run(model, imglogdir, eta=opt.eta,
            vanilla=opt.vanilla_sample, n_samples=opt.n_samples, custom_steps=opt.custom_steps,
            batch_size=opt.batch_size, nplog=numpylogdir, dpm=opt.dpm)
    elif getattr(opt, "recon_only", False):
        logger.info("两阶段仅重建，已保存权重，跳过采样与解码")
    else:
        logger.info("两阶段已生成图像至 %s，跳过常规 run() 采样", imglogdir)

    logger.info("done.")

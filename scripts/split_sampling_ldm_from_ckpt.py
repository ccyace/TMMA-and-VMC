"""
用已有 LDM 量化 UNet ckpt 做分段 DDIM 推理（同一模型，不重新 BRECQ）。

对齐 scripts/split_sampling_from_ckpt.py，但对象是潜空间 LDM：
  - 前半 / 后半均在 latent 上跑 generalized_steps
  - 后半结束后用 FP VAE decode 存 PNG

输出（默认写到 ckpt 同级目录；可用 --outdir 一次性指定）:
  <outdir 或 ckpt 目录>/
    ├── intermediate_noise_latents.pt
    ├── split_sampling_ldm.log
    └── images/
        ├── 0.png
        └── ...

示例（Church W8A8，自定义输出目录）:
  $env:PYTHONPATH="C:\\Users\\ASUS\\Desktop\\ODE-scale\\src\\taming-transformers"
  python scripts/split_sampling_ldm_from_ckpt.py ^
    -r models/ldm/lsun_churches256/model.ckpt ^
    --cali_ckpt church_LDM_sample/add_w8a8_9.81/w8a8_LDM_church.pth ^
    --outdir church_LDM_sample/add_w8a8_9.81/run_2k ^
    --weight_bit 8 --act_bit 8 --quant_act --a_sym ^
    --ddim_steps 400 --skip_type quad --eta 0 ^
    --max_images 2000 --batch_size 2 --fresh_images
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime

import numpy as np
import torch
import torchvision.utils as tvu
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_TAMING_CANDIDATES = [
    os.path.join(os.path.dirname(_REPO), "ODE-scale", "src", "taming-transformers"),
    os.path.join(_REPO, "src", "taming-transformers"),
]
for _p in _TAMING_CANDIDATES:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from ddim.functions.denoising import compute_alpha, generalized_steps
from ldm.util import instantiate_from_config
from qdiff import QuantModel
from qdiff.utils import resume_cali_model

logger = logging.getLogger(__name__)


def build_ddim_seq(num_timesteps: int, ddim_steps: int, skip_type: str):
    if skip_type == "uniform":
        skip = num_timesteps // ddim_steps
        seq = list(range(0, num_timesteps, skip))
    elif skip_type == "quad":
        seq = (np.linspace(0, np.sqrt(num_timesteps * 0.8), ddim_steps) ** 2)
        seq = [int(s) for s in list(seq)]
        seq = sorted(list(set(seq)))
    else:
        raise ValueError(f"不支持的 skip_type: {skip_type}")
    return seq


def generalized_steps_limited(x, seq, model, b, max_steps, **kwargs):
    """只跑完整 DDIM 序列的前 max_steps 次去噪迭代。"""
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for step_idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            if step_idx >= max_steps:
                break
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.cpu())
            c1 = kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.cpu())
    return xs, x0_preds


def build_split_plan(num_timesteps, ddim_steps, skip_type):
    full_seq = build_ddim_seq(num_timesteps, ddim_steps, skip_type)
    split_idx = len(full_seq) // 2
    second_half_seq = full_seq[:split_idx]
    reverse_seq = list(reversed(full_seq))
    return {
        "full_seq": full_seq,
        "split_idx": split_idx,
        "second_half_seq": second_half_seq,
        "first_start_t": reverse_seq[0],
        "first_end_t": reverse_seq[split_idx],
        "second_start_t": reverse_seq[split_idx],
        "second_end_t": reverse_seq[-1],
        "n_steps": len(full_seq),
    }


def resolve_paths(args):
    args.cali_ckpt = os.path.abspath(args.cali_ckpt)
    if not os.path.isfile(args.cali_ckpt):
        raise FileNotFoundError(f"找不到量化 ckpt: {args.cali_ckpt}")
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"找不到 FP LDM ckpt: {args.resume}")

    ckpt_dir = os.path.dirname(args.cali_ckpt)

    # --outdir: 图 / 中间态 / log 的统一根目录；未指定则仍用 ckpt 同级目录
    if getattr(args, "outdir", None):
        out_root = os.path.abspath(args.outdir)
    else:
        out_root = ckpt_dir
    os.makedirs(out_root, exist_ok=True)
    args.logdir = out_root

    default_images = os.path.join(out_root, "images")
    if args.image_folder:
        args.image_folder = os.path.abspath(args.image_folder)
    elif args.fresh_images and not getattr(args, "outdir", None):
        # 未指定 outdir 时，fresh 仍在 ckpt 旁打时间戳子目录，避免覆盖旧图
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.image_folder = os.path.join(ckpt_dir, f"images_split_{stamp}")
    else:
        args.image_folder = default_images
    os.makedirs(args.image_folder, exist_ok=True)

    # 中间态默认与 PNG 同属 out_root；可用 --intermediate_path 单独覆盖
    if args.intermediate_path is None:
        args.intermediate_path = os.path.join(out_root, "intermediate_noise_latents.pt")
    else:
        args.intermediate_path = os.path.abspath(args.intermediate_path)
        inter_dir = os.path.dirname(args.intermediate_path)
        if inter_dir:
            os.makedirs(inter_dir, exist_ok=True)
    return args


def next_image_index(image_folder):
    pattern = re.compile(r"^(\d+)\.png$")
    max_id = -1
    for name in os.listdir(image_folder):
        m = pattern.match(name)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def load_fp_ldm(resume_ckpt: str, config_path: str, device):
    if config_path:
        config = OmegaConf.load(config_path)
    else:
        cfg_dir = os.path.dirname(os.path.abspath(resume_ckpt))
        config = OmegaConf.load(os.path.join(cfg_dir, "config.yaml"))
    pl_sd = torch.load(resume_ckpt, map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    if getattr(model, "use_ema", False) and hasattr(model, "model_ema"):
        model.model_ema.store(model.model.parameters())
        model.model_ema.copy_to(model.model)
        logger.info("Switched to EMA weights")
    return model, config


def load_quantized_unet(ldm_model, config, args, device):
    unet = instantiate_from_config(config.model.params.unet_config)
    unet.load_state_dict(ldm_model.model.diffusion_model.state_dict(), strict=False)
    unet.to(device)
    unet.eval()

    wq_params = {"n_bits": args.weight_bit, "channel_wise": True, "scale_method": "max"}
    aq_params = {
        "n_bits": args.act_bit,
        "symmetric": args.a_sym,
        "channel_wise": False,
        "scale_method": "max",
        "leaf_param": args.quant_act,
    }
    qnn = QuantModel(
        model=unet,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        sm_abit=args.sm_abit,
    )
    qnn.to(device)
    qnn.eval()

    channels = config.model.params.channels
    image_size = config.model.params.image_size
    cali_data = (
        torch.randn(1, channels, image_size, image_size),
        torch.randint(0, 1000, (1,)),
    )
    logger.info("Loading quantized UNet ckpt: %s", args.cali_ckpt)
    resume_cali_model(qnn, args.cali_ckpt, cali_data, args.quant_act, "qdiff", cond=False)
    qnn.eval()
    return qnn


def run_partial_sampling(model, start_x, seq, betas, batch_size, device, eta, desc, max_steps=None):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in tqdm(range(0, start_x.shape[0], batch_size), desc=desc):
            x = start_x[i : i + batch_size].to(device)
            if max_steps is None:
                xs, _ = generalized_steps(x, seq, model, betas, eta=eta)
            else:
                xs, _ = generalized_steps_limited(x, seq, model, betas, max_steps, eta=eta)
            outputs.append(xs[-1].cpu())
    return torch.cat(outputs, dim=0)


def run_first_half(qnn, args, config, betas, split_plan, device):
    if os.path.exists(args.intermediate_path) and not args.overwrite_intermediate:
        logger.info("中间态已存在，跳过前半段: %s", args.intermediate_path)
        logger.info("若需重跑，请加 --overwrite_intermediate")
        return torch.load(args.intermediate_path, map_location="cpu")

    channels = config.model.params.channels
    size = config.model.params.image_size
    all_noise = torch.randn(args.max_images, channels, size, size)
    intermediate = run_partial_sampling(
        qnn,
        all_noise,
        split_plan["full_seq"],
        betas,
        args.batch_size,
        device,
        args.eta,
        "前半段潜空间采样",
        max_steps=split_plan["split_idx"],
    )
    torch.save(intermediate, args.intermediate_path)
    logger.info("已保存中间 latent: %s shape=%s", args.intermediate_path, tuple(intermediate.shape))
    return intermediate


@torch.no_grad()
def run_second_half(qnn, ldm_model, args, betas, split_plan, device, intermediate):
    start_id = 0 if args.fresh_images else next_image_index(args.image_folder)
    if start_id > 0:
        logger.info("检测到已有 PNG，从编号 %d 续写 -> %s", start_id, args.image_folder)

    total = min(intermediate.shape[0], args.max_images)
    if start_id >= total:
        logger.info("已有图像数 %d >= max_images %d，跳过后半段", start_id, total)
        return

    logger.info("后半段 PNG 输出: %s  (编号 %d ~ %d)", args.image_folder, start_id, total - 1)
    for i in tqdm(range(start_id, total, args.batch_size), desc="后半段采样+解码"):
        end = min(i + args.batch_size, total)
        x = intermediate[i:end].to(device)
        xs, _ = generalized_steps(x, split_plan["second_half_seq"], qnn, betas, eta=args.eta)
        z = xs[-1]
        imgs = ldm_model.decode_first_stage(z)
        for j in range(imgs.shape[0]):
            tvu.save_image(imgs[j], os.path.join(args.image_folder, f"{i + j}.png"))
    logger.info("后半段完成，写入 %d 张 PNG -> %s", total - start_id, args.image_folder)


def parse_args():
    p = argparse.ArgumentParser(description="LDM: 用已有量化 UNet ckpt 做分段 DDIM 采样")
    p.add_argument(
        "-r",
        "--resume",
        type=str,
        default=os.path.join(_REPO, "models", "ldm", "lsun_churches256", "model.ckpt"),
        help="FP LDM Lightning ckpt（含 VAE）",
    )
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="LDM config.yaml；默认与 -r 同目录",
    )
    p.add_argument("--cali_ckpt", type=str, required=True, help="已量化 UNet state_dict .pth")
    p.add_argument("--weight_bit", type=int, default=8)
    p.add_argument("--act_bit", type=int, default=8)
    p.add_argument("--quant_act", action="store_true")
    p.add_argument("--a_sym", action="store_true")
    p.add_argument("--sm_abit", type=int, default=8)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--max_images", type=int, default=8)
    p.add_argument("--ddim_steps", type=int, default=400, help="对齐 sample_diffusion_ldm -c")
    p.add_argument("--skip_type", type=str, default="quad", choices=["quad", "uniform"])
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="自定义输出根目录：写入 images/、intermediate_noise_latents.pt、split_sampling_ldm.log",
    )
    p.add_argument("--intermediate_path", type=str, default=None,
                   help="中间 latent 路径；默认 <outdir或ckpt目录>/intermediate_noise_latents.pt")
    p.add_argument("--overwrite_intermediate", action="store_true")
    p.add_argument("--first_half_only", action="store_true")
    p.add_argument("--second_half_only", action="store_true")
    p.add_argument("--image_folder", type=str, default=None,
                   help="PNG 目录；默认 <outdir或ckpt目录>/images")
    p.add_argument("--fresh_images", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args = resolve_paths(args)

    log_path = os.path.join(args.logdir, "split_sampling_ldm.log")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ldm_model, config = load_fp_ldm(args.resume, args.config, device)
    betas = ldm_model.betas.float().to(device)
    num_timesteps = int(ldm_model.num_timesteps)
    split_plan = build_split_plan(num_timesteps, args.ddim_steps, args.skip_type)

    logger.info("=" * 60)
    logger.info("FP LDM           : %s", args.resume)
    logger.info("quant UNet ckpt  : %s", args.cali_ckpt)
    logger.info("outdir / logdir  : %s", args.logdir)
    logger.info("intermediate     : %s", args.intermediate_path)
    logger.info("image output     : %s", args.image_folder)
    logger.info(
        "DDIM ~%d-step(%s, actual %d): 前半 %d 步 (%d->%d), 后半 %d 步 (%d->%d)",
        args.ddim_steps,
        args.skip_type,
        split_plan["n_steps"],
        split_plan["split_idx"],
        split_plan["first_start_t"],
        split_plan["first_end_t"],
        split_plan["split_idx"],
        split_plan["second_start_t"],
        split_plan["second_end_t"],
    )
    logger.info("=" * 60)

    qnn = load_quantized_unet(ldm_model, config, args, device)

    if args.second_half_only:
        if not os.path.exists(args.intermediate_path):
            raise FileNotFoundError(f"未找到中间态: {args.intermediate_path}")
        intermediate = torch.load(args.intermediate_path, map_location="cpu")
        run_second_half(qnn, ldm_model, args, betas, split_plan, device, intermediate)
        return

    intermediate = run_first_half(qnn, args, config, betas, split_plan, device)
    if args.first_half_only:
        logger.info("first_half_only 完成")
        return

    run_second_half(qnn, ldm_model, args, betas, split_plan, device, intermediate)


if __name__ == "__main__":
    main()

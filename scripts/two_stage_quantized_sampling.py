import argparse, os, gc, glob, datetime, yaml
import json
import logging
import math
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.cuda import amp
from pytorch_lightning import seed_everything
import ddim.models
from ddim import models
from ddim.models.diffusion import Model
from ddim.datasets import inverse_data_transform
from ddim.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from ddim.functions.ckpt_util import get_ckpt_path
from ddim.functions.denoising import generalized_steps, compute_alpha

import torchvision.utils as tvu

from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock,
    block_reconstruction, layer_reconstruction,
)
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import resume_cali_model, get_train_samples
from qdiff.init_perturbation import (
    activation_quantization_probe,
    initialize_tmma_interpolation,
    initialize_tmma_scales,
    save_json as save_init_probe_json,
    snapshot_activation_scales,
    two_stage_trajectory_probe,
)
from qdiff.recon_metrics import ReconPlotContext, run_recon_model

logger = logging.getLogger(__name__)


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def load_max_avg_statistics(json_path: str) -> dict:
    """
    从JSON文件加载max_avg统计信息。

    兼容两种格式：
      1) {"layers": {layer: {"0": {"max_avg": ...}, "1": {...}}, ...}}
      2) {layer: {"0": {"max_avg": ...}, "1": {...}}, ...}  # layerwise_activation_statistics.json
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        statistics = json.load(f)

    if isinstance(statistics, dict) and "layers" in statistics:
        layer_blob = statistics["layers"]
    else:
        layer_blob = {
            k: v
            for k, v in statistics.items()
            if isinstance(v, dict) and ("0" in v or "1" in v)
        }

    max_avg_stats = {}
    for layer_name, groups in layer_blob.items():
        if not isinstance(groups, dict):
            continue
        max_avg_stats[layer_name] = {
            0: groups.get('0', {}).get('max_avg', None) if isinstance(groups.get('0'), dict) else None,
            1: groups.get('1', {}).get('max_avg', None) if isinstance(groups.get('1'), dict) else None,
        }

    logger.info(f"从 {json_path} 加载了 {len(max_avg_stats)} 层的max_avg统计信息")
    return max_avg_stats


def apply_max_avg_clipping_to_fp_model(model: nn.Module, max_avg_stats: dict, group_id: int):
    """
    为全精度模型添加max_avg截断hook，用于在BRECQ重建时截断激活
    
    Args:
        model: 全精度模型
        max_avg_stats: max_avg统计信息字典 {layer_name: {0: max_avg_group0, 1: max_avg_group1}}
        group_id: 组ID (0 或 1)
    
    Returns:
        hooks: 注册的hook列表，用于后续移除
    """
    hooks = []
    
    def make_clipping_hook(layer_name, max_avg_val):
        """创建截断hook，返回修改后的output"""
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                # 使用绝对值截断（对称截断），因为激活值可能有正有负
                output_abs = output.abs()
                output_abs_clipped = torch.clamp(output_abs, 0.0, max_avg_val)
                # 恢复符号
                output = torch.sign(output) * output_abs_clipped
            elif isinstance(output, (list, tuple)):
                # 如果output是tuple/list，递归处理每个元素
                output = type(output)([
                    torch.sign(item) * torch.clamp(item.abs(), 0.0, max_avg_val) if isinstance(item, torch.Tensor) else item
                    for item in output
                ])
            return output
        return hook_fn
    
    # 为每一层找到对应的max_avg值并注册hook
    for name, module in model.named_modules():
        # 只对叶子模块（实际计算层）添加hook
        if len(list(module.children())) == 0:
            # 查找对应的层名
            layer_key = None
            # 方法1: 直接匹配完整路径
            if name in max_avg_stats:
                layer_key = name
            else:
                # 方法2: 提取层的关键部分
                for layer_name in max_avg_stats.keys():
                    if layer_name in name or name.endswith(layer_name.split('.')[-1]):
                        layer_key = layer_name
                        break
            
            if layer_key and max_avg_stats[layer_key][group_id] is not None:
                max_avg = max_avg_stats[layer_key][group_id]
                if max_avg > 0:
                    hook = module.register_forward_hook(make_clipping_hook(name, max_avg))
                    hooks.append((name, hook, max_avg, layer_key))
                    if len(hooks) <= 5:  # 只打印前5个
                        logger.debug(f"为层 {name} (匹配 {layer_key}) 注册max_avg截断hook: max_avg={max_avg:.4f}")
    
    if hooks:
        logger.info(f"为组{group_id}注册了 {len(hooks)} 个全精度模型激活截断hook")
        logger.info(f"  示例: {hooks[0][0]} (匹配 {hooks[0][3]}) -> max_avg={hooks[0][2]:.4f}")
    else:
        logger.warning(f"未找到任何层匹配max_avg统计信息")
    
    return hooks


def remove_hooks(hooks):
    """移除所有注册的hook"""
    for name, hook, max_avg, layer_key in hooks:
        hook.remove()
    logger.info(f"移除了 {len(hooks)} 个hook")


def set_activation_truncation_from_max_avg(qnn: nn.Module, max_avg_stats: dict, group_id: int):
    """
    为量化模型设置基于max_avg的激活截断阈值
    
    Args:
        qnn: 量化模型
        max_avg_stats: max_avg统计信息字典
        group_id: 组ID (0 或 1)
    """
    logger.info(f"为组{group_id}设置激活截断阈值（基于max_avg）")
    
    set_count = 0
    missing_count = 0
    max_avg_values = []
    
    for name, module in qnn.named_modules():
        # 检查是否是激活量化器
        if isinstance(module, UniformAffineQuantizer):
            # 查找对应的层名
            layer_key = None
            for layer_name in max_avg_stats.keys():
                if layer_name in name:
                    layer_key = layer_name
                    break
            
            if layer_key and max_avg_stats[layer_key][group_id] is not None:
                max_avg = max_avg_stats[layer_key][group_id]
                if max_avg <= 0:
                    logger.warning(f"层 {name}: max_avg值异常（{max_avg:.4f} <= 0），跳过")
                    continue
                if max_avg > 1000:
                    logger.warning(f"层 {name}: max_avg值异常大（{max_avg:.4f} > 1000），可能需要检查")
                
                # 存储max_avg值到量化器中
                if not hasattr(module, 'truncation_e_y'):
                    module.truncation_e_y = None
                module.truncation_e_y = max_avg
                max_avg_values.append((layer_key, max_avg))
                set_count += 1
            else:
                missing_count += 1
                if missing_count <= 5:
                    logger.debug(f"未找到层 {name} 的max_avg统计信息")
        
        elif isinstance(module, QuantModule):
            if hasattr(module, 'act_quantizer') and isinstance(module.act_quantizer, UniformAffineQuantizer):
                layer_key = None
                for layer_name in max_avg_stats.keys():
                    if layer_name in name:
                        layer_key = layer_name
                        break
                
                if layer_key and max_avg_stats[layer_key][group_id] is not None:
                    max_avg = max_avg_stats[layer_key][group_id]
                    if max_avg > 0:
                        if not hasattr(module.act_quantizer, 'truncation_e_y'):
                            module.act_quantizer.truncation_e_y = None
                        module.act_quantizer.truncation_e_y = max_avg
                        max_avg_values.append((layer_key, max_avg))
                        set_count += 1
    
    if max_avg_values:
        max_avg_array = np.array([v[1] for v in max_avg_values])
        logger.info(f"max_avg统计: min={max_avg_array.min():.4f}, max={max_avg_array.max():.4f}, "
                   f"mean={max_avg_array.mean():.4f}")
    
    logger.info(f"为 {set_count} 个量化器设置了组{group_id}的max_avg截断阈值")


def wrap_activation_quantizer_forward(quantizer: UniformAffineQuantizer):
    """
    包装UniformAffineQuantizer的forward方法，添加max_avg截断
    """
    if hasattr(quantizer, '_original_forward_wrapped'):
        return  # 已经包装过了
    
    original_forward = quantizer.forward
    original_init_quantization_scale = quantizer.init_quantization_scale
    
    def forward_with_max_avg_clipping(x: torch.Tensor):
        if hasattr(quantizer, 'truncation_e_y') and quantizer.truncation_e_y is not None:
            max_avg = quantizer.truncation_e_y
            if quantizer.sym:
                x = torch.clamp(x, -max_avg, max_avg)
            else:
                x = torch.clamp(x, 0.0, max_avg)
        return original_forward(x)
    
    def init_quantization_scale_with_max_avg(x: torch.Tensor, channel_wise: bool = False):
        if hasattr(quantizer, 'truncation_e_y') and quantizer.truncation_e_y is not None:
            max_avg = quantizer.truncation_e_y
            if quantizer.sym:
                x = torch.clamp(x, -max_avg, max_avg)
            else:
                x = torch.clamp(x, 0.0, max_avg)
        return original_init_quantization_scale(x, channel_wise)
    
    quantizer.forward = forward_with_max_avg_clipping
    quantizer.init_quantization_scale = init_quantization_scale_with_max_avg
    quantizer._original_forward_wrapped = True


def apply_max_avg_clipping_to_model(qnn: nn.Module):
    """
    为模型中的所有激活量化器应用max_avg截断包装
    """
    for name, module in qnn.named_modules():
        if isinstance(module, UniformAffineQuantizer):
            wrap_activation_quantizer_forward(module)
        elif isinstance(module, QuantModule):
            if hasattr(module, 'act_quantizer') and isinstance(module.act_quantizer, UniformAffineQuantizer):
                wrap_activation_quantizer_forward(module.act_quantizer)


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        config.split_shortcut = self.args.split
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float()
        self.betas = self.betas.to(self.device)
        betas = self.betas
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        model = Model(self.config)

        # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        if self.config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif self.config.data.dataset == "LSUN":
            name = f"lsun_{self.config.data.category}"
        else:
            raise ValueError
        ckpt = get_ckpt_path(f"ema_{name}")
        logger.info("Loading checkpoint {}".format(ckpt))
        model.load_state_dict(torch.load(ckpt, map_location=self.device))

        #权重加上补偿量
        delta_path = self.args.delta_ckpt  # <-- 新增参数，传 delta 权重路径
        if os.path.exists(delta_path):
            logger.info(f"Loading delta weights from {delta_path}")
            delta_dict = torch.load(delta_path, map_location=self.device)

            # for name, module in model.named_modules():
            #     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            #         if name in delta_dict:
            #             delta = delta_dict[name].to(module.weight.device)
            #             logger.info(f"Applying delta to layer {name}")
            #             with torch.no_grad():
            #                 module.weight += delta
            #         else:
            #             logger.warning(f"No delta found for layer {name}, skipping.")

        else:
            logger.warning(f"Delta weights file {delta_path} not found. Skipping delta compensation.")

        model.to(self.device)
        model.eval()
        assert(self.args.cond == False)
        
        # 检查是否启用两阶段量化模式
        if self.args.two_stage and self.args.ptq:
            logger.info("=" * 75)
            logger.info("两阶段量化模式")
            logger.info("=" * 75)
            self.two_stage_quantized_sampling(model)
            return
        
        if self.args.ptq:
            if self.args.quant_mode == 'qdiff':
                wq_params = {'n_bits': args.weight_bit, 'channel_wise': True, 'scale_method': 'max'}
                aq_params = {'n_bits': args.act_bit, 'symmetric': args.a_sym, 'channel_wise': False, 'scale_method': 'max', 'leaf_param': args.quant_act}
                if self.args.resume:
                    logger.info('Load with min-max quick initialization')
                    wq_params['scale_method'] = 'max'
                    aq_params['scale_method'] = 'max'
                if self.args.resume_w:
                    wq_params['scale_method'] = 'max'
                qnn = QuantModel(
                    model=model, weight_quant_params=wq_params, act_quant_params=aq_params,
                    sm_abit=self.args.sm_abit)
                qnn.to(self.device)
                qnn.eval()

                if self.args.resume:
                    image_size = self.config.data.image_size
                    channels = self.config.data.channels
                    cali_data = (torch.randn(1, channels, image_size, image_size), torch.randint(0, 1000, (1,)))
                    resume_cali_model(qnn, args.cali_ckpt, cali_data, args.quant_act, "qdiff", cond=False)
                else:
                    logger.info(f"Sampling data from {self.args.cali_st} timesteps for calibration")
                    sample_data = torch.load(self.args.cali_data_path)
                    cali_data = get_train_samples(self.args, sample_data, custom_steps=0)
                    del(sample_data)
                    gc.collect()
                    logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape}")

                    cali_xs, cali_ts = cali_data
                    if self.args.resume_w:
                        resume_cali_model(qnn, self.args.cali_ckpt, cali_data, False, cond=False)
                    else:
                        logger.info("Initializing weight quantization parameters")
                        qnn.set_quant_state(True, False) # enable weight quantization, disable act quantization
                        _ = qnn(cali_xs[:8].cuda(), cali_ts[:8].cuda())
                        logger.info("Initializing has done!")

                    # Kwargs for weight rounding calibration
                    kwargs = dict(cali_data=cali_data, batch_size=self.args.cali_batch_size,
                                iters=self.args.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
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

                    if not self.args.resume_w:
                        logger.info("Doing weight calibration")
                        recon_model(qnn)
                        qnn.set_quant_state(weight_quant=True, act_quant=False)
                    if self.args.quant_act:
                        logger.info("UNet model")
                        logger.info(model)
                        logger.info("Doing activation calibration")
                        # Initialize activation quantization parameters
                        qnn.set_quant_state(True, True)
                        # === 如果使用单阶段量化，可以在这里加载max_avg进行截断 ===
                        # 注意：两阶段量化模式不需要在这里设置

                        with torch.no_grad():
                            inds = np.random.choice(cali_xs.shape[0], 64, replace=False)
                            # _ = qnn(cali_xs[:64].cuda(), cali_ts[:64].cuda())
                            _ = qnn(cali_xs[inds].cuda(), cali_ts[inds].cuda())

                            if self.args.running_stat:
                                logger.info('Running stat for activation quantization')
                                qnn.set_running_stat(True)
                                for i in range(int(cali_xs.size(0) / 64)):
                                    _ = qnn(
                                        (cali_xs[i * 64:(i + 1) * 64].to(self.device),
                                        cali_ts[i * 64:(i + 1) * 64].to(self.device)))
                                qnn.set_running_stat(False)

                        kwargs = dict(
                            cali_data=cali_data, iters=self.args.cali_iters_a, act_quant=True,
                            opt_mode='mse', lr=self.args.cali_lr, p=self.args.cali_p)
                        recon_model(qnn)
                        qnn.set_quant_state(weight_quant=True, act_quant=True)

                    logger.info("Saving calibrated quantized UNet model")
                    for m in qnn.model.modules():
                        if isinstance(m, AdaRoundQuantizer):
                            m.zero_point = nn.Parameter(m.zero_point)
                            m.delta = nn.Parameter(m.delta)
                        elif isinstance(m, UniformAffineQuantizer) and self.args.quant_act:
                            if m.zero_point is not None:
                                if not torch.is_tensor(m.zero_point):
                                    m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                                else:
                                    m.zero_point = nn.Parameter(m.zero_point)
                    torch.save(qnn.state_dict(), os.path.join(self.args.logdir, "ckpt.pth"))

                model = qnn

        model.to(self.device)
        if self.args.verbose:
            logger.info("quantized model")
            logger.info(model)

        model.eval()

        self.sample_fid(model)
    

    def _resolve_group_id(self, stage_tag):
        tag = str(stage_tag).lower()
        if "group0" in tag or "stage1" in tag or tag.endswith("_0"):
            return 0
        if "group1" in tag or "stage2" in tag or tag.endswith("_1"):
            return 1
        raise ValueError(f"无法从 stage_tag={stage_tag!r} 推断 group_id")

    def _calibrate_qnn_with_dataset(self, fp_model, cali_data_path, stage_tag):
        """用指定时间段校准集做 BRECQ，并保存 ckpt_{stage_tag}.pth。"""
        group_id = self._resolve_group_id(stage_tag)
        use_trunc = (
            bool(getattr(self.args, "max_avg_json", None))
            and not bool(getattr(self.args, "no_act_truncation", False))
        )
        max_avg_stats = getattr(self, "max_avg_stats", None)

        logger.info("[%s] 加载校准数据: %s (group_id=%d)", stage_tag, cali_data_path, group_id)
        if not cali_data_path or not os.path.isfile(cali_data_path):
            raise FileNotFoundError(
                f"[{stage_tag}] 找不到校准集: {cali_data_path}\n"
                "请传入 --cali_data_path_group0 / --cali_data_path_group1"
            )
        sample_data = torch.load(cali_data_path, map_location="cpu")
        cali_data = get_train_samples(self.args, sample_data, custom_steps=0)
        del sample_data
        gc.collect()
        cali_xs, cali_ts = cali_data
        logger.info("[%s] 校准数据形状: %s %s", stage_tag, tuple(cali_xs.shape), tuple(cali_ts.shape))
        # 打印时间步覆盖范围，便于确认 g0/g1
        ts_min = float(cali_ts.min())
        ts_max = float(cali_ts.max())
        logger.info("[%s] 校准时间步范围: [%.1f, %.1f]", stage_tag, ts_min, ts_max)

        wq_params = {"n_bits": self.args.weight_bit, "channel_wise": True, "scale_method": "max"}
        aq_params = {
            "n_bits": self.args.act_bit, "symmetric": self.args.a_sym, "channel_wise": False,
            "scale_method": "max", "leaf_param": self.args.quant_act,
        }
        qnn = QuantModel(
            model=fp_model, weight_quant_params=wq_params, act_quant_params=aq_params,
            sm_abit=self.args.sm_abit)
        qnn.to(self.device)
        qnn.eval()

        # Bedroom 256 等大分辨率：初始化前向 batch 不能用 8，否则 8GB 卡直接 OOM
        spatial = int(cali_xs.shape[-1]) if cali_xs.ndim >= 4 else 32
        init_bs = 1 if spatial >= 128 else min(8, int(cali_xs.shape[0]))
        qnn.set_quant_state(True, False)
        _ = qnn(cali_xs[:init_bs].to(self.device), cali_ts[:init_bs].to(self.device))
        logger.info("[%s] 权重量化初始化完成 (init_bs=%d, spatial=%d)", stage_tag, init_bs, spatial)

        def recon_model(root_model, kwargs, recon_stage):
            """Run BRECQ and, when requested, record early Adam diagnostics."""
            if not hasattr(self, "_recon_plot_context"):
                self._recon_plot_context = ReconPlotContext(self.args)
            run_recon_model(
                qnn=qnn,
                root_module=root_model,
                kwargs=kwargs,
                plot_ctx=self._recon_plot_context,
                stage_name=recon_stage,
                layer_recon_fn=layer_reconstruction,
                block_recon_fn=block_reconstruction,
            )

        kwargs_w = dict(
            cali_data=cali_data, batch_size=self.args.cali_batch_size,
            iters=self.args.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
            warmup=0.2, act_quant=False, opt_mode="mse",
        )
        logger.info("[%s] 开始权重 BRECQ 重建", stage_tag)
        recon_model(qnn, kwargs_w, "weight_" + stage_tag)
        qnn.set_quant_state(weight_quant=True, act_quant=False)

        stage_probe = {
            "stage": stage_tag,
            "group_id": group_id,
            "init_perturb_rho": getattr(self.args, "init_perturb_rho", None),
        }
        if self.args.quant_act:
            qnn.set_quant_state(True, True)
            perturb_enabled = getattr(self.args, "init_perturb_rho", None) is not None
            if perturb_enabled and not (use_trunc and max_avg_stats is not None):
                raise ValueError(
                    "--init_perturb_rho requires --max_avg_json and cannot be used with "
                    "--no_act_truncation; it needs the TMMA group statistics."
                )
            if perturb_enabled and self.args.running_stat:
                raise ValueError(
                    "Do not combine --init_perturb_rho with --running_stat: running-stat "
                    "updates would overwrite the controlled initial scale intervention."
                )
            with torch.no_grad():
                spatial = int(cali_xs.shape[-1]) if cali_xs.ndim >= 4 else 32
                init_n = 1 if spatial >= 128 else min(64, int(cali_xs.shape[0]))
                _ = qnn(cali_xs[:init_n].to(self.device), cali_ts[:init_n].to(self.device))
                if self.args.running_stat:
                    qnn.set_running_stat(True)
                    for i in range(0, cali_xs.size(0), 64):
                        _ = qnn(cali_xs[i:i + 64].to(self.device), cali_ts[i:i + 64].to(self.device))
                    qnn.set_running_stat(False)

                group_stats = None
                if use_trunc and max_avg_stats is not None:
                    group_stats = {
                        layer: groups.get(group_id, None)
                        for layer, groups in max_avg_stats.items()
                    }
                if perturb_enabled:
                    # The preceding forward is the ordinary Min--Max start.  We
                    # now intervene only on its initial scale; clipping is not
                    # retained, so BRECQ sees the same objective for every rho.
                    minmax_scales = snapshot_activation_scales(qnn)
                    tmma_scales, init_summary = initialize_tmma_interpolation(
                        qnn, group_stats, float(self.args.init_perturb_rho)
                    )
                    stage_probe["initialization"] = init_summary
                    torch.save(
                        {
                            "minmax": minmax_scales,
                            "tmma": tmma_scales,
                            "perturbed": snapshot_activation_scales(qnn),
                            "rho": float(self.args.init_perturb_rho),
                        },
                        os.path.join(self.args.logdir, "init_scales_" + stage_tag + ".pt"),
                    )
                elif group_stats is not None:
                    init_summary = initialize_tmma_scales(qnn, group_stats)
                    stage_probe["initialization"] = init_summary
                    logger.info(
                        "[%s] 激活 BRECQ 前：组 %d TMMA 初始化 %d 个量化器",
                        stage_tag,
                        group_id,
                        init_summary["matched_quantizers"],
                    )

            kwargs_a = dict(
                cali_data=cali_data, iters=self.args.cali_iters_a, act_quant=True,
                opt_mode="mse", lr=self.args.cali_lr, p=self.args.cali_p,
            )
            logger.info("[%s] 开始激活 BRECQ 重建", stage_tag)
            recon_model(qnn, kwargs_a, "act_" + stage_tag)

            qnn.set_quant_state(weight_quant=True, act_quant=True)
            if perturb_enabled:
                stage_probe["post_reconstruction_quantizers"] = len(snapshot_activation_scales(qnn))
                stage_probe["activation_probe"] = activation_quantization_probe(
                    qnn,
                    cali_data,
                    self.device,
                    batch_size=self.args.init_perturb_probe_batch_size,
                    max_batches=self.args.init_perturb_probe_batches,
                )
                save_init_probe_json(
                    os.path.join(self.args.logdir, "init_perturbation_" + stage_tag + ".json"),
                    stage_probe,
                )

        # 固化量化参数后保存
        for m in qnn.model.modules():
            if isinstance(m, AdaRoundQuantizer):
                m.zero_point = nn.Parameter(m.zero_point)
                m.delta = nn.Parameter(m.delta)
            elif isinstance(m, UniformAffineQuantizer) and self.args.quant_act:
                if m.zero_point is not None:
                    if not torch.is_tensor(m.zero_point):
                        m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                    else:
                        m.zero_point = nn.Parameter(m.zero_point)

        ckpt_out = os.path.join(self.args.logdir, f"ckpt_{stage_tag}.pth")
        torch.save(qnn.state_dict(), ckpt_out)
        logger.info("[%s] 保存量化模型: %s", stage_tag, ckpt_out)
        return qnn

    def _save_combined_two_stage_ckpt(self, stage1_path, stage2_path, stage2_state=None):
        """将两阶段权重合并保存为单个 ckpt_two_stage.pth。"""
        stage1_sd = torch.load(stage1_path, map_location="cpu")
        if stage2_state is not None:
            stage2_sd = {k: v.detach().cpu() if torch.is_tensor(v) else v
                         for k, v in stage2_state.items()}
        else:
            stage2_sd = torch.load(stage2_path, map_location="cpu")

        combined = {
            "stage1_group0": stage1_sd,
            "stage2_group1": stage2_sd,
            "meta": {
                "weight_bit": self.args.weight_bit,
                "act_bit": self.args.act_bit,
                "quant_act": bool(self.args.quant_act),
                "a_sym": bool(self.args.a_sym),
                "split": bool(self.args.split),
                "cali_data_path_group0": getattr(self.args, "cali_data_path_group0", None),
                "cali_data_path_group1": getattr(self.args, "cali_data_path_group1", None),
                "stage1_ckpt": os.path.basename(stage1_path),
                "stage2_ckpt": os.path.basename(stage2_path),
            },
        }
        out = os.path.join(self.args.logdir, "ckpt_two_stage.pth")
        torch.save(combined, out)
        logger.info("已合并保存两阶段权重: %s (keys: stage1_group0, stage2_group1, meta)", out)
        return out

    def _build_two_stage_seq(self):
        """构造正确的前后半 DDIM 子序列（升序传入 generalized_steps）。"""
        ddim_steps = 100
        if self.args.skip_type == "uniform":
            full_seq = list(range(0, self.num_timesteps, self.num_timesteps // ddim_steps))
            if len(full_seq) > ddim_steps:
                full_seq = full_seq[:ddim_steps]
        elif self.args.skip_type == "quad":
            full_seq = (
                np.linspace(0, np.sqrt(self.num_timesteps * 0.8), ddim_steps) ** 2
            )
            full_seq = [int(s) for s in list(full_seq)]
        else:
            raise ValueError(f"两阶段模式不支持的 --skip_type: {self.args.skip_type}")

        if len(full_seq) != ddim_steps:
            raise RuntimeError(
                f"构造出的DDIM步数异常: {len(full_seq)} != {ddim_steps}, skip_type={self.args.skip_type}"
            )

        reverse_seq = list(reversed(full_seq))
        split_idx = ddim_steps // 2
        first_half_reverse = reverse_seq[:split_idx]
        second_half_reverse = reverse_seq[split_idx:]

        # generalized_steps 内部会再 reversed(seq)，所以这里传升序
        first_half_seq = list(reversed(first_half_reverse))
        second_half_seq = list(reversed(second_half_reverse))

        logger.info(
            "DDIM 100-step(%s) 切分: 前半=%d步(去噪 %d->%d, 使用 full_seq+limited), "
            "后半=%d步(去噪 %d->%d)",
            self.args.skip_type,
            len(first_half_seq), first_half_reverse[0], first_half_reverse[-1],
            len(second_half_seq), second_half_reverse[0], second_half_reverse[-1],
        )
        return full_seq, first_half_seq, second_half_seq, split_idx

    def _generalized_steps_limited(self, x, seq, model, max_steps, eta):
        """完整升序 seq 上只跑前 max_steps 次迭代，停在中点（不跳到 t=-1）。"""
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            xs = [x]
            for step_idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
                if step_idx >= max_steps:
                    break
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(self.betas, t.long())
                at_next = compute_alpha(self.betas, next_t.long())
                xt = xs[-1].to(x.device)
                et = model(xt, t)
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next.cpu() if xt_next.device.type != "cpu" else xt_next)
            # 保证返回 CPU tensor
            out = xs[-1]
            return out.cpu() if out.device.type != "cpu" else out

    def _run_partial_sampling(self, model, start_x, seq, batch_size, eta, desc,
                              save_png_dir=None, save_start_idx=0, max_steps=None):
        """
        分段 DDIM 采样。
        - max_steps=None: 对 seq 跑完 generalized_steps（后半段）
        - max_steps=K: 对完整升序 seq 只跑前 K 步（前半段，避免落到 t=-1）
        若提供 save_png_dir，则每个 batch 结束后立刻写 PNG。
        """
        model.eval()
        outputs = []
        img_idx = save_start_idx
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, start_x.shape[0], batch_size), desc=desc):
                x = start_x[i:i + batch_size].to(self.device)
                if max_steps is None:
                    xs, _ = generalized_steps(x, seq, model, self.betas, eta=eta)
                    batch_out = xs[-1].cpu()
                else:
                    batch_out = self._generalized_steps_limited(
                        x, seq, model, max_steps, eta
                    )
                outputs.append(batch_out)
                if save_png_dir is not None:
                    os.makedirs(save_png_dir, exist_ok=True)
                    vis = inverse_data_transform(self.config, batch_out)
                    for j in range(vis.shape[0]):
                        tvu.save_image(
                            vis[j], os.path.join(save_png_dir, f"{img_idx}.png")
                        )
                        img_idx += 1
        return torch.cat(outputs, dim=0)

    def two_stage_quantized_sampling(self, fp_model):
        """
        两阶段 w4a8:
          1) stage1: 用前半时间步校准集 BRECQ → 保存 ckpt_stage1_group0.pth → 前半 DDIM
          2) stage2: 用后半时间步校准集 BRECQ → 保存 ckpt_stage2_group1.pth → 后半 DDIM
        采样序列与 split_sampling_from_ckpt / two_stage_newcalibrations 一致（升序子序列）。
        """
        logger.info("开始两阶段量化采样流程（分组校准 + 正确分段 DDIM）")

        if self.args.sample_type != "generalized":
            raise ValueError("两阶段采样仅支持 DDIM generalized 模式")
        if self.args.timesteps != 100:
            logger.warning("两阶段模式固定 DDIM100，当前 --timesteps=%d 将被忽略。", self.args.timesteps)

        g0 = getattr(self.args, "cali_data_path_group0", None)
        g1 = getattr(self.args, "cali_data_path_group1", None)
        if not g0 or not g1:
            raise ValueError(
                "两阶段模式需要 --cali_data_path_group0 与 --cali_data_path_group1\n"
                "例如:\n"
                "  --cali_data_path_group0 get_calibrations/out/dynamic_cali/cifar_g0_slice.pt\n"
                "  --cali_data_path_group1 get_calibrations/out/dynamic_cali/cifar_g1_slice.pt\n"
                "扁平校准集请配合 --cali_st 1"
            )

        use_trunc = (
            bool(getattr(self.args, "max_avg_json", None))
            and not bool(getattr(self.args, "no_act_truncation", False))
        )
        if use_trunc:
            self.max_avg_stats = load_max_avg_statistics(self.args.max_avg_json)
            logger.info("启用 TMMA 激活尺度初始化（max_avg）")
        else:
            self.max_avg_stats = None
            logger.info("不进行 TMMA 激活尺度初始化（与 Q-Diffusion baseline 一致）")

        full_seq, first_half_seq, second_half_seq, split_idx = self._build_two_stage_seq()
        batch_size = getattr(self.config.sampling, "batch_size", 50)

        # 最终 PNG 写到 images/，与中间产物同级，便于查看
        images_dir = os.path.join(self.args.logdir, "images")
        os.makedirs(images_dir, exist_ok=True)
        self.args.image_folder = images_dir

        # ===== Stage 1: group0 =====
        logger.info("=" * 75)
        logger.info("第一阶段：前半时间步校准集 BRECQ (group0)")
        logger.info("=" * 75)
        fp_model_stage1 = Model(self.config)
        fp_model_stage1.load_state_dict(fp_model.state_dict())
        fp_model_stage1.to(self.device)
        fp_model_stage1.eval()
        qnn_stage1 = self._calibrate_qnn_with_dataset(
            fp_model_stage1, g0, "stage1_group0"
        )

        all_noise = torch.randn(
            self.args.max_images,
            self.config.data.channels,
            self.config.data.image_size,
            self.config.data.image_size,
        )
        intermediate_images = self._run_partial_sampling(
            qnn_stage1, all_noise, full_seq, batch_size, self.args.eta, "前半段采样",
            max_steps=split_idx,
        )
        inter_path = os.path.join(self.args.logdir, "intermediate_noise_images.pt")
        torch.save(intermediate_images, inter_path)
        logger.info("中间噪声已保存: %s | finite=%s mean=%.4f",
                    inter_path,
                    bool(torch.isfinite(intermediate_images).all()),
                    float(intermediate_images[torch.isfinite(intermediate_images)].mean())
                    if torch.isfinite(intermediate_images).any() else float("nan"))

        trajectory_probe_enabled = (
            getattr(self.args, "init_perturb_rho", None) is not None
            and self.args.init_perturb_probe_trajectories > 0
        )
        # Keep only a CPU copy until stage 2 has completed.  This avoids a
        # second BRECQ run while keeping GPU memory available for calibration.
        if trajectory_probe_enabled:
            qnn_stage1.to("cpu")
            del fp_model_stage1, all_noise
        else:
            del qnn_stage1, fp_model_stage1, all_noise
        torch.cuda.empty_cache()
        gc.collect()

        # ===== Stage 2: group1 =====
        logger.info("=" * 75)
        logger.info("第二阶段：后半时间步校准集 BRECQ (group1)")
        logger.info("=" * 75)
        fp_model_stage2 = Model(self.config)
        fp_model_stage2.load_state_dict(fp_model.state_dict())
        fp_model_stage2.to(self.device)
        fp_model_stage2.eval()
        qnn_stage2 = self._calibrate_qnn_with_dataset(
            fp_model_stage2, g1, "stage2_group1"
        )

        if trajectory_probe_enabled:
            two_stage_trajectory_probe(
                fp_model=fp_model,
                stage1_qnn=qnn_stage1,
                stage2_qnn=qnn_stage2,
                betas=self.betas,
                seq=full_seq,
                split_idx=split_idx,
                device=self.device,
                n_trajectories=self.args.init_perturb_probe_trajectories,
                channels=self.config.data.channels,
                image_size=self.config.data.image_size,
                seed=self.args.init_perturb_probe_seed,
                output_prefix=os.path.join(self.args.logdir, "init_perturbation_trajectory"),
            )

        stage1_ckpt = os.path.join(self.args.logdir, "ckpt_stage1_group0.pth")
        stage2_ckpt = os.path.join(self.args.logdir, "ckpt_stage2_group1.pth")
        self._save_combined_two_stage_ckpt(
            stage1_ckpt, stage2_ckpt, stage2_state=qnn_stage2.state_dict()
        )

        final_latents = self._run_partial_sampling(
            qnn_stage2, intermediate_images, second_half_seq, batch_size, self.args.eta,
            "后半段采样", save_png_dir=images_dir, save_start_idx=0,
        )
        logger.info("最终 latent | finite=%s", bool(torch.isfinite(final_latents).all()))

        torch.save(final_latents, os.path.join(self.args.logdir, "final_samples.pt"))
        logger.info(
            "两阶段完成：分阶段 ckpt_stage1_group0.pth / ckpt_stage2_group1.pth；"
            "合并 ckpt_two_stage.pth；图像 %s (%d 张，已边采样边写入)",
            images_dir, final_latents.shape[0],
        )


    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        logger.info(f"starting from image {img_id}")
        total_n_samples = self.args.max_images
        n_rounds = math.ceil((total_n_samples - img_id) / config.sampling.batch_size)

        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        with torch.no_grad():
            for i in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                with amp.autocast(enabled=False):
                    x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                if img_id + x.shape[0] > self.args.max_images:
                    assert(i == n_rounds - 1)
                    n = self.args.max_images - img_id
                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from ddim.functions.denoising import generalized_steps

            betas = self.betas
            xs = generalized_steps(
                x, seq, model, betas, eta=self.args.eta, args=self.args)
            x = xs
        elif self.args.sample_type == "dpm_solver":
            logger.info(f"use dpm-solver with {self.args.timesteps} steps")
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
            model_fn = model_wrapper(
                model,
                noise_schedule,
                model_type="noise"
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
            return dpm_solver.sample(
                x,
                steps=self.args.timesteps,
                order=3,
                skip_type="time_uniform",
                method="singlestep",
            )
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")
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
    parser.add_argument(
        "--max_images", type=int, default=10000, help="number of images to sample"
    )

    # qdiff specific configs
    parser.add_argument(
        "--cali_st", type=int, default=1, 
        help="number of timesteps used for calibration"
    )
    parser.add_argument(
        "--cali_batch_size", type=int, default=32, 
        help="batch size for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_n", type=int, default=1024, 
        help="number of samples for each timestep for qdiff reconstruction"
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
        help="calibration dataset name"
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
        help="act quantizers use symmetric quantization"
    )
    parser.add_argument(
        "--running_stat", action="store_true",
        help="use running statistics for act quantizers"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument("--split", action="store_true",
        help="split shortcut connection into two parts"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    parser.add_argument(
        "--delta_ckpt", type=str, default="", help="Path to saved delta weights (ΔW)"
    )
    parser.add_argument(
        "--two_stage", action="store_true",
        help="use two-stage quantization and sampling"
    )
    parser.add_argument(
        "--max_avg_json", type=str, default=None,
        help="Path to JSON with grouped TMMA max_avg statistics"
    )
    parser.add_argument(
        "--no_act_truncation", action="store_true",
        help="disable TMMA activation-scale initialization (legacy option name)"
    )
    perturb = parser.add_argument_group("controlled initialization perturbation")
    perturb.add_argument(
        "--init_perturb_rho", type=float, default=None,
        help="enable the mechanism probe: S0=STMMA+rho(SMinMax-STMMA); requires --a_sym and --max_avg_json",
    )
    perturb.add_argument(
        "--init_perturb_probe_batches", type=int, default=4,
        help="calibration batches used for post-reconstruction activation MSE/clipping diagnostics",
    )
    perturb.add_argument(
        "--init_perturb_probe_batch_size", type=int, default=32,
        help="batch size for activation diagnostics",
    )
    perturb.add_argument(
        "--init_perturb_probe_trajectories", type=int, default=16,
        help="matched FP/quantized DDIM trajectories for direct-noise and latent-error curves; 0 disables",
    )
    perturb.add_argument(
        "--init_perturb_probe_seed", type=int, default=2026,
        help="fixed initial-noise seed for matched trajectory diagnostics",
    )
    perturb.add_argument(
        "--init_perturb_run_name", type=str, default="",
        help="deterministic subdirectory name below --logdir/samples for a rho run",
    )
    recon = parser.add_argument_group("BRECQ early-Adam diagnostics")
    recon.add_argument("--recon_plot_dir", type=str, default="",
                       help="directory for selected-layer BRECQ loss/gradient traces")
    recon.add_argument("--recon_plot_layers", type=str, default="",
                       help="comma-separated QuantModel layer names to trace")
    recon.add_argument("--recon_plot_layers_file", type=str, default="",
                       help="JSON file containing selected QuantModel layer names")
    recon.add_argument("--recon_plot_every", type=int, default=10,
                       help="record every N BRECQ iterations")
    recon.add_argument("--recon_plot_max_iters", type=int, default=500,
                       help="only record the early N BRECQ iterations")
    parser.add_argument(
        "--cali_data_path_group0", type=str, default=None,
        help="前半时间步校准集 (扁平 xs/ts，配合 --cali_st 1)，如 cifar_g0_slice.pt"
    )
    parser.add_argument(
        "--cali_data_path_group1", type=str, default=None,
        help="后半时间步校准集 (扁平 xs/ts，配合 --cali_st 1)，如 cifar_g1_slice.pt"
    )
    return parser


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    parser = get_parser()
    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # fix random seed
    seed_everything(args.seed)

    # setup logger
    if args.resume and args.cali_ckpt:
        logdir = os.path.dirname(os.path.abspath(args.cali_ckpt))
        imglogdir = os.path.join(logdir, "images")
    else:
        run_name = args.init_perturb_run_name or now
        logdir = os.path.join(args.logdir, "samples", run_name)
        imglogdir = os.path.join(logdir, "img")

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(imglogdir, exist_ok=True)
    args.logdir = logdir
    args.image_folder = imglogdir
    log_path = os.path.join(logdir, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(75 * "=")
    # logger.info(f"Host {os.uname()[1]}")
    logger.info("logging to:")
    logger.info(logdir)
    logger.info("image folder: %s", imglogdir)
    logger.info(75 * "=")

    runner = Diffusion(args, config)
    #, device = "cuda:1"
    runner.sample()

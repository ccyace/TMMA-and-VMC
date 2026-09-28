cifar10 on ddim:
 训练 python scripts/two_stage_quantized_sampling.py --config configs/cifar10.yml --use_pretrained --timesteps 100 --eta 0 --skip_type quad --ptq --two_stage --quant_mode qdiff --weight_bit 4 --act_bit 8 --quant_act --a_sym --split --cali_st 1 --cali_batch_size 32 --cali_n 4000 --cali_iters 20000 --cali_iters_a 5000 --cali_data_path_group0  --cali_data_path_group1  --max_avg_json "C:\Users\ASUS\Desktop\q-diffusion-master\activation_max_avg_stats.json" -l 

 采样 python scripts/split_sampling_from_ckpt.py --config configs/cifar10.yml --cali_ckpt  --weight_bit 4 --act_bit 8 --quant_act --a_sym --split --skip_type quad --eta 0 --max_images 50000 --seed 1234



 LDM_church权重通过网盘分享的文件：LDM_church
链接: https://pan.baidu.com/s/1jLthCHD01XNBgsxLACGbjg?pwd=nnm7 提取码: nnm7 
--来自百度网盘超级会员v1的分享


重建python scripts/sample_diffusion_ldm.py `
  -r models/ldm/lsun_churches256/model.ckpt `
  -l  `
  --seed 40 `
  --ptq --two_stage --recon_only `
  --quant_mode qdiff `
  --weight_bit 8 --act_bit 8 --quant_act --a_sym `
  --cali_st 20 --cali_n 256 --cali_batch_size 1 `
  --cali_iters 2000 --cali_iters_a 500 `
  --cali_data_path_group0 get_calibrations/out/dynamic_cali/lsun_church_ldm_ddim400_sample256_g0_slice.pt `
  --cali_data_path_group1 get_calibrations/out/dynamic_cali/lsun_church_ldm_ddim400_sample256_g1_slice.pt `
  --max_avg_json church_ldm_layerwise_activation_statistics.json


采样
python scripts/split_sampling_ldm_from_ckpt.py `
  -r models/ldm/lsun_churches256/model.ckpt `
  --cali_ckpt church_LDM_sample/add_w8a8_9.81/w8a8_LDM_church.pth `
  --outdir  `
  --weight_bit 8 --act_bit 8 --quant_act --a_sym `
  --ddim_steps 400 --skip_type quad --eta 0 `
  --max_images 2000 --batch_size 2 

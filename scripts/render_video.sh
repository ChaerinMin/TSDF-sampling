CUDA_VISIBLE_DEVICES=0 python -m render_video \
  --gin_configs=configs/train_vanilla_nerf.gin \
  --gin_render_configs=configs/render_vanilla_nerf.gin
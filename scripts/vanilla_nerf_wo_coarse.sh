CUDA_VISIBLE_DEVICES=1 python -m train \
  --gin_configs=configs/train_vanilla_nerf_wo_coarse_unnorm.gin
  # --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  # --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'"
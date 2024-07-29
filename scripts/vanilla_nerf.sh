CUDA_VISIBLE_DEVICES=0 python -m train \
  --gin_configs=configs/train_vanilla_nerf.gin
  # --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  # --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'"
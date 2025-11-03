export HF_HOME="/mnt/hf"
export WANDB_MODE="disabled"

DATASET=libero_spatial_no_noops

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path pretrained_models/configs \
  --vlm_path /mnt/luyuxiang/models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --use_minivlm True \
  --data_root_dir /mnt/data/modified_libero_rlds \
  --dataset_name $DATASET \
  --run_root_dir experiments/checkpoints/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150000 \
  --save_freq 100 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --image_aug True \
  --lora_rank 64 \
  --wandb_entity "innovator" \
  --wandb_project "openvla-oft" \
  --run_id_note "mini-$(date +%m%d_%H%M)"

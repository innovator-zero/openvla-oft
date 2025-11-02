NPROC_PER_NODE=${NPROC_PER_NODE:-1}

. /mnt/luyuxiang/miniconda3/etc/profile.d/conda.sh

conda activate openvla-oft

export HF_HOME="/mnt/hf"
export WANDB_MODE="offline"

DATASET=libero_${1}_no_noops

torchrun --standalone --nnodes 1 --nproc-per-node $NPROC_PER_NODE vla-scripts/finetune.py \
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
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 10000 \
  --max_steps 150000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 64 \
  --wandb_entity "innovator" \
  --wandb_project "openvla-oft" \
  --run_id_note "mini-$(date +%m%d_%H%M)" \
  2>&1 | tee -a "experiments/logs/TRAIN_$(date +'%Y%m%d_%H%M').txt"

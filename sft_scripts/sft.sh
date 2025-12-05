#!/bin/bash

# variables
script_name=arm_thinker_sft
full_batch_size=256
per_device_batch_size=2
gradient_accumulation_steps=$((full_batch_size / (per_device_batch_size * PROC_PER_NODE * NNODES)))
echo "gradient_accumulation_steps: ${gradient_accumulation_steps}"
epoch=1.0
lr=2e-5
image_max_pixels=$((4096 * 28 * 28))
cutoff_len=32768
model_name_or_path=models--Qwen--Qwen2.5-VL-7B-Instruct
tokenized_path=/path/to/LLaMA-Factory/tokenized_cache/${script_name}
mkdir -p $tokenized_path
mix_strategy=concat
save_only_model=true
output_dir=/path/to/LLaMA-Factory/saves/qwen2_5_vl_7b/full_sft/${script_name}

# log
log_dir=${output_dir}/logs
mkdir -p $log_dir

torchrun \
    --nnodes $NNODES --nproc_per_node $PROC_PER_NODE  --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    /path/to/LLaMA-Factory/src/train.py \
    --deepspeed /path/to/LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
    --model_name_or_path $model_name_or_path \
    --image_max_pixels ${image_max_pixels} \
    --crop_to_patches true \
    --trust_remote_code true \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --freeze_vision_tower true \
    --freeze_multi_modal_projector true \
    --freeze_language_model false \
    --dataset ARM-Thinker-SFT-Data \
    --template qwen2_vl \
    --cutoff_len ${cutoff_len} \
    --overwrite_cache true \
    --tokenized_path ${tokenized_path} \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --mix_strategy ${mix_strategy} \
    --output_dir $output_dir \
    --logging_steps 5 \
    --plot_loss true \
    --overwrite_output_dir false \
    --save_only_model ${save_only_model} \
    --per_device_train_batch_size $per_device_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate ${lr} \
    --num_train_epochs ${epoch} \
    --lr_scheduler_type cosine \
    --save_steps 100 \
    --warmup_ratio 0.1 \
    --bf16 true \   
    --ddp_timeout 180000000 \
    --flash_attn fa2 \
    --report_to tensorboard \
    2>&1 | tee -a "${log_dir}/training_log_$(date +%Y%m%d_%H%M%S).txt"
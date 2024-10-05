NNODES=1
GPUS_PER_NODE=8
MASTER_ADDR=127.0.0.1
MASTER_PORT=6000
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

LAUNCHER="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file ./configs/8gpusfsdp_llama_7b.yml \
    --num_machines $NNODES \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    --num_processes $WORLD_SIZE \
    --rdzv_conf rdzv_backend=c10d \
    --max_restarts 0 \
    --tee 3 \
    "

export CMD=" \
    -m train \
    --output_dir ./output \
    --model_name_or_path ./VeriSeek \
    --train_data ./data/train_ \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --normalized \
    --temperature 0.02 \
    --train_group_size 8 \
    --negatives_cross_device \
    --query_max_len 4096 \
    --generative_max_len 4096 \
    --passage_max_len 64 \
    --mode unified \
    --logging_steps 1 \
    --bf16 \
    --pooling_method mean \
    --loss_gen_type mixed \
    --attn bbcc \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --save_only_model \
    --num_train_epochs 1 \
    --save_steps 5000 \
    "

clear; 
bash -c "$LAUNCHER $CMD" 2>&1

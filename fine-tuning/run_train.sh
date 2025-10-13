# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=1,2,3,4

DATASET=ml_1m
BASE_MODEL=huggyllama/llama-7b
DATA_PATH=../data
OUTPUT_DIR=./ckpt/ml_1m_hyper/

python -m torch.distributed.run --nproc_per_node=3 --master_port=1445  lora_finetune.py \
    --base_model $BASE_MODEL\
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 4 \
    --tasks seqrec \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --index_file .index_hyper.json\
    --wandb_run_name test\
    --temperature 1.0

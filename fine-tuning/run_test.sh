# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2

DATASET=ml_1m
#METHOD_VARIATION=***
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/
RESULTS_FILE=./results/$DATASET/ddp_results.json
RESULTS_DETAIL_FILE=./results/$DATASET/ddp_details.json
BASE_MODEL=huggyllama/llama-7b


python -m torch.distributed.run --nproc_per_node=3 --master_port=5821 test_ddp_log.py \
	--ckpt_path ./ckpt/ml_1m_hyp/ \
    --base_model $BASE_MODEL\
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --results_detail_file $RESULTS_DETAIL_FILE \
    --test_batch_size 6 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index_hyper.json \
    --inter_file .inter.json


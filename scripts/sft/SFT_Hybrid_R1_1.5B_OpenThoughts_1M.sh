#!/usr/bin/bash

export MASTER_ADDR="127.0.0.1" # select the master address
export MASTER_PORT="45521" # select the port

# Device Configs
NNODES=1 # number of nodes. 
NPROC_PER_NODE=8 # number of gpus (processes) per node
export WORLD_SIZE=$(($NNODES * $NPROC_PER_NODE)) # number of gpus we have in total. Our experiments used 8x8=64 A100
mode=$1 # resume from checkpoint

# Task Configs
TAG="Hybrid-R1-1.5B-OpenThoughts-1M" # this will be the name of output folder
DATA_INDEX_PATH=CACHE # path to the cache folder. Will generate if not exists
PROJECT_PATH=$(pwd)
OUTPUT_PATH="$PROJECT_PATH/outputs"

# Transformer Configs
HIDEN_SIZE=1536 # hidden size
NUM_LAYERS=28 # number of layers
NUM_ATTN_HEADS=12 # number of attention heads
NUM_QUERY_GROUPS=2 # number of query groups
FFN_HIDDEN_SIZE=8960 # feed forward network hidden size
RMS_NORM_EPS=1e-6 # epsilon for rms layer normalization
SEQ_LENGTH=16384 # sequence length
MAX_POSITION_EMBEDDINGS=32768 # maximum position embeddings
ROPE_BASE=10000

# Get Data Blend
DATA_BLEND="assets/data/Vinnnf-Hybrid-OpenThoughts2-1M-1.5B/Tokenized-Vinnnf-Hybrid-OpenThoughts2-1M-1.5B-deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B-${SEQ_LENGTH}_text_document"

# Training Configs
TOKENIZER_MODEL="$PROJECT_PATH/assets/checkpoints/deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B" # path to the tokenizer model

TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
LR=1e-5
MIN_LR=1e-6
TRAIN_ITERS=17000 # number of iterations to train for
WARMUP_ITERS=$(expr $TRAIN_ITERS \* 5 / 100)
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128

# intervals
SAVE_INTERVALS=1000
LOG_INTERVALS=10
EVAL_INTERVALS=100
EVAL_ITERS=10

# Set Training configs
CKPT_SUBDIR="$OUTPUT_PATH/checkpoints/$TAG/train_iters_$TRAIN_ITERS"
if [ "$mode" = "train" ]; then
    LOAD="assets/checkpoints/deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B_Megatron_TP${TENSOR_PARALLEL_SIZE}PP${PIPELINE_PARALLEL_SIZE}"
    EXTRA_CMD="--no-load-optim --no-load-rng --finetune --load $LOAD" 
else
    LOAD="$CKPT_SUBDIR"
    EXTRA_CMD="--load $LOAD"
fi  

cd $PROJECT_PATH; mkdir -p $CKPT_SUBDIR; export WANDB_API_KEY=$WANDB_API_KEY; echo Start Training

# cp current script to the output folder
cp $0 $CKPT_SUBDIR

OPTIONS=" \
--wandb-project Tiny-O1 \
--wandb-exp-name $TAG \
--dataloader-type cyclic \
--add-qkv-bias \
--disable-bias-linear \
--swiglu \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--position-embedding-type rope \
--rotary-base $ROPE_BASE \
--tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
--pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
--context-parallel-size $CONTEXT_PARALLEL_SIZE \
--num-layers $NUM_LAYERS  \
--hidden-size $HIDEN_SIZE \
--num-attention-heads $NUM_ATTN_HEADS \
--seq-length $SEQ_LENGTH \
--max-position-embeddings $MAX_POSITION_EMBEDDINGS \
--ffn-hidden-size $FFN_HIDDEN_SIZE --normalization RMSNorm \
--micro-batch-size $MICRO_BATCH_SIZE \
--global-batch-size $GLOBAL_BATCH_SIZE \
--group-query-attention \
--num-query-groups $NUM_QUERY_GROUPS \
--train-iters $TRAIN_ITERS \
--lr $LR \
--min-lr $MIN_LR \
--lr-decay-style cosine \
--log-interval $LOG_INTERVALS \
--eval-iters $EVAL_ITERS \
--eval-interval $EVAL_INTERVALS \
--data-path "$DATA_BLEND"  \
--data-cache-path $DATA_INDEX_PATH \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model ${TOKENIZER_MODEL} \
--save-interval $SAVE_INTERVALS \
--save $CKPT_SUBDIR \
--split 999,1,0 \
--clip-grad 1.0 \
--weight-decay 0.1 \
--adam-beta1 0.9 \
--adam-beta2 0.95 \
--init-method-std 0.02  \
--log-num-zeros-in-grad \
--lr-warmup-iters $WARMUP_ITERS \
--exit-on-missing-checkpoint \
--use-flash-attn \
--bf16 \
--exit-signal-handler \
--ckpt-format torch \
--rotary-percent 1.0 \
--calculate-per-token-loss \
--rotary-seq-len-interpolation-factor 1 \
--norm-epsilon $RMS_NORM_EPS \
--use-distributed-optimizer \
--overlap-param-gather \
--overlap-grad-reduce \
--sft \
--untie-embeddings-and-output-weights \
${EXTRA_CMD}"

export CUDA_DEVICE_MAX_CONNECTIONS=1;

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT pretrain_gpt.py ${OPTIONS}
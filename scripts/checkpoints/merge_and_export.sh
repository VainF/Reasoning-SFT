CKPT=$1
OUTPUT_CKPT=${CKPT}_tp1pp1
hf_model_name=$2
export_name=$3

python tools/checkpoint/convert.py --loader mcore --load-dir $CKPT --save-dir $OUTPUT_CKPT --model-type GPT --saver mcore --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 --position-embedding-type rope --megatron-path ${PWD}

# load the iter number stored in the latest_checkpointed_iteration.txt
iter=$(cat $OUTPUT_CKPT/latest_checkpointed_iteration.txt)

iter_padded=$(printf "%07d" "$iter")

# load the iter with 7 filled zeros
python scripts/checkpoints/export_to_hf.py --hf_model $hf_model_name --megatron_ckpt $OUTPUT_CKPT/iter_${iter_padded}/mp_rank_00/model_optim_rng.pt --save_ckpt $export_name
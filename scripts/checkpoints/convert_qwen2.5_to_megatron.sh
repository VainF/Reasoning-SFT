SIZE=$1 # model size (e.g. 1.5B)
TP=$2 # tensor parallelism, e.g. 1
PP=$3 # pipeline parallelism, e.g. 1

python tools/checkpoint/convert.py --loader llama_mistral --load-dir assets/checkpoints/Qwen_Qwen2.5_${SIZE} --save-dir assets/checkpoints/Qwen_Qwen2.5_${SIZE}_megatron_TP${TP}PP${PP} --model-type GPT --model-size qwen2.5 --checkpoint-type hf --tokenizer-model assets/checkpoints/Qwen_Qwen2.5_${SIZE} --saver mcore --target-pipeline-parallel-size $PP --target-tensor-parallel-size $TP
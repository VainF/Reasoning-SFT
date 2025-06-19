#!/bin/bash
# Get tokenizer model as the first argument
tokenizer_model="$1"
input_file="$2"
seq_length="$3"

# Extract the basename of the tokenizer model (i.e. remove directory path)
base_tokenizer_model=$(basename "$tokenizer_model")

# Extract the base name of the input file (removing the directory and the .json extension)
base_input=$(basename "$input_file" .json)
base_dir=$(dirname "$input_file")
# Construct the output prefix using the base input name, tokenizer model, and sequence length
output_prefix="${base_dir}/Tokenized-${base_input}-${base_tokenizer_model}-${seq_length}"

python tools/preprocess_sft_data.py \
  --input "$input_file" \
  --output-prefix "$output_prefix" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "$tokenizer_model" \
  --seq-length "$seq_length" \
  --workers 16 \
  --partitions 1
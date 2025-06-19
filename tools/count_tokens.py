import sys
import json, tqdm, os
from transformers import AutoTokenizer
import numpy as np

if len(sys.argv) != 2:
    print("Usage: python script.py <input_jsonl>")
    sys.exit(1)

input_file = sys.argv[1]

extracted_data = []
seen_ids = set()


num_tokens = []
num_completion_tokens = []
num_samples = 0
tokenizer = AutoTokenizer.from_pretrained("assets/checkpoints/Qwen_QwQ_32B", add_bos_token=False)

with open(input_file, 'r') as infile:
    for line in tqdm.tqdm(infile):
        data = json.loads(line)

        if 'resps' in data:
            output = data.get("resps", None)[0][0]
            prompt = data['arguments']['gen_args_0']["arg_0"]
        else:
            output = data.get("output", None)
            prompt = data['instruction']

        conversation = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': output},
        ]
        tokens = tokenizer.apply_chat_template(conversation, return_tensors="pt")
        completion_tokens = tokenizer.encode(output, return_tensors="pt")
        num_completion_tokens.append(completion_tokens.shape[1])
        num_tokens.append(tokens.shape[1])
        num_samples += 1

print(f"Total number of samples: {num_samples}")
print(f">=4096: {len([x for x in num_tokens if x >= 4096])}")
print(f">=16384: {len([x for x in num_tokens if x >= 16384])}")

num_tokens = [x for x in num_tokens if x <= 16384]
# keep two float numbers
print(f"Mean-Std: {np.mean(num_tokens):.1f}+-{np.std(num_tokens):.1f}")
argmax = np.argmax(num_tokens)
argmin = np.argmin(num_tokens)
print(f"Max: {num_tokens[argmax]}, {argmax}")
print(f"Min: {num_tokens[argmin]}, {argmin}")
print(f"Median: {np.median(num_tokens)}")




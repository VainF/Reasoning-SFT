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


losses = []
num_tokens = []

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", add_bos_token=False)

with open(input_file, 'r') as infile:
    for line in tqdm.tqdm(infile):
        data = json.loads(line)
        loss = data.get("loss", None)
        n_tokens = data.get("n_tokens", None)

        losses.append(loss)
        num_tokens.append(n_tokens)

threshold = 0.6
print(f"Total number of samples: {len(losses)}")
print(f"Num tokens Mean-Std: {np.mean(num_tokens):.1f}+-{np.std(num_tokens):.1f}")
print(f"Loss Mean-Std: {np.mean(losses):.1f}+-{np.std(losses):.1f}")
argmax = np.argmax(losses)
argmin = np.argmin(losses)
print(f"Max: {losses[argmax]}, {argmax}")
print(f"Min: {losses[argmin]}, {argmin}")
print(f"Median: {np.median(losses)}")
num_below_threshold = len([x for x in losses if x <= threshold])
print(f"Loss<={threshold}: {num_below_threshold} ({num_below_threshold/len(losses)*100:.1f}%)")

# length of samples with loss>0.8
num_tokens = [x for i, x in enumerate(num_tokens) if losses[i] < threshold]
print(f"avg tokens for loss<{threshold}: {np.mean(num_tokens):.1f}")


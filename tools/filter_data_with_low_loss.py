import sys
import json, tqdm, os
from transformers import AutoTokenizer
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python script.py <input_jsonl> <output_dir>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

cnt=0
threshold = 0.8
with open(output_file, 'w') as outfile:
    with open(input_file, 'r') as infile:
        for line in tqdm.tqdm(infile):
            data = json.loads(line)
            loss = data["loss"]
            if loss<=threshold:
                outfile.write(line)
                cnt+=1
print(f"Total number of samples: {cnt}")


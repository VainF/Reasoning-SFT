import sys
import json, tqdm, os
from math_verify import parse, verify, LatexExtractionConfig

long_cot_data = "assets/data/generated/open_thoughts_r1_7b_greedy.json"

n_samples = 0
output_file = 'assets/data/generated/open_thoughts_long_short.json'
do_shuffle = True
os.makedirs(os.path.dirname(output_file), exist_ok=True)
data_lines = []
n_consistent = 0
errors = 0
with open(output_file, 'w') as outfile:
    with open(long_cot_data, 'r') as long_cot_file:
        for long_line in tqdm.tqdm(long_cot_file):
            long_data = json.loads(long_line)
            short_data = long_data.copy()
            short_answer = "<short>\n"+long_data['output'].split("</think>")[-1].lstrip("\n")
            short_data['output'] = short_answer

            data_lines.append([long_data, short_data])
            n_samples += 2

    if do_shuffle:
        import random
        random.shuffle(data_lines)
        print("Shuffled data")
    data_lines = [item for sublist in data_lines for item in sublist]
    print(data_lines[-1],data_lines[-1].keys())
    for item in data_lines:
        json.dump(item, outfile)
        outfile.write("\n")

print(f"Total samples: {n_samples}")
            
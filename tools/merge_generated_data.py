import sys
import json, tqdm, os

data_list = [
    'gen_results/progressive/fused_1.5B_0.2.json',
    'gen_results/progressive/fused_1.5B_0.4.json',
    'gen_results/progressive/fused_1.5B_0.6.json',
    'gen_results/progressive/fused_1.5B_0.8.json',
    'gen_results/progressive/fused_1.5B_1.0_v2.json',
    'assets/data/open-thoughts-OpenThoughts-114k/open-thoughts-OpenThoughts-114k-shuffled.json'
]
n_samples = 0
output_file = 'assets/data/progressive_open_thoughts_1.5B/progressive_1.5B_with_R1.json'
do_shuffle = True
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as outfile:
    for data_file in data_list:

        data_lines = []
        with open(data_file, 'r') as infile:
            for line in tqdm.tqdm(infile):
                try:
                    data = json.loads(line)
                    data_lines.append(data)
                except json.JSONDecodeError as e:
                    print(f"Skipping a line due to error: {e}")
        print(f"Loaded {len(data_lines)} items from {data_file}")
        n_samples += len(data_lines)
        if do_shuffle:
            import random
            random.shuffle(data_lines)
            print("Shuffled data")
        print(data_lines[0], data_lines[-1])
        for item in data_lines:
            json.dump(item, outfile)
            outfile.write("\n")
print(f"Merged {len(data_list)} files into {output_file}")
print(f"Total samples: {n_samples}")
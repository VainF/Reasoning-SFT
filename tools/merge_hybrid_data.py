import sys
import json, tqdm, os


long_dst='assets/data/generated/open_thoughts_r1_7b_greedy.json'
short_dst='assets/data/generated/open_thoughts_7b_instruct_greedy.json'

n_samples = 0
output_file = 'assets/data/generated/open_thoughts_hybrid.json'
do_shuffle = True
os.makedirs(os.path.dirname(output_file), exist_ok=True)

items = []
with open(long_dst, 'r') as infile1:
    with open(short_dst, 'r') as infile2:
        for line1, line2 in tqdm.tqdm(zip(infile1, infile2)):
            try:
                data1 = json.loads(line1)
                data2 = json.loads(line2)
                items.append([data1, data2])
            except json.JSONDecodeError as e:
                print(f"Skipping a line due to error: {e}")
print(f"Loaded {len(items)} items from {long_dst} and {short_dst}")

# shuffle
if do_shuffle:
    import random
    random.shuffle(items)

# flatten the datasets
items = [item for sublist in items for item in sublist]

with open(output_file, 'w') as outfile:
    for item in items:
        json.dump(item, outfile)
        outfile.write("\n")
print(f"{len(items)} items merged into {output_file}")
import sys
import json, tqdm, os
from math_verify import parse, verify, LatexExtractionConfig

long_cot_data = "assets/data/generated/open_thoughts_r1_7b_greedy.json"
short_cot_data = "assets/data/generated/open_thoughts_7b_instruct_greedy.json"

n_samples = 0
output_file = 'assets/data/generated/open_thoughts_consistency.json'
do_shuffle = True
os.makedirs(os.path.dirname(output_file), exist_ok=True)
data_lines = []
n_consistent = 0
errors = 0
with open(output_file, 'w') as outfile:
    with open(long_cot_data, 'r') as long_cot_file:
        with open(short_cot_data, 'r') as short_cot_file:
            for long_line, short_line in tqdm.tqdm(zip(long_cot_file, short_cot_file)):
                long_data = json.loads(long_line)
                short_data = json.loads(short_line)
                assert long_data['instruction'] == short_data['instruction']
                long_data['is_long'] = True
                short_data['is_long'] = False
                
                # Do the math verification
                try:
                    # we need to remove prove questions or  answers that are not in the correct format
                    if ("\\boxed" in long_data['output'] and "\\boxed" in short_data['output']) and "prove" not in long_data['instruction']:
                        long_answer = parse(long_data['output'][-4000:])
                        short_answer = parse(short_data['output'][-4000:])
                        consistency = verify(long_answer, short_answer)
                    else:
                        consistency = False
                except:
                    consistency = False

                long_data['consistency'] = consistency
                short_data['consistency'] = consistency

                if consistency:
                    n_consistent += 2

                data_lines.append([long_data, short_data])
                n_samples += 2

                if n_samples%1000 ==0:
                    print(f"Processed {n_samples} samples, {n_consistent} consistent samples")


    if do_shuffle:
        import random
        random.shuffle(data_lines)
        print("Shuffled data")
    data_lines = [item for sublist in data_lines for item in sublist]
    print(data_lines[-1],data_lines[-1].keys())
    for item in data_lines:
        json.dump(item, outfile)
        outfile.write("\n")
print(f"Merged {long_cot_data} and {short_cot_data} into {output_file}")
print(f"Total samples: {n_samples}")
print(f"Consistent samples: {n_consistent}")
            
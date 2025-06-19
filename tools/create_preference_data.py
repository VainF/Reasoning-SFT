import sys
import json, tqdm, os
from math_verify import parse, verify, LatexExtractionConfig

long_cot_data = "assets/data/generated/openr1_math_r1_7b_greedy.json"
short_cot_data = "assets/data/generated/openr1_math_7b_instruct_greedy.json"

n_samples = 0
output_file = 'assets/data/generated/openr1_math_7b_preference.json'
do_shuffle = True
os.makedirs(os.path.dirname(output_file), exist_ok=True)
data_lines = []
n_consistent = 0
errors = 0
confusion_matrix = [[0, 0], [0, 0]]
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
                
                gold = parse(long_data['answer'])
                long_answer = parse(long_data['output'][-4000:])
                short_answer = parse(short_data['output'][-4000:])

                correctness = [verify(gold, long_answer), verify(gold, short_answer)]

                consistency = verify(long_answer, short_answer)
                long_data['consistency'] = consistency
                short_data['consistency'] = consistency

                if consistency:
                    n_consistent += 1
                if not correctness[0] and not correctness[1]:
                    confusion_matrix[0][0] += 1
                elif correctness[0] and not correctness[1]:
                    confusion_matrix[1][0] += 1
                elif not correctness[0] and correctness[1]:
                    confusion_matrix[0][1] += 1
                else:
                    confusion_matrix[1][1] += 1

                long_data['correctness'] = correctness
                short_data['correctness'] = correctness

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
print(f"Merged {long_cot_data} and {short_cot_data} into {output_file}")
print(f"Total samples: {n_samples}")
print(f"Confusion matrix: {confusion_matrix}")
print(f"Consistent samples: {n_consistent}")
            
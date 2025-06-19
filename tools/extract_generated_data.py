import sys
import json, tqdm, os
from transformers import AutoTokenizer
if len(sys.argv) != 4:
    print("Usage: python script.py tokenizer_path input_file output_file")
    sys.exit(1)

tokenizer_path = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]

extracted_data = []
seen_ids = set()

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_bos_token=False)

try:
    with open(input_file, 'r') as infile:
        for line in tqdm.tqdm(infile):
            try:
                data = json.loads(line)
                instruction = data['doc']["problem"]
                doc_id = data['doc_id']
                output = data.get("resps", None)[0][0]
                
                conversation = [
                    {'role': 'user', 'content': instruction},
                    {'role': 'assistant', 'content': output},
                ]
                tokens = tokenizer.apply_chat_template(conversation, return_tensors="pt")
                if tokens.shape[1]>=16384:
                    #print(f"Skipping a line due to token length: {tokens.shape[1]}")
                    continue

                if "</think>" in output:
                    output = "<think>\n" + output
    
                if doc_id in seen_ids:
                    print(f"Skipping duplicate doc_id: {doc_id}")
                    continue
                seen_ids.add(doc_id)
                item = {"instruction": instruction, "output": output}
                target = data.get("target", None)
                if target is not None:
                    item["target"] = target
                extracted_data.append(item)
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Skipping a line due to error: {e}")
except FileNotFoundError:
    print(f"Error: File '{input_file}' not found.")
    sys.exit(1)

print(extracted_data[0], extracted_data[-1], len(extracted_data))
os.makedirs(os.path.dirname(output_file), exist_ok=True)
try:
    with open(output_file, 'w') as outfile:
        for item in extracted_data:
            json.dump(item, outfile)
            outfile.write("\n")
    print(f"Extraction of {len(extracted_data)} items completed successfully.")
except IOError as e:
    print(f"Error writing to file '{output_file}': {e}")
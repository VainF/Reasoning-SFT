from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
class EOSStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # Stop if all last tokens are EOS
        return all(sequence[-1] == self.eos_token_id for sequence in input_ids)


checkpoint = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#assistant_checkpoint = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
assistant_checkpoint = "Qwen/Qwen2.5-Math-1.5B"

import json, tqdm, sys, os, random
device = f"cuda:{sys.argv[3]}"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Get the EOS token ID
eos_token_id = tokenizer.eos_token_id
stopping_criteria = StoppingCriteriaList([EOSStoppingCriteria(eos_token_id)])

model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map=device, torch_dtype=torch.float16)
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint, device_map=device, torch_dtype=torch.float16)
assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_checkpoint)
# Check if vocab of assistant model matches the vocab of the model, one by one
print(model, assistant_model)

#import tqdm
#print("Checking vocab match")
#for key, assistant_model_token in tqdm.tqdm(assistant_tokenizer.vocab.items()):
#    if assistant_model_token != tokenizer.vocab[key]:
#        print(f"Token mismatch at token {key}: {assistant_model_token} != {tokenizer.vocab[key]}")
#        raise ValueError("Vocab mismatch")

if True:
    # load prompt from open-thoughts/OpenThoughts-114k
    prompt_text = []
    json_file = "assets/data/open-thoughts-OpenThoughts-114k/open-thoughts-OpenThoughts-114k-shuffled.json"
    with open(json_file, "r") as f:
        for line in tqdm.tqdm(f):
            data = json.loads(line)
            prompt_text.append("<｜begin▁of▁sentence｜><｜User｜>"+data['instruction']+"<｜Assistant｜>")
    device_partition = float(sys.argv[3]) / 8, (float(sys.argv[3]) + 1) / 8
    prompt_text = prompt_text[int(device_partition[0]*len(prompt_text)):int((device_partition[1])*len(prompt_text))]
    print(f"Loaded {len(prompt_text)} prompts from {json_file}")
else:
    prompt_text = [
        "<｜begin▁of▁sentence｜><｜User｜>",
    ]
#print("Pre-tokenizing prompts...")
#prompt = [ tokenizer(p, return_tensors="pt").to(device) for p in tqdm.tqdm(prompt_text) ]
n_prompts = len(prompt_text) if len(prompt_text)>100 else 50000

save_path = sys.argv[1]
seed = int(sys.argv[2])
torch.manual_seed(seed)
with torch.no_grad():
    with open(save_path, "w") as f:
        f.write("[\n")
        for k in tqdm.trange(n_prompts):
            promopt_id = k
            adv_configs = {
                'alpha': 0.2 * random.random(),
            }
            print(adv_configs)
            #torch.manual_seed(random.randint(0, 1000000))
            prompt_raw = prompt_text[promopt_id]
            prompt = tokenizer(prompt_raw, return_tensors="pt").to(device)

            outputs = model.generate(**prompt, assistant_model=assistant_model, adv_configs=adv_configs, do_sample=True, max_length=16384, temperature=0.6, repetition_penalty=1.0, eos_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            # split the outputs into prompt and response
            
            response = outputs[0].split(prompt_raw)[-1].strip()

            if k!=0:
                f.write(",\n")
            f.write(json.dumps({"response": response, "prompt": prompt_raw, "adv_configs": adv_configs}, ensure_ascii=False))
            # flush the buffer
            f.flush()
        f.write("]")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

def flatten_state_dict(state_dict):
    # Flatten all tensors in a state dict into one long vector.
    return torch.cat([v.view(-1) for v in state_dict.values() if v is not None])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the models.
    print("Loading models...")
    model1 = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").to(device)
    model2 = AutoModelForCausalLM.from_pretrained("assets/checkpoints/Qwen_Qwen2.5_Math_1.5B_DeepSeek").to(device)
    
    # Set model2 to train mode so that gradients are computed.
    model2.train()  

    # 2. Compute deltaW = W1 - W2 for each matching parameter.
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    deltaW = {}
    for key in state_dict1:
        if key in state_dict2:
            # Ensure both tensors are on the same device.
            deltaW[key] = state_dict1[key].to(device) - state_dict2[key].to(device)
    # Flatten deltaW into a single vector.
    deltaW_flat = flatten_state_dict(deltaW)
    print("Computed deltaW.")
    
    # Use the tokenizer associated with model2.
    tokenizer = AutoTokenizer.from_pretrained("assets/checkpoints/Qwen_Qwen2.5_Math_1.5B_DeepSeek")
    
    # 3. Process the JSONL file.
    input_file_path = "assets/data/open-thoughts-OpenThoughts-114k/open-thoughts-OpenThoughts-114k-shuffled.json"
    output_file_path = "assets/data/open-thoughts-OpenThoughts-114k/open-thoughts-OpenThoughts-114k-cossim.json"
    
    with open(input_file_path, "r", encoding="utf-8") as fin, \
         open(output_file_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            try:
                sample = json.loads(line)
                instruction = sample.get("instruction", "")
                output_text = sample.get("output", "")
                # Combine instruction and output.
                prompt = instruction + "\n" + output_text
                
                # Tokenize and move inputs to device.
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Clear any previous gradients.
                model2.zero_grad()
                
                # 3.1 Compute next-token prediction loss.
                # The labels are the same as input_ids (i.e. language modeling loss).
                outputs = model2(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                # Compute gradients with respect to model2's parameters.
                loss.backward()
                
                # 4. Extract and flatten gradients.
                grads = []
                for param in model2.parameters():
                    grads.append(param.grad.view(-1))
                grad_flat = torch.cat(grads)
                
                # Compute the cosine similarity between the gradient and deltaW.
                cos_sim = torch.nn.functional.cosine_similarity(grad_flat, deltaW_flat, dim=0).item()
                
                # Save the cosine similarity in the sample.
                sample["cos_sim"] = cos_sim
                
                # Write the updated sample to the output file.
                fout.write(json.dumps(sample) + "\n")
            
            except Exception as e:
                print("Error processing a line:", e)
    
    print("Processing complete. Results saved to", output_file_path)

if __name__ == "__main__":
    main()

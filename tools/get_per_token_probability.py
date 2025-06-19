#!/usr/bin/env python

import argparse
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm 

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM loss on a dataset.")
    parser.add_argument("model_card", type=str, help="HuggingFace model card name")
    parser.add_argument("dataset_path", type=str, help="Path to the JSONL dataset")
    args = parser.parse_args()

    model_name = args.model_card
    dataset_path = args.dataset_path
    
    # Prepare output filename
    base, ext = os.path.splitext(dataset_path)
    # Replace slashes and other special characters in model name with underscores
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    output_filename = f"{base}_with_{safe_model_name}_loss{ext}"
    
    # Load model and tokenizer
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    # If CUDA is available, use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Evaluating loss on dataset: {dataset_path}")
    # Read input JSONL and write output JSONL
    with open(dataset_path, "r", encoding="utf-8") as fin, \
         open(output_filename, "w", encoding="utf-8") as fout:
        
        for line_idx, line in tqdm.tqdm(enumerate(fin)):
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            
            instruction = data['instruction']
            output_text = data['output']

            text = [
                {'role': 'user', 'content': instruction},
                {'role': 'assistant', 'content': output_text}
            ]
            
            # Create conversation
            inputs = tokenizer.apply_chat_template(text, return_tensors="pt")
            #print(tokenizer.apply_chat_template(text, tokenize=False))
            input_ids = inputs.to(device)

            # Compute the loss
            with torch.no_grad():
                # The labels are the same as the input IDs (language modeling objective)
                outputs = model(
                    input_ids=input_ids,
                    labels=input_ids
                )
                # get per-token probability

                shifted_logits = logits[:, :-1, :]
                shifted_labels = input_ids[:, 1:]

                # Convert shifted logits to probabilities
                probabilities = torch.softmax(shifted_logits, dim=-1)

                # Gather the probability of the true token at each position
                token_probs = probabilities.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)

            # You can store the loss into the data dictionary
            data[f"loss"] = loss
            data["token_probs"] = token_probs.cpu().numpy().tolist()
            
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

            fout.flush()

    print(f"Done. Output saved to: {output_filename}")

if __name__ == "__main__":
    main()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from torch.nn import CrossEntropyLoss

def flatten_state_dict(state_dict):
    # Flatten all tensors in a state dict into one long vector.
    return torch.cat([v.view(-1) for v in state_dict.values() if v is not None])

def main():
    # 1. Load the models in bf16
    print("Loading models...")
    model_name = 'outputs/export/7B_Instruct_OpenThoughts'
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Set up the data paths
    input_file_path = "assets/data/generated/open_thoughts_r1_7b_greedy.json"
    output_file_path = "assets/data/generated/open_thoughts_r1_7b_greedy-loss.json"
    
    n_lines = 0
    from tqdm import tqdm

    # We will use a no_grad context to avoid storing gradients.
    with torch.no_grad():
        with open(input_file_path, "r", encoding="utf-8") as fin, \
             open(output_file_path, "w", encoding="utf-8") as fout:

            for line in tqdm(fin):
                sample = json.loads(line)
                instruction = sample["instruction"]
                output_text = sample["output"]

                prompt = [
                    {'role': 'user', 'content': instruction},
                    {'role': 'assistant', 'content': output_text}
                ]

                # 2.1 Tokenize and move inputs to device.
                inputs = tokenizer.apply_chat_template(
                    prompt, 
                    return_tensors="pt"
                )

                # Optionally, print the first prompt in plain text for debugging:
                if n_lines == 0:
                    print("First tokenized (text):")
                    print(tokenizer.apply_chat_template(
                        prompt, return_tensors="pt", tokenize=False
                    ))

                # 3. Forward pass with labels for language modeling
                #    (causal LM objective)
                # You should explicitly pass input_ids to `labels`, like so:
                outputs = model(
                    inputs,
                    labels=inputs,
                    return_dict=True
                )

                # 3.1 Get per-token cross-entropy loss
                logits = outputs.logits  # shape: [batch, seq_len, vocab_size]

                # We usually shift the logits and labels to align predictions with targets
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs[..., 1:].contiguous()

                # Compute loss per token
                loss_fct = CrossEntropyLoss(reduction="none")
                token_level_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                # Reshape to [batch_size, seq_len - 1]
                token_level_loss = token_level_loss.view(shift_labels.size())

                # 3.2 Average per-token loss for the batch (we have batch_size=1, but in general):
                avg_loss_per_sample = token_level_loss.mean(dim=1)  # shape: [batch_size]

                # Store per-token loss (as a list), plus the mean
                # For a single sample, batch_size is 1, so index with [0]
                sample["per_token_loss"] = token_level_loss[0].tolist()
                sample["loss"] = float(avg_loss_per_sample[0].item())
                sample["n_tokens"] = int(shift_labels.size(1))

                # Write the updated sample to the output file
                fout.write(json.dumps(sample) + "\n")
                n_lines += 1


    
    print("Processing complete. Results saved to", output_file_path)

if __name__ == "__main__":
    main()

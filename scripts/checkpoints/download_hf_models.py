import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-card", type=str, default="meta-llama/Meta-Llama-3-8B")
args = parser.parse_args()

HF_TOKEN = os.environ.get("HF_TOKEN")
ASSETS_DIR = "assets"
CKPT_DIR = f"{ASSETS_DIR}/checkpoints"
CACHE_DIR = f"{ASSETS_DIR}/cache"

os.makedirs(CKPT_DIR, exist_ok=True)

def save_hf_model(model_id, save_directory):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        cache_dir=f"./assets/cache", 
        use_auth_token=HF_TOKEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    # Save model and tokenizer to the specified directory
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")

save_directory = f"{CKPT_DIR}/{args.model_card.replace('/', '-').replace('-', '_')}"
save_hf_model(args.model_card, save_directory)
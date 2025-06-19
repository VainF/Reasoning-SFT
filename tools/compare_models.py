import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, sys
def compare(model_name_1, model_name_2):
    """
    Fuse two Hugging Face LLMs using the weighted sum: w = w1 * alpha + w2 * (1 - alpha)
    
    Args:
        model_name_1 (str): Name or path of the first model.
        model_name_2 (str): Name or path of the second model.
        alpha (float): Weighting factor for model 1. (0 <= alpha <= 1)
        save_path (str): Path to save the fused model.
    """
    
    # Load both models
    model_1 = AutoModelForCausalLM.from_pretrained(model_name_1, torch_dtype=torch.float32)
    model_2 = AutoModelForCausalLM.from_pretrained(model_name_2, torch_dtype=torch.float32)
    
    # Ensure the models have the same architecture
    assert model_1.state_dict().keys() == model_2.state_dict().keys(), "Model architectures do not match!"
    
    # Fuse model weights
    for (pname_1, p1), (pname_2, p2) in zip(model_1.named_parameters(), model_2.named_parameters()):
        if p1.size() != p2.size():
            print(f"Skipping {pname_1} and {pname_2} due to size mismatch: {p1.size()} != {p2.size()}")
            continue
        param_sim = torch.cosine_similarity(p1.flatten(), p2.flatten(), dim=0)
        param_mse = torch.nn.functional.mse_loss(p1.flatten(), p2.flatten())
        print(f"{pname_1} Sim={param_sim} MSE={param_mse}")


# Example usage
if __name__ == "__main__":
    model_1 = sys.argv[1]
    model_2 = sys.argv[2]
    compare(model_1, model_2)
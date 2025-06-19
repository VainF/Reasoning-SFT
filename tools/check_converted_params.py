# load two huggingface models and compare the parameters
# if the parameters are the same, then the conversion is successful

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys, os
def main():
    model_name1 = sys.argv[1]
    model_name2 = sys.argv[2]

    model_1 = AutoModelForCausalLM.from_pretrained(model_name1)
    model_2 = AutoModelForCausalLM.from_pretrained(model_name2)

    model_1_params = model_1.state_dict()
    model_2_params = model_2.state_dict()

    for key in model_1_params.keys():
        if key in model_2_params.keys():
            if torch.equal(model_1_params[key], model_2_params[key]):
                #print(f"Parameter {key} is the same, {model_1_params[key].shape}, {model_2_params[key].shape}")
                pass
            elif 'lm_head' or 'embed_tokens' in key:
                shape_1 = model_1_params[key].shape
                shape_2 = model_2_params[key].shape
                samll_size = min(shape_1[0], shape_2[0])
                if torch.equal(model_1_params[key][:samll_size], model_2_params[key][:samll_size]):
                    print(f"Parameter {key} is the same, {model_1_params[key].shape}, {model_2_params[key].shape}")
                else:
                    print(f"Parameter {key} is different, {model_1_params[key].shape}, {model_2_params[key].shape}")
            else:
                print(f"Parameter {key} is different, {model_1_params[key].shape}, {model_2_params[key].shape}")
        else:
            print(f"Parameter {key} not found in model 2")

if __name__ == "__main__":
    main()
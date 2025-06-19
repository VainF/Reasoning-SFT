# Megatron-SFT

This repository is a customized version of [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM), extended to support Supervised Fine-Tuning (SFT). **Megatron-SFT** applies prompt masking to train exclusively on the response. It was used to train the hybrid reasoning model [Thinkless-1.5B-Warmup](https://huggingface.co/Vinnnf/Thinkless-1.5B-Warmup) using SFT. You can also use the code for standard SFT. 

## Setup

We recommend using a Docker container to run this code, as installing Transformer Engine and Megatron-LM might be a bit complex.

```bash
# In your user account
cd Megatron-SFT
pip install -r requirements.txt # install the transformers in your user account
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v "$PWD":"$PWD" -v $HOME:$HOME -w "$PWD" -it --rm nvcr.io/nvidia/pytorch:24.12-py3 
```

Running the above command will mount both the current directory and your home directory into the Docker container. To mount additional directories, simply add `-v /path/to/dir:/path/to/dir` to the command.

Once inside the Docker container, install all necessary packages:
```bash
# In the docker
pip install -r requirements.txt
```

## Example: Hybrid Reasoning via SFT (DeepSeek-R1-Distill-Qwen-1.5B)

In this example, we show how to finetune the ``deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`` to enable hybrid reasoning. 

> [!IMPORTANT]
> Since docker create files in root mode, we first download and preprocess models and data in your user account, so that you can easily modify the files using your editor such VSCode.

### 1. Prepare the LLM

```bash
# In your user account 
python scripts/checkpoints/download_hf_models.py --model-card deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```
The huggingface model will be saved in `assets/checkpoints`.
```bash
assets
├── cache
└── checkpoints
    └── deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B
        ├── config.json
        ├── generation_config.json
        ├── model.safetensors
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── tokenizer.json
```

Then, we modify the tokeinizer files to replace the `<|quad_start|>` token with a control token `<short>`. 
```
#assets/checkpoints/deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B/tokenizer_config.json
"151650": {
      "content": "<|short|>", # originally "<|quad_start|>"
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false # originally true
    },
```

```
#assets/checkpoints/deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B/tokenizer.json
    {
      "id": 151650,
      "content": "<short>", # originally "<|quad_start|>"
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false # originally true
    },
```

Remove the final `<think>` in the chat template, and remove the split (`content = content.split('</think>')[-1]`).
```
#assets/checkpoints/deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B/tokenizer_config.json
  "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>\\n'}}{% endif %}",
```


### 2. Convert to Megatron Format

Convert the HF model to Megatron format:
```bash
# In the docker
bash scripts/checkpoints/convert_deepseek_r1_to_megatron.sh 1.5B 1 1 
```
We have three parameters here: 
* The model size: `1.5B`, `32B` etc.
* Tensor Parallel and Pipeline Parallel: `1 1` means no tensor parallel and no pipeline parallel. You can try `2 1` or `1 2` to use tensor parallel or pipeline parallel, respectively.

The above command will create a megatron checkpoint like this:
```bash
assets
├── cache
└── checkpoints
    ├── deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B
    └── deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B_Megatron_TP1PP1
        ├── iter_0000001
        │   └── mp_rank_00
        │       └── model_optim_rng.pt
        └── latest_checkpointed_iteration.txt
```

### 3. Prepare the Hybrid Reasoning Dataset

Download the hybrid reasoning dataset from Huggingface and save it as a json file. 
```bash
# In your user account
bash scripts/data/download_hf_dataset.py --dataset-card Vinnnf/Hybrid-OpenThoughts2-1M-1.5B
```
```bash
assets
├── cache
├── checkpoints
└── data
    └── Vinnnf-Hybrid-OpenThoughts2-1M-1.5B
        └── Vinnnf-Hybrid-OpenThoughts2-1M-1.5B.json
```

Pre-tokenize the dataset:
```bash
# In the docker
bash scripts/data/tokenize_dataset.sh assets/checkpoints/deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B assets/data/Vinnnf-Hybrid-OpenThoughts2-1M-1.5B/Vinnnf-Hybrid-OpenThoughts2-1M-1.5B.json 16384
```
```
assets/
├── cache
├── checkpoints
└── data
    └── Vinnnf-Hybrid-OpenThoughts2-1M-1.5B
        ├── Tokenized-Vinnnf-Hybrid-OpenThoughts2-1M-1.5B-deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B-16384_text_document.bin
        ├── Tokenized-Vinnnf-Hybrid-OpenThoughts2-1M-1.5B-deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B-16384_text_document.idx
        └── Vinnnf-Hybrid-OpenThoughts2-1M-1.5B.json
```
The parameters are:
* The path to the tokenizer model (usually the HF model path)
* The path to the dataset JSON file
* The maximum sequence length for training, a value larger than 16384 is recommended.


### 4. Training

Run the fine-tuning script:
```bash 
bash scripts/sft/SFT_Hybrid_R1_1.5B_OpenThoughts_1M.sh train
```

Auto Resume:
```bash 
bash scripts/sft/SFT_Hybrid_R1_1.5B_OpenThoughts_1M.sh resume
```

### 5. Export to Huggingface Format
```bash
bash scripts/checkpoints/merge_and_export.sh PATH_TO_YOUR_CKPT assets/checkpoints/deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B assets/checkpoints/export/Hybrid_R1_1.5B
```

## Acknowledgement

This implementation is also heavily based on [alibaba/Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/toolkits/sft_data_preprocessing).

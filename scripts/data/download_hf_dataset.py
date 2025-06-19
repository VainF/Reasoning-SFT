from datasets import load_dataset
import os

import argparse

parser = argparse.ArgumentParser(description="Download and prepare the OpenThoughts dataset.")
parser.add_argument('--dataset-card', type=str, required=True)
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()


DATASET_CARD=args.dataset_card

CACHE_DIR = "assets/cache"
DATA_DIR = f"assets/data/{DATASET_CARD.replace('/', '-')}"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Preprocess instruction and output, we assume that the huggingface dataset already has the `instruction` and `output` fields.
dst = load_dataset(DATASET_CARD, cache_dir=CACHE_DIR)
#dst['train'] = dst['train'].map(
#    lambda x: {
#        'instruction': x["problem"],
#        'output': x["solution"]
#    }
#)
#dst['train'] = dst['train'].remove_columns(
#    [col for col in dst['train'].column_names if col not in ['instruction', 'output']]
#)

# save as json files
dst[args.split].to_json(f'{DATA_DIR}/{DATASET_CARD.replace("/", "-")}.json', orient='records', lines=True)
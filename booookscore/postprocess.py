import os
import pickle
import json
import argparse
from tqdm import tqdm
from booookscore.utils import APIClient, count_tokens

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str)
parser.add_argument("--model", type=str, help="model name")
parser.add_argument("--api", type=str, help="api to use", choices=["openai", "anthropic", "together"])
parser.add_argument("--api_key", type=str, help="path to the api key")
parser.add_argument("--remove_artifacts", action="store_true", help="whether to explicitly as a model to remove artifacts from summaries")
args = parser.parse_args()

if args.remove_artifacts:
    client = APIClient(args.api, args.api_key, args.model)
    with open(f"prompts/remove_artifacts.txt", "r") as f:
        template = f.read()

data = json.load(open(args.input_path, 'r'))
save_path = args.input_path.replace('.json', '_cleaned.json')
cleaned_summaries = {}
if os.path.exists(save_path):
    cleaned_summaries = json.load(open(save_path, 'r'))
for book in tqdm(data, total=len(data), desc="Iterating over books"):
    if book in cleaned_summaries:
        print(f"Skipping {book}")
        continue
    summary = data[book]
    if isinstance(summary, dict):
        summary = summary['final_summary']
    elif isinstance(summary, list):
        summary = summary[-1]
    if args.remove_artifacts:
        num_tokens = count_tokens(summary)
        prompt = template.format(summary)
        cleaned_summary = client.obtain_response(prompt, max_tokens=num_tokens, temperature=0)
        cleaned_summaries[book] = cleaned_summary
    else:
        cleaned_summaries[book] = summary
    json.dump(cleaned_summaries, open(save_path, 'w'))

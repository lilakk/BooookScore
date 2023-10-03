import os
import pickle
import tqdm
import json
import argparse
from scripts.utils import obtain_response, count_tokens

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--chunk_size", type=int)
parser.add_argument("--summary_strategy", type=str, choices=['inc', 'hier'])
args = parser.parse_args()


def remove_artifacts(path):
    if path.endswith('.pkl'):
        data = pickle.load(open(path, 'rb'))
        save_path = path.replace('.pkl', '_cleaned.json')
    elif path.endswith('.json'):
        data = json.load(open(path, 'r'))
        save_path = path.replace('.json', '_cleaned.json')
    cleaned_summaries = {}
    if os.path.exists(save_path):
        cleaned_summaries = json.load(open(save_path, 'r'))
    with open(f"prompts/remove_artifacts.txt", "r") as f:
        template = f.read()
    for book in tqdm.tqdm(data, total=len(data), desc="Iterating over books"):
        summary = data[book]
        if book in cleaned_summaries:
            print(f"Skipping {book} because it already exists in {save_path}")
            continue
        print(f"ORIGINAL SUMMARY: {summary}\n\n")
        num_tokens = count_tokens(summary)
        prompt = template.format(summary)
        response = obtain_response(prompt, max_tokens=num_tokens, temperature=0)
        cleaned_summary = response['choices'][0]['message']['content'].strip()
        print(f"CLEANED SUMMARY: {cleaned_summary}\n\n")
        cleaned_summaries[book] = cleaned_summary
        json.dump(cleaned_summaries, open(save_path, 'w'))


remove_artifacts('novel_summaries_hier_claude/2048_final.json')

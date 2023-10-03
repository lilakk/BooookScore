import os
import pickle
import tqdm
import json
import argparse
from scripts.utils import obtain_response, count_tokens

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str)
args = parser.parse_args()

INPUT_PATH = args.input_path


def remove_artifacts():
    if INPUT_PATH.endswith('.pkl'):
        data = pickle.load(open(INPUT_PATH, 'rb'))
        save_path = INPUT_PATH.replace('.pkl', '_cleaned.json')
    elif INPUT_PATH.endswith('.json'):
        data = json.load(open(INPUT_PATH, 'r'))
        save_path = INPUT_PATH.replace('.json', '_cleaned.json')
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
        cleaned_summary = response
        print(f"CLEANED SUMMARY: {cleaned_summary}\n\n")
        cleaned_summaries[book] = cleaned_summary
        json.dump(cleaned_summaries, open(save_path, 'w'))


remove_artifacts()

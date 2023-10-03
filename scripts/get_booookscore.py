import os
import numpy as np
import tqdm
import argparse
import json
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from scripts.utils import obtain_response

model = SentenceTransformer('all-MiniLM-L6-v2')

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--chunk_size", type=int)
parser.add_argument("--summary_strategy", type=str, choices=['inc', 'hier'])
args = parser.parse_args()

MODEL = args.model
SUMMARY_STRATEGY = args.summary_strategy
CHUNK_SIZE = args.chunk_size
WORD_RATIO = 0.65

all_labels = ['entity omission', 'event omission', 'causal omission', 'salience', 'discontinuity', 'duplication', 'inconsistency', 'language']


def validate_response(response):
    lines = response.split('\n')
    if len(lines) < 2:
        print("Number of lines is less than 2")
        return False, [], []
    
    # only keep the lines with questions and types
    lines = [line for line in lines if "Questions: " in line or "Types: " in line]
    
    questions_pos = lines[0].find("Questions: ")
    types_pos = -1
    types_pos = lines[1].find("Types: ")

    if questions_pos == -1 or types_pos == -1:
        print("Questions or types not found")
        return False, [], []
    
    questions = lines[0][questions_pos + len("Questions: "):].strip()
    types = lines[1][types_pos + len("Types: "):].strip()

    if "no confusion" in questions:
        if "no confusion" not in types:
            print("No confusion in questions but not in types")
            return False, [], []
        else:
            return True, None, None

    if types is not None:
        types = types.split(', ')
        for t in types:
            if t not in all_labels:
                return False, [], []
    
    if questions is not None and types is None:
        raise ValueError("Questions is not None but types is None")

    return True, questions, types


def get_annotations():
    data = json.load(open(f"summaries/{MODEL}-{CHUNK_SIZE}-{SUMMARY_STRATEGY}-cleaned.json", 'r'))
    save_path = f"gpt4-annotations/{MODEL}-{CHUNK_SIZE}-{SUMMARY_STRATEGY}.json"
    spans_questions = defaultdict(dict)
    if os.path.exists(save_path):
        spans_questions = json.load(open(save_path, 'r'))
        spans_questions = defaultdict(dict, spans_questions)
        print(f"LOADED {len(spans_questions)} spans_questions FROM {save_path}")

    with open("prompts/get_gpt4_annotations.txt", 'r') as f:
        template = f.read()
    
    for book, summary in tqdm.tqdm(data.items(), total=len(data), desc="Iterating over summaries"):
        if book in spans_questions:
            print(f"Skipping {book}")
        
        sentences = sent_tokenize(summary)
        for sentence in tqdm.tqdm(sentences, total=len(sentences), desc="Iterating over sentences"):
            print(f"SENTENCE:\n\n{sentence}\n")
            prompt = template.format(summary, sentence)
            max_len = 100
            print(f"MAX LEN: {max_len}")

            response = obtain_response(prompt, max_tokens=max_len, temperature=0, echo=False)
            response = response['choices'][0]['message']['content'].strip()
            print(f"RESPONSE:\n\n{response}\n")

            valid, questions, types = validate_response(response)

            while not valid:
                print("Invalid response, please try again")
                response = obtain_response(prompt, max_tokens=max_len, temperature=0, echo=False)
                print(f"RESPONSE:\n\n{response}\n")
                valid, questions, types = validate_response(response)
             
            if questions is not None:
                spans_questions[book][sentence] = {
                    'questions': questions,
                    'types': types
                }
                with open(save_path, 'w') as f:
                    json.dump(spans_questions, f)
        
        if len(spans_questions[book]) == 0:
            spans_questions[book] = None
            
        with open(save_path, 'w') as f:
            json.dump(spans_questions, f)


def get_booookscore():
    data = json.load(open(f"gpt4-annotations/{MODEL}-{CHUNK_SIZE}-{SUMMARY_STRATEGY}.json", 'r'))
    summary_data = json.load(open(f"summaries/{MODEL}-{CHUNK_SIZE}-{SUMMARY_STRATEGY}-cleaned.json", 'r'))
    data = {k: v for k, v in sorted(data.items(), key=lambda item: item[0])}

    scores = {}
    for book, summary in summary_data.items():
        if data[book] == None:
            scores[book] = 1
            continue
        sentences = sent_tokenize(summary)
        if type(data[book]) == dict and "singles" in data[book].keys():
            confusing_sentences = len(data[book]["singles"]) + len(data[book]["relations"])
        else:
            confusing_sentences = len(data[book])
        scores[book] = 1 - confusing_sentences / len(sentences)
    avg_score = np.mean(list(scores.values()))
    print(f"Average confusion score: {avg_score}")


get_annotations()
get_booookscore()

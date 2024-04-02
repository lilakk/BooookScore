import os
import numpy as np
import tqdm
import argparse
import json
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from utils import OpenAIClient

all_labels = ['entity omission', 'event omission', 'causal omission', 'salience', 'discontinuity', 'duplication', 'inconsistency', 'language']


class Scorer():
    def __init__(self, openai_key, model, bsize=None):
        self.client = OpenAIClient(openai_key, model)
        self.summary_path = args.summary_path
        self.annot_path = args.annot_path
        self.bsize = bsize

    def validate_response(self, response):
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
        types = types.lower()

        if "no confusion" in questions:
            if "no confusion" not in types:
                print("No confusion in questions but not in types")
                return False, [], []
            else:
                return True, None, None

        if types is not None:
            types = types.split(', ')
            for t in types:
                if t.lower() not in all_labels:
                    print(f"Invalid type: {t}")
                    return False, [], []
        
        if questions is not None and types is None:
            raise ValueError("Questions is not None but types is None")

        return True, questions, types

    def get_annot(self):
        summaries = json.load(open(self.summary_path, 'r'))
        annots = defaultdict(dict)
        if os.path.exists(self.annot_path):
            annots = json.load(open(self.annot_path, 'r'))
            annots = defaultdict(dict, annots)
            print(f"LOADED {len(annots)} annots FROM {self.annot_path}")

        with open("prompts/get_annotations.txt", 'r') as f:
            template = f.read()
        
        for book, summary in tqdm.tqdm(summaries.items(), total=len(summaries), desc="Iterating over summaries"):
            if book in annots:
                print(f"Skipping {book}")
            
            sentences = sent_tokenize(summary)
            for sentence in tqdm.tqdm(sentences, total=len(sentences), desc="Iterating over sentences"):
                print(f"SENTENCE:\n\n{sentence}\n")
                prompt = template.format(summary, sentence)
                max_len = 100
                print(f"MAX LEN: {max_len}")

                response = self.client.obtain_response(prompt, max_tokens=max_len, temperature=0)
                print(f"RESPONSE:\n\n{response}\n")

                valid, questions, types = validate_response(response)

                while not valid:
                    print("Invalid response, please try again")
                    response = self.client.obtain_response(prompt, max_tokens=max_len, temperature=0)
                    print(f"RESPONSE:\n\n{response}\n")
                    valid, questions, types = validate_response(response)
                
                if questions is not None:
                    annots[book][sentence] = {
                        'questions': questions,
                        'types': types
                    }
                    with open(self.annot_path, 'w') as f:
                        json.dump(annots, f)
            
            if len(annots[book]) == 0:
                annots[book] = None
                
            with open(self.annot_path, 'w') as f:
                json.dump(annots, f)

    def get_score(self):
        if not os.path.exists(self.annot_path):
            print("No annotations found, getting annotations...")
            self.get_annot()
        annots = json.load(open(self.annot_path, 'r'))
        annots = defaultdict(dict, annots)
        summaries = json.load(open(self.summary_path, 'r'))
        annots = {k: v for k, v in sorted(annots.items(), key=lambda item: item[0])}
        scores = {}
        for book, summary in summaries.items():
            if annots[book] == None:
                scores[book] = 1
                continue
            sentences = sent_tokenize(summary)
            if type(annots[book]) == dict and "singles" in annots[book].keys():
                confusing_sentences = len(annots[book]["singles"]) + len(annots[book]["relations"])
            else:
                confusing_sentences = len(annots[book])
            scores[book] = 1 - confusing_sentences / len(sentences)
        avg_score = np.mean(list(scores.values()))
        print(f"Average BooookScore: {avg_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", type=str, help="Path to the input data")
    parser.add_argument("--annot_path", type=str, help="Path to the annotated data")
    parser.add_argument("--openai_key", type=str, help="Path to the OpenAI key")
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--bsize", type=int, default=None, help="Set to None to disable batching")
    args = parser.parse_args()

    scorer = Scorer(args.openai_key, args.model, args.bsize)

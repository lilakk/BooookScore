import os
import numpy as np
import argparse
import json
import pickle
import time
import shutil
import traceback
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from typing import List, Any, Dict, Optional
from multiprocessing.pool import ThreadPool
from threading import Lock
from booookscore.utils import APIClient


class Scorer():
    def __init__(self,
        model,
        api,
        api_key,
        summ_path,
        annot_path,
        template_path,
        v2=False,
        batch_size=10
    ):
        self.client = APIClient(api, api_key, model)
        self.summ_path = summ_path
        self.annot_path = annot_path
        self.template_path = template_path
        self.v2 = v2
        self.batch_size = batch_size
        self.all_labels = ['entity omission', 'event omission', 'causal omission', 'salience', 'discontinuity', 'duplication', 'inconsistency', 'language']

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
                if t.lower() not in self.all_labels:
                    print(f"Invalid type: {t}")
                    return False, [], []
        
        if questions is not None and types is None:
            raise ValueError("Questions is not None but types is None")

        return True, questions, types

    def gen_batch(self, records: List[Any], batch_size: int):
        batch_start = 0
        while batch_start < len(records):
            batch_end = batch_start + batch_size
            batch = records[batch_start:batch_end]
            batch_start = batch_end
            yield batch

    def parse_response(self, response):
        start_index = response.find("[")
        end_index = response.rfind("]") + 1
        answers = json.loads(response[start_index:end_index])
        return answers

    def calc_instance(self, summary, sentences, template, num_retries, model_name):
        formatted_batch = "\n".join([f"{n+1}. {s}" for n, s in enumerate(sentences)])
        prompt = template.format(summary=summary, sentences=formatted_batch)
        for _ in range(num_retries):
            try:
                response = self.client.obtain_response(prompt, model_name=model_name)
                answers = self.parse_response(response)
                break
            except Exception:
                print(traceback.format_exc())
                time.sleep(10)
        assert len(answers) == len(sentences)
        for answer, sentence in zip(answers, sentences):
            answer["sentence"] = sentence
        return answers

    def create_callback(self, book, cache, cache_path, lock):
        def on_result(result):
            with lock:
                for answer in result:
                    sentence = answer.pop("sentence")
                    cache[book][sentence] = answer
                cache_temp_path = cache_path + ".tmp"
                with open(cache_temp_path, "w") as w:
                    json.dump(cache, w)
                shutil.move(cache_temp_path, cache_path)
        return on_result

    def get_annot(self, num_retries=3):
        assert self.summ_path and os.path.exists(self.summ_path), f"Summaries path {self.summ_path} does not exist"
        summaries = json.load(open(self.summ_path, 'r'))
        annots = defaultdict(dict)
        if os.path.exists(self.annot_path):
            annots = json.load(open(self.annot_path, 'r'))
            annots = defaultdict(dict, annots)
            print(f"LOADED {len(annots)} annots FROM {self.annot_path}")

        with open(template_path, 'r') as f:
            template = f.read()
        
        for book, summary in tqdm(summaries.items(), total=len(summaries), desc="Iterating over summaries"):
            if book in annots:
                print(f"Skipping {book}")
            
            sentences = sent_tokenize(summary)

            if not self.v2:
                for i, sentence in tqdm(enumerate(sentences), total=len(sentences), desc="Iterating over sentences"):
                    prompt = template.format(summary, sentence)
                    response = self.client.obtain_response(prompt, max_tokens=100, temperature=0)
                    valid, questions, types = self.validate_response(response)
                    while not valid:
                        response = self.client.obtain_response(prompt, max_tokens=max_len, temperature=0)
                        valid, questions, types = self.validate_response(response)
                    annots[book][sentence] = {
                        'questions': questions,
                        'types': types
                    }
            else:
                sentences = [s for s in sentences if s not in annots[book]]
                lock = Lock()
                batches = list(self.gen_batch(sentences, self.batch_size))
                tasks = [
                    (summary, batch, template, num_retries, self.client.model) for batch in batches
                ]
                with ThreadPool(2) as pool:
                    callback = self.create_callback(book, annots, self.annot_path, lock)
                    results = [
                        pool.apply_async(self.calc_instance, args=task, callback=callback)
                        for task in tasks
                    ]
                    results = [result.get() for result in results]
            
            with open(self.annot_path, 'w') as f:
                json.dump(annots, f)

    def get_score(self):
        if not os.path.exists(self.annot_path):
            print("No annotations found, getting annotations...")
            self.get_annot()
        annots = json.load(open(self.annot_path, 'r'))
        annots = defaultdict(dict, annots)
        annots = {k: v for k, v in sorted(annots.items(), key=lambda item: item[0])}
        scores = dict()
        for book, annot in annots.items():
            confusing_sentences = 0
            for sentence, sentence_annot in annot.items():
                if sentence_annot["questions"] or sentence_annot["types"]:
                    confusing_sentences += 1
            scores[book] = 1 - confusing_sentences / len(annots[book])
        avg_score = np.mean(list(scores.values()))
        return avg_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summ_path", type=str, help="must set if you don't have annotations yet")
    parser.add_argument("--annot_path", type=str, help="path to save annotations to")
    parser.add_argument("--api", type=str, help="api to use", choices=["openai", "anthropic", "together"])
    parser.add_argument("--api_key", type=str, help="path to a txt file storing your OpenAI api key")
    parser.add_argument("--model", type=str, default="gpt-4", help="evaluator model")
    parser.add_argument("--v2", action="store_true", help="use v2, which batches sentences during annotation (this setup was not used in the paper)")
    parser.add_argument("--batch_size", type=int, help="batch size if v2 is used")
    args = parser.parse_args()

    if args.v2:
        template_path = "prompts/get_annotations_v2.txt"
    else:
        template_path = "prompts/get_annotations.txt"
    scorer = Scorer(
        model=args.model,
        api=args.api,
        api_key=args.api_key,
        summ_path=args.summ_path,
        annot_path=args.annot_path,
        template_path=template_path,
        v2=args.v2,
        batch_size=args.batch_size
    )
    score = scorer.get_score()
    print(f"BooookScore = {score}")

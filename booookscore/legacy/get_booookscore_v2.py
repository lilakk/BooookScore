import os
import json
import time
import shutil
import traceback
from typing import List, Any, Dict, Optional
from multiprocessing.pool import ThreadPool
from threading import Lock

import fire
import tqdm
import numpy as np
from nltk.tokenize import sent_tokenize
from collections import defaultdict

from scripts.utils import obtain_response

DEFAULT_CACHE_DIR = "annotations/v2"
DEFAULT_MODEL_NAME = "gpt-4-turbo-preview"
DEFAULT_TEMPLATE_PATH = "prompts/get_annotations_v2.txt"
DEFAULT_BATCH_SIZE = 10
DEFAULT_PARALLEL_REQUESTS_COUNT = 2


def gen_batch(records: List[Any], batch_size: int):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start:batch_end]
        batch_start = batch_end
        yield batch


def parse_response(response):
    start_index = response.find("[")
    end_index = response.rfind("]") + 1
    answers = json.loads(response[start_index:end_index])
    return answers


def calc_instance(summary, sentences, template, num_retries, model_name):
    formatted_batch = "\n".join([f"{n+1}. {s}" for n, s in enumerate(sentences)])
    prompt = template.format(summary=summary, sentences=formatted_batch)
    for _ in range(num_retries):
        try:
            response = obtain_response(prompt, model_name=model_name)
            answers = parse_response(response)
            break
        except Exception:
            print(traceback.format_exc())
            time.sleep(10)
    assert len(answers) == len(sentences)
    for answer, sentence in zip(answers, sentences):
        answer["sentence"] = sentence
    return answers


def create_callback(book, cache, cache_path, lock):
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


def calc_booookscore_v2(
    summaries: Dict[str, str],
    cache_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    template_path: str = DEFAULT_TEMPLATE_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    parallel_requests_count: int = DEFAULT_PARALLEL_REQUESTS_COUNT,
    num_retries: int = 3,
):
    cache = defaultdict(dict)
    if os.path.exists(cache_path):
        with open(cache_path) as r:
            cache = json.load(r)
            cache = defaultdict(dict, cache)
        print(f"LOADED CACHE FROM {cache_path}")

    with open(template_path, "r") as f:
        template = f.read()

    for book, summary in tqdm.tqdm(
        summaries.items(), total=len(summaries), desc="Iterating over summaries"
    ):
        sentences = sent_tokenize(summary)
        sentences = [s for s in sentences if s not in cache[book]]

        lock = Lock()
        batches = list(gen_batch(sentences, batch_size))
        tasks = [
            (summary, batch, template, num_retries, model_name) for batch in batches
        ]
        with ThreadPool(parallel_requests_count) as pool:
            callback = create_callback(book, cache, cache_path, lock)
            results = [
                pool.apply_async(calc_instance, args=task, callback=callback)
                for task in tasks
            ]
            # Waiting for all results
            results = [result.get() for result in results]

    scores = dict()
    for book, answers in cache.items():
        confusing_sentences = 0
        for sentence, sentence_answers in answers.items():
            if sentence_answers["questions"] or sentence_answers["types"]:
                confusing_sentences += 1
        scores[book] = 1 - confusing_sentences / len(answers)
    avg_score = np.mean(list(scores.values()))
    return avg_score


def get_booookscore_v2(
    input_path: str,
    cache_path: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    template_path: str = DEFAULT_TEMPLATE_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    parallel_requests_count: int = DEFAULT_PARALLEL_REQUESTS_COUNT,
):
    if cache_path is None:
        cache_path = os.path.join(DEFAULT_CACHE_DIR, os.path.basename(input_path))
    with open(input_path) as r:
        summaries = json.load(r)
    avg_score = calc_booookscore_v2(
        summaries=summaries,
        cache_path=cache_path,
        model_name=model_name,
        template_path=template_path,
        batch_size=batch_size,
        parallel_requests_count=parallel_requests_count,
    )
    print(f"Average confusion score: {avg_score}")


if __name__ == "__main__":
    fire.Fire(get_booookscore_v2)

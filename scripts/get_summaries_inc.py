import os
import pickle
import json
import tqdm
import argparse
from utils import obtain_response, count_tokens


def compress(response, summary, chunk, templates, summary_len, word_limit, num_chunks, j):
    chunk_trims = 0
    compressed_summary = None
    summary_words = len(summary.split())
    ori_expected_words = int(word_limit * j / num_chunks)  # no need to be j + 1 since we're compressing the summary at the previous chunk
    expected_words = ori_expected_words
    actual_words = expected_words
    
    dic = {}  # keep track of each trimmed summary and their actual number of words

    while response[-1] not in ['.', '?', '!', '\"', '\''] \
    or count_tokens(response) >= summary_len \
    or actual_words < int(expected_words * 0.8) or actual_words > int(expected_words * 1.2):
        if chunk_trims == 6:
            print(f"\nCOMPRESSION FAILED AFTER 6 ATTEMPTS, SKIPPING\n")
            if not all([v['valid_response'] == False for v in dic.values()]):
                dic = {k: v for k, v in dic.items() if v['valid_response'] == True}
            print(f"DICTIONARY LENGTH: {len(dic)}")
            closest_key = min(dic, key=lambda x:abs(x-ori_expected_words))  # find the trimmed summary with actual # words closest to the expected # words
            print(f"EXPECTED WORDS: {ori_expected_words} | CLOSEST KEY: {closest_key} | ALL KEYS: {dic.keys()}")

            return dic[closest_key]['compressed_summary'], dic[closest_key]['response'], chunk_trims, 1
        
        print(f"\nCOMPRESSION REQUIRED, ATTEMPT {chunk_trims + 1}\n")
        print(f"MAX LEN: {summary_len} | ACTUAL LEN: {count_tokens(response)}")

        expected_words = int(ori_expected_words * (1 - chunk_trims * 0.05))
        prompt = templates["compress_template"].format(summary, summary_words, expected_words, expected_words)

        response = obtain_response(prompt, max_tokens=summary_len, temperature=1)
        compressed_summary = response
        print(f"TRIMMED SUMMARY: {compressed_summary}\n")
        actual_words = len(compressed_summary.split())
        current_tokens = count_tokens(compressed_summary)
        print(f"EXPECTED WORDS: {expected_words} | ACTUAL WORDS: {actual_words} | CURRENT TOKENS: {current_tokens}\n\n")

        if compressed_summary[-1] not in ['.', '?', '!', '\"', '\''] \
        or count_tokens(compressed_summary) >= summary_len \
        or actual_words < int(expected_words * 0.8) or actual_words > int(expected_words * 1.2):
            print(f"INVALID TRIMMED SUMMARY, CONTINUE TO NEXT ATTEMPT\n\n")
            chunk_trims += 1
            continue
        
        num_words = int(word_limit * (j + 1) / num_chunks)
        prompt = templates['template'].format(chunk, compressed_summary, num_words, num_words)
        response = obtain_response(prompt, max_tokens=summary_len, temperature=0.5)

        dic[actual_words] = {
            'compressed_summary': compressed_summary,
            'response': response,
            'valid_response': response[-1] in ['.', '?', '!', '\"', '\''] \
            and count_tokens(response) < summary_len
        }
        print(f"VALID_RESPONSE: {dic[actual_words]['valid_response']}")
        chunk_trims += 1

    return compressed_summary, response, chunk_trims, 0


def get_summaries():
    data = pickle.load(open(INPUT_PATH, 'rb'))

    with open("prompts/get_summaries_inc/init.txt", "r") as f:
        init_template = f.read()
    with open("prompts/get_summaries_inc/intermediate.txt", "r") as f:
        template = f.read()
    with open("prompts/get_summaries_inc/compress.txt", "r") as f:
        compress_template = f.read()

    num_trims = 0
    total_chunks = 0
    skipped_chunks = 0

    new_data = {}
    if os.path.exists(SAVE_PATH):
        new_data = json.load(open(SAVE_PATH, "r"))
    
    for i, book in tqdm.tqdm(enumerate(data), total=len(data), desc="Iterating over books"):
        if book in new_data and len(new_data[book]) >= len(data[book]):
            print(f"Skipping {book} because it already exists in {path}")
            continue
        total_chunks += len(data[book])
        new_chunks = []
        prev_summary = None
        if len(new_data) > i:
            new_chunks = new_data[book]
            prev_summary = new_chunks[-1]
        dd = data[book]
        summary_len = min(MAX_SUMMARY_LEN, 1200)
        word_limit = int(summary_len * WORD_RATIO)
        num_chunks = len(dd)
        print(f"Book {book} max summary length: {summary_len}, word limit: {word_limit}")
        
        for j, chunk in tqdm.tqdm(enumerate(dd), total=len(dd), desc="Iterating over chunks"):
            if j < len(new_chunks):
                print(f"Skipping chunk {j}...")
                continue
            new_chunk = {}
            num_words = int(word_limit * (j + 1) / len(dd))
            if prev_summary is None:
                prompt = init_template.format(chunk)
            else:
                prompt = template.format(chunk, prev_summary)
            
            response = obtain_response(prompt, max_tokens=summary_len, temperature=0.5)
            print(f"\n\nCHUNK SUMMARY:\n{response}\n\n")
            actual_words = len(response.split())
            print(f"ACTUAL WORDS: {actual_words}")
            
            # compress prev_summary if the current one is too long or doesn't end in punctuation
            if prev_summary is not None and (response[-1] not in ['.', '?', '!', '\"', '\''] \
            or count_tokens(response) >= summary_len):
                templates = {
                    "template": template,
                    "compress_template": compress_template
                }
                compressed_summary, response, chunk_trims, skipped = compress(response, prev_summary, chunk, templates, summary_len, word_limit, num_chunks, j)
                num_trims += chunk_trims
                skipped_chunks += skipped
                new_chunks[j - 1] = compressed_summary

            prev_summary = response
            new_chunks.append(response)

            if (j + 1) % 5 == 0:
                new_data[book] = new_chunks
                print(f"saving data for book {i} at chunk {j}...")
                json.dump(new_data, open(SAVE_PATH, 'w'))
            
        new_data[book] = new_chunks
        json.dump(new_data, open(SAVE_PATH, 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="path to the pickle file containing the chunked data")
    parser.add_argument("--save_path", type=str, help="path to the json file to save the data")
    parser.add_argument("--max_context_len", type=int, help="max content length of the model")
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--max_summary_len", type=int, default=900, help="max length of the final summary")
    args = parser.parse_args()

    INPUT_PATH = args.input_path
    SAVE_PATH = args.save_path
    MAX_CONTEXT_LEN = args.max_context_len
    MAX_SUMMARY_LEN = args.max_summary_len
    CHUNK_SIZE = args.chunk_size
    WORD_RATIO = 0.65

    get_summaries()

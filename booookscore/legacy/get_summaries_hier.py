import os
import pickle
from transformers import GPT2Tokenizer
import tqdm
import time
import argparse
import json
import math
from collections import defaultdict
from utils import obtain_response, count_tokens


def check_summary_validity(summary, token_limit):
    if len(summary) == 0:
        raise ValueError("Empty summary returned")
    if count_tokens(summary) > token_limit or summary[-1] not in ['.', '?', '!', '\"', '\'']:
        return False
    else:
        return True


def summarize(texts, token_limit, level):
    text = texts['text']
    context = texts['context']
    word_limit = round(token_limit * WORD_RATIO)
    if level == 0:
        prompt = init_template.format(text, word_limit)
    else:
        prompt = template.format(text, word_limit)
        if len(context) > 0 and level > 0:
            prompt = context_template.format(context, text, word_limit)
    print(f"PROMPT:\n\n---\n\n{prompt}\n\n---\n\n")
    response = obtain_response(prompt, max_tokens=token_limit, temperature=0.5)
    print(f"SUMMARY:\n\n---\n\n{response}\n\n---\n\n")

    while len(response) == 0:
        print("Empty summary, retrying in 10 seconds...")
        time.sleep(10)
        response = obtain_response(prompt, max_tokens=token_limit, temperature=0.5)
        print(f"SUMMARY:\n\n---\n\n{response}\n\n---\n\n")

    attempts = 0
    while not check_summary_validity(response, token_limit):
        word_limit = word_limit * (1 - 0.1 * attempts)
        if level == 0:
            prompt = init_template.format(text, word_limit)
        else:
            prompt = template.format(text, word_limit)
            if len(context) > 0 and level > 0:
                prompt = context_template.format(context, text, word_limit)
        if attempts == 6:
            print("Failed to generate valid summary after 6 attempts, skipping")
            return response
        print(f"Invalid summary, retrying: attempt {attempts}")
        response = obtain_response(prompt, max_tokens=token_limit, temperature=1)
        print(f"SUMMARY:\n\n---\n\n{response}\n\n---\n\n")
        attempts += 1
    return response


def estimate_levels(book_chunks, summary_limit=450):
    num_chunks = len(book_chunks)
    chunk_limit = CHUNK_SIZE
    levels = 0

    while num_chunks > 1:
        chunks_that_fit = (MAX_CONTEXT_LEN - count_tokens(template.format('', 0)) - 20) // chunk_limit  # number of chunks that could fit into the current context
        num_chunks = math.ceil(num_chunks / chunks_that_fit)  # number of chunks after merging
        chunk_limit = summary_limit
        levels += 1

    summary_limits = [MAX_SUMMARY_LEN]
    for _ in range(levels-1):
        summary_limits.append(int(summary_limits[-1] * WORD_RATIO))
    summary_limits.reverse()  # since we got the limits from highest to lowest, but we need them from lowest to highest
    return levels, summary_limits


def recursive_summary(book, summaries, level, chunks, summary_limits):
    """
    Merges chunks into summaries recursively until the summaries are small enough to be summarized in one go.

    chunks: list of chunks
    level: current level
    summaries_dict: dictionary of summaries for each level
    summary_limits: list of summary limits for each level
    """
    print(f"Level {level} has {len(chunks)} chunks")
    i = 0
    if level == 0 and len(summaries[book]['summaries_dict'][0]) > 0:
        # resume from the last chunk
        i = len(summaries[book]['summaries_dict'][0])
    if level >= len(summary_limits):  # account for underestimates
        summary_limit = MAX_SUMMARY_LEN
    else:
        summary_limit = summary_limits[level]
    
    summaries_dict = summaries[book]['summaries_dict']

    if level > 0 and len(summaries_dict[level]) > 0:
        if count_tokens('\n\n'.join(chunks)) + MAX_SUMMARY_LEN + count_tokens(context_template.format('', '', 0)) + 20 <= MAX_CONTEXT_LEN:  # account for overestimates
            summary_limit = MAX_SUMMARY_LEN
        num_tokens = MAX_CONTEXT_LEN - summary_limit - count_tokens(context_template.format('','', 0)) - 20  # Number of tokens left for context + concat
    else:
        if count_tokens('\n\n'.join(chunks)) + MAX_SUMMARY_LEN + count_tokens(template.format('', 0)) + 20 <= MAX_CONTEXT_LEN:
            summary_limit = MAX_SUMMARY_LEN
        num_tokens = MAX_CONTEXT_LEN - summary_limit - count_tokens(template.format('', 0)) - 20

    while i < len(chunks):
        context = ""
        # Generate previous level context
        context = summaries_dict[level][-1] if len(summaries_dict[level]) > 0 else ""
        context_len = math.floor(0.2 * num_tokens)
        if count_tokens(context) > context_len:
            context_tokens = tokenizer.encode(context)[:context_len]
            context = tokenizer.decode(context_tokens)
            if '.' in context:
                context = context.rsplit('.', 1)[0] + '.'
        
        texts = {}
        # Concatenate as many chunks as possible
        if level == 0:
            text = chunks[i]
        else:
            j = 1
            text = f"Summary {j}:\n\n{chunks[i]}"
            while i + 1 < len(chunks) and count_tokens(context + text + f"\n\nSummary {j+1}:\n\n{chunks[i+1]}") + 20 <= num_tokens:
                i += 1
                j += 1
                text += f"\n\nSummary {j}:\n\n{chunks[i]}"
        texts = {
            'text': text,
            'context': context
        }

        # Calling the summarize function to produce the summaries
        print(f"Level {level} chunk {i}")
        print(f"Summary limit: {summary_limit}")
        summary = summarize(texts, summary_limit, level)
        summaries_dict[level].append(summary)
        i += 1

        json.dump(summaries, open(SAVE_PATH, 'w'))

    # If the summaries still too large, recursively call the function for the next level
    if len(summaries_dict[level]) > 1:
        # save the current level summaries
        return recursive_summary(book, summaries, level + 1, summaries_dict[level], summary_limits)
    else:
        return summaries_dict[level][0]  # the final summary


def summarize_book(book, chunks, summaries):
    levels, summary_limits = estimate_levels(chunks)
    print(f"Book {book} has {levels} levels by estimate")
    print(f"Summary limits: {summary_limits}")
    
    level = 0
    if len(summaries[book]['summaries_dict']) > 0:
        if len(summaries[book]['summaries_dict']) == 1:  # if there is only one level so far
            if len(summaries[book]['summaries_dict'][0]) == len(chunks):  # if level 0 is finished, set level to 1
                level = 1
            elif len(summaries[book]['summaries_dict'][0]) < len(chunks):  # else, resume at level 0
                level = 0
            else:
                raise ValueError(f"Invalid summaries_dict at level 0 for {book}")
        else:  # if there're more than one level so far, resume at the last level
            level = len(summaries[book]['summaries_dict'])
        print(f"Resuming at level {level}")
    
    final_summary = recursive_summary(book, summaries, level, chunks, summary_limits)
    
    return final_summary, summaries


def get_hierarchical_summaries():
    data = pickle.load(open(INPUT_PATH, 'rb'))
    summaries = defaultdict(dict)
    if os.path.exists(SAVE_PATH):
        print("Loading existing summaries...")
        summaries = json.load(open(SAVE_PATH, 'r'))
        # convert all keys into int
        for book in summaries:
            summaries[book]['summaries_dict'] = defaultdict(list, {int(k): v for k, v in summaries[book]['summaries_dict'].items()})

    for i, book in tqdm.tqdm(enumerate(data), total=len(data), desc="Iterating over books"):
        if book in summaries and 'final_summary' in summaries[book]:
            print("Already processed, skipping...")
            continue
        chunks = data[book]
        if book in summaries and 'summaries_dict' in summaries[book]:
            if len(summaries[book]['summaries_dict']) == 1 and len(summaries[book]['summaries_dict'][0]) < len(chunks):
                level = 0
            elif len(summaries[book]['summaries_dict']) == 1 and len(summaries[book]['summaries_dict'][0]) == len(chunks):
                level = len(summaries[book]['summaries_dict']) - 1
                chunks = summaries[book]['summaries_dict'][level]
        else:
            summaries[book] = {
                'summaries_dict': defaultdict(list)
            }
        final_summary, summaries = summarize_book(book, chunks, summaries)
        summaries[book]['final_summary'] = final_summary
        with open(SAVE_PATH, 'w') as f:
            json.dump(summaries, f)


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="path to the pickle file containing the chunked data")
    parser.add_argument("--save_path", type=str, help="path to the json file to save the data")
    parser.add_argument("--max_context_len", type=int, help="max content length of the model")
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--max_summary_len", type=int, default=900, help="max length of the final summary")
    args = parser.parse_args()

    INPUT_PATH = args.input_path
    SAVE_PATH = args.save_path
    CHUNK_SIZE = args.chunk_size
    MAX_CONTEXT_LEN = args.max_context_len
    MAX_SUMMARY_LEN = args.max_summary_len
    WORD_RATIO = 0.65

    init_template = open("prompts/get_summaries_hier/init.txt", "r").read()
    template = open("prompts/get_summaries_hier/merge.txt", "r").read()
    context_template = open("prompts/get_summaries_hier/merge_context.txt", "r").read()

    get_hierarchical_summaries()

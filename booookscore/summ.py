import os
import pickle
import time
import argparse
import json
import math
import tiktoken
from tqdm import tqdm
from collections import defaultdict
from utils import APIClient, count_tokens


class Summarizer():
    def __init__(self,
        model,
        api,
        api_key,
        summ_path,
        method,
        chunk_size,
        max_context_len,
        max_summary_len,
        word_ratio=0.65
    ):
        self.client = APIClient(api, api_key, model)
        self.summ_path = summ_path
        assert method in ['inc', 'hier']
        self.method = method
        self.chunk_size = chunk_size
        self.max_context_len = max_context_len
        self.max_summary_len = max_summary_len
        self.word_ratio = word_ratio
    
    def check_summary_validity(self, summary, token_limit):
        if len(summary) == 0:
            raise ValueError("Empty summary returned")
        if count_tokens(summary) > token_limit or summary[-1] not in ['.', '?', '!', '\"', '\'']:
            return False
        else:
            return True

    def summarize(self, texts, token_limit, level):
        text = texts['text']
        context = texts['context']
        word_limit = round(token_limit * self.word_ratio)
        if level == 0:
            prompt = self.templates['init_template'].format(text, word_limit)
        else:
            prompt = self.templates['template'].format(text, word_limit)
            if len(context) > 0 and level > 0:
                prompt = self.templates['context_template'].format(context, text, word_limit)
        response = self.client.obtain_response(prompt, max_tokens=token_limit, temperature=0.5)

        while len(response) == 0:
            print("Empty summary, retrying in 10 seconds...")
            time.sleep(10)
            response = self.client.obtain_response(prompt, max_tokens=token_limit, temperature=0.5)

        attempts = 0
        while not self.check_summary_validity(response, token_limit):
            word_limit = word_limit * (1 - 0.1 * attempts)
            if level == 0:
                prompt = self.templates['init_template'].format(text, word_limit)
            else:
                prompt = self.templates['template'].format(text, word_limit)
                if len(context) > 0 and level > 0:
                    prompt = self.templates['context_template'].format(context, text, word_limit)
            if attempts == 6:
                print("Failed to generate valid summary after 6 attempts, skipping")
                return response
            print(f"Invalid summary, retrying: attempt {attempts}")
            response = self.client.obtain_response(prompt, max_tokens=token_limit, temperature=1)
            attempts += 1
        return response

    def estimate_levels(self, book_chunks, summary_limit=450):
        num_chunks = len(book_chunks)
        chunk_limit = self.chunk_size
        levels = 0

        while num_chunks > 1:
            chunks_that_fit = (self.max_context_len - count_tokens(self.templates['template'].format('', 0)) - 20) // chunk_limit  # number of chunks that could fit into the current context
            num_chunks = math.ceil(num_chunks / chunks_that_fit)  # number of chunks after merging
            chunk_limit = summary_limit
            levels += 1

        summary_limits = [self.max_summary_len]
        for _ in range(levels-1):
            summary_limits.append(int(summary_limits[-1] * self.word_ratio))
        summary_limits.reverse()  # since we got the limits from highest to lowest, but we need them from lowest to highest
        return levels, summary_limits

    def recursive_summary(self, book, summaries, level, chunks, summary_limits):
        """
        Merges chunks into summaries recursively until the summaries are small enough to be summarized in one go.

        chunks: list of chunks
        level: current level
        summaries_dict: dictionary of summaries for each level
        summary_limits: list of summary limits for each level
        """
        i = 0
        if level == 0 and len(summaries[book]['summaries_dict'][0]) > 0:
            # resume from the last chunk
            i = len(summaries[book]['summaries_dict'][0])
        if level >= len(summary_limits):  # account for underestimates
            summary_limit = self.max_summary_len
        else:
            summary_limit = summary_limits[level]
        
        summaries_dict = summaries[book]['summaries_dict']

        if level > 0 and len(summaries_dict[level]) > 0:
            if count_tokens('\n\n'.join(chunks)) + self.max_summary_len + count_tokens(self.templates['context_template'].format('', '', 0)) + 20 <= self.max_context_len:  # account for overestimates
                summary_limit = self.max_summary_len
            num_tokens = self.max_context_len - summary_limit - count_tokens(self.templates['context_template'].format('','', 0)) - 20  # Number of tokens left for context + concat
        else:
            if count_tokens('\n\n'.join(chunks)) + self.max_summary_len + count_tokens(self.templates['template'].format('', 0)) + 20 <= self.max_context_len:
                summary_limit = self.max_summary_len
            num_tokens = self.max_context_len - summary_limit - count_tokens(self.templates['template'].format('', 0)) - 20

        while i < len(chunks):
            context = ""
            # Generate previous level context
            context = summaries_dict[level][-1] if len(summaries_dict[level]) > 0 else ""
            context_len = math.floor(0.2 * num_tokens)
            if count_tokens(context) > context_len:
                context_tokens = encoding.encode(context)[:context_len]
                context = encoding.decode(context_tokens)
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
            summary = self.summarize(texts, summary_limit, level)
            summaries_dict[level].append(summary)
            i += 1

        # If the summaries still too large, recursively call the function for the next level
        if len(summaries_dict[level]) > 1:
            # save the current level summaries
            return self.recursive_summary(book, summaries, level + 1, summaries_dict[level], summary_limits)
        else:
            return summaries_dict[level][0]  # the final summary

    def summarize_book(self, book, chunks, summaries):
        levels, summary_limits = self.estimate_levels(chunks)
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
        
        final_summary = self.recursive_summary(book, summaries, level, chunks, summary_limits)
        
        return final_summary, summaries

    def get_hierarchical_summaries(self, book_path):
        data = pickle.load(open(book_path, 'rb'))
        self.templates = {
            'init_template': open("prompts/get_summaries_hier/init.txt", "r").read(),
            'template': open("prompts/get_summaries_hier/merge.txt", "r").read(),
            'context_template': open("prompts/get_summaries_hier/merge_context.txt", "r").read()
        }
        summaries = defaultdict(dict)
        if os.path.exists(self.summ_path):
            print("Loading existing summaries...")
            summaries = json.load(open(self.summ_path, 'r'))
            # convert all keys into int
            for book in summaries:
                summaries[book]['summaries_dict'] = defaultdict(list, {int(k): v for k, v in summaries[book]['summaries_dict'].items()})

        for i, book in tqdm(enumerate(data), total=len(data), desc="Iterating over books"):
            if book in summaries:
                print("Already processed, skipping...")
                continue
            chunks = data[book]
            summaries[book] = {
                'summaries_dict': defaultdict(list)
            }
            final_summary, summaries = self.summarize_book(book, chunks, summaries)
            summaries[book]['final_summary'] = final_summary
            with open(self.summ_path, 'w') as f:
                json.dump(summaries, f)

    def compress(self, response, summary, chunk, summary_len, word_limit, num_chunks, j):
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
                closest_key = min(dic, key=lambda x:abs(x-ori_expected_words))  # find the trimmed summary with actual # words closest to the expected # words
                return dic[closest_key]['compressed_summary'], dic[closest_key]['response'], chunk_trims, 1
            
            expected_words = int(ori_expected_words * (1 - chunk_trims * 0.05))
            prompt = self.templates["compress_template"].format(summary, summary_words, expected_words, expected_words)

            response = self.client.obtain_response(prompt, max_tokens=summary_len, temperature=1)
            compressed_summary = response
            actual_words = len(compressed_summary.split())
            current_tokens = count_tokens(compressed_summary)

            if compressed_summary[-1] not in ['.', '?', '!', '\"', '\''] \
            or count_tokens(compressed_summary) >= summary_len \
            or actual_words < int(expected_words * 0.8) or actual_words > int(expected_words * 1.2):
                chunk_trims += 1
                continue
            
            num_words = int(word_limit * (j + 1) / num_chunks)
            prompt = self.templates['template'].format(chunk, compressed_summary, num_words, num_words)
            response = self.client.obtain_response(prompt, max_tokens=summary_len, temperature=0.5)

            dic[actual_words] = {
                'compressed_summary': compressed_summary,
                'response': response,
                'valid_response': response[-1] in ['.', '?', '!', '\"', '\''] \
                and count_tokens(response) < summary_len
            }
            chunk_trims += 1

        return compressed_summary, response, chunk_trims, 0

    def get_incremental_summaries(self, book_path):
        data = pickle.load(open(book_path, 'rb'))
        self.templates = {
            "init_template": open("prompts/get_summaries_inc/init.txt", "r").read(),
            "template": open("prompts/get_summaries_inc/intermediate.txt", "r").read(),
            "compress_template": open("prompts/get_summaries_inc/compress.txt", "r").read()
        }

        num_trims = 0
        total_chunks = 0
        skipped_chunks = 0

        new_data = {}
        if os.path.exists(self.summ_path):
            new_data = json.load(open(self.summ_path, "r"))
        
        for i, book in tqdm(enumerate(data), total=len(data), desc="Iterating over books"):
            if book in new_data and len(new_data[book]) >= len(data[book]):
                print(f"Skipping {book}")
                continue
            total_chunks += len(data[book])
            new_chunks = []
            prev_summary = None
            if len(new_data) > i:
                new_chunks = new_data[book]
                prev_summary = new_chunks[-1]
            dd = data[book]
            word_limit = int(self.max_summary_len * self.word_ratio)
            num_chunks = len(dd)
            
            for j, chunk in tqdm(enumerate(dd), total=len(dd), desc="Iterating over chunks"):
                if j < len(new_chunks):
                    print(f"Skipping chunk {j}...")
                    continue
                new_chunk = {}
                num_words = int(word_limit * (j + 1) / len(dd))
                if prev_summary is None:
                    prompt = self.templates['init_template'].format(chunk)
                else:
                    prompt = self.templates['template'].format(chunk, prev_summary)
                
                response = self.client.obtain_response(prompt, max_tokens=self.max_summary_len, temperature=0.5)
                actual_words = len(response.split())
                
                # compress prev_summary if the current one is too long or doesn't end in punctuation
                if prev_summary is not None and (response[-1] not in ['.', '?', '!', '\"', '\''] \
                or count_tokens(response) >= self.max_summary_len):
                    compressed_summary, response, chunk_trims, skipped = self.compress(response, prev_summary, chunk, self.max_summary_len, word_limit, num_chunks, j)
                    num_trims += chunk_trims
                    skipped_chunks += skipped
                    new_chunks[j - 1] = compressed_summary

                prev_summary = response
                new_chunks.append(response)

                if (j + 1) % 10 == 0:
                    new_data[book] = new_chunks
                    json.dump(new_data, open(self.summ_path, 'w'))
                
            new_data[book] = new_chunks
            json.dump(new_data, open(self.summ_path, 'w'))

    def get_summaries(self, book_path):
        if self.method == 'inc':
            self.get_incremental_summaries(book_path)
        elif self.method == 'hier':
            self.get_hierarchical_summaries(book_path)
        else:
            raise ValueError("Invalid method")


if __name__ == "__main__":
    encoding = tiktoken.get_encoding('cl100k_base')

    parser = argparse.ArgumentParser()
    parser.add_argument("--book_path", type=str, help="path to the file containing the chunked data")
    parser.add_argument("--summ_path", type=str, help="path to the json file to save the data")
    parser.add_argument("--model", type=str, help="summarizer model")
    parser.add_argument("--api", type=str, help="api to use", choices=["openai", "anthropic", "together"])
    parser.add_argument("--api_key", type=str, help="path to a txt file storing your OpenAI api key")
    parser.add_argument("--method", type=str, help="method for summarization", choices=['inc', 'hier'])
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--max_context_len", type=int, help="max content length of the model")
    parser.add_argument("--max_summary_len", type=int, default=900, help="max length of the final summary")
    args = parser.parse_args()

    summarizer = Summarizer(
        args.model,
        args.api,
        args.api_key,
        args.summ_path,
        args.method,
        args.chunk_size,
        args.max_context_len,
        args.max_summary_len
    )
    summarizer.get_summaries(args.book_path)

import os
import pickle
import torch
from transformers import GPT2Tokenizer
import tqdm
import argparse

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

parser = argparse.ArgumentParser()
parser.add_argument("--chunk_size", type=int, default=2048)
args = parser.parse_args()

CHUNK_SIZE = args.chunk_size

save_path = f"data/all_books_chunked_{CHUNK_SIZE}.pkl"


def find_puncutations(text, comma=False):
    if comma:
        puncs = ['.', '?', '!', ',']
    else:
        puncs = ['.', '?', '!']
    puncs_idx = []
    for i, c in enumerate(text):
        if c in puncs:
            puncs_idx.append(i)
    return puncs_idx


def truncate(text, chunk_size):
    ori_text = text
    ori_len = len(text)
    last_punc = 0
    if "." in text:
        last_punc = max(last_punc, text.rindex("."))
    if "?" in text:
        last_punc = max(last_punc, text.rindex("?"))
    if "!" in text:
        last_punc = max(last_punc, text.rindex("!"))
    if last_punc != 0:
        text = text[:last_punc + 1]
    while len(tokenizer(text)['input_ids']) > chunk_size:
        puncs_idx = find_puncutations(text)
        try:
            text = text[:puncs_idx[-2] + 1]
        except:
            puncs_idx = find_puncutations(text, comma=True)
            text = text[:puncs_idx[-2] + 1]
    new_len = len(text)
    diff = ori_len - new_len
    truncated = ori_text[new_len:]
    return text, truncated


def process_books(path, chunk_size):
    books = pickle.load(open('data/all_books.pkl', 'rb'))

    new_data = {}
    if os.path.exists(path):
        new_data = pickle.load(open(path, 'rb'))
    
    for i, book in tqdm.tqdm(enumerate(books), total=len(books)):
        if book in new_data:
            print("Already processed, skipping...")
            continue
        doc = books[book]
        encodings = tokenizer(doc, return_tensors='pt')['input_ids'].squeeze()
        encodings = torch.split(encodings, chunk_size)
        chunks = []
        chunk_truncated = None
        for j, c in tqdm.tqdm(enumerate(encodings), total=len(encodings), desc=f"Processing {i}th instance"):
            chunk_text = tokenizer.decode(c)
            if chunk_truncated is not None:
                chunk_text = chunk_truncated + chunk_text
            chunk_text, chunk_truncated = truncate(chunk_text, chunk_size)
            chunks.append(chunk_text)
            assert len(tokenizer(chunk_text)['input_ids']) <= chunk_size
        
        while len(chunk_truncated) > 0:
            if len(tokenizer(chunk_truncated)['input_ids']) > chunk_size:
                remaining_encodings = tokenizer(chunk_truncated, return_tensors='pt')['input_ids'].squeeze()
                remaining_encodings = torch.split(remaining_encodings, chunk_size)
                remaining_chunks = []
                chunk_truncated = ''
                for j, c in tqdm.tqdm(enumerate(remaining_encodings), total=len(remaining_encodings), desc=f"Processing {i}th instance {j}th remaining chunk"):
                    chunk_text = tokenizer.decode(c)
                    if len(chunk_truncated) > 0:
                        chunk_text = chunk_truncated + chunk_text
                    chunk_text, chunk_truncated = truncate(chunk_text, chunk_size)
                    remaining_chunks.append(chunk_text)
                    assert len(tokenizer(chunk_text)['input_ids']) <= chunk_size
                assert len(tokenizer(chunk_truncated)['input_ids']) <= chunk_size
                remaining_chunks.append(chunk_truncated)
                chunks.extend(remaining_chunks)
            else:
                chunks.append(chunk_truncated)
                break
        if len(chunks[-1]) < 30:
            chunks.pop()
        print(len(tokenizer.decode(tokenizer(doc, return_tensors='pt')['input_ids'].squeeze())))
        print(len(''.join(chunks)))
        assert len(tokenizer.decode(tokenizer(doc, return_tensors='pt')['input_ids'].squeeze())) - len(''.join(chunks)) < 30
        new_data[book] = chunks
        with open(path, 'wb') as f:
            pickle.dump(new_data, f)
    pickle.dump(new_data, open(path, 'wb'))
    return new_data


path = f'data/all_books_chunked_{CHUNK_SIZE}.pkl'
process_books(path, CHUNK_SIZE)

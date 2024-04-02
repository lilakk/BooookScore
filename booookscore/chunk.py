import os
import torch
import pickle
from tqdm import tqdm
import argparse
from booookscore.utils import count_tokens


def find_punctuations(text, comma=False):
    if comma:
        puncs = ['.', '?', '!', ',', '."', '?"', '!"', ".'", "?'", "!'"]
    else:
        puncs = ['.', '?', '!', '."', '?"', '!"', ".'", "?'", "!'"]
    
    puncs_idx = []
    for i, c in enumerate(text):
        if c in puncs:
            puncs_idx.append(i)
        elif c == '"' or c == "'":
            if i > 0 and text[i-1] in ['.', '?', '!']:
                puncs_idx.append(i)
    
    return puncs_idx


def truncate(text, chunk_size):
    ori_text = text
    ori_len = len(text)
    
    while count_tokens(text) > chunk_size:
        puncs_idx = find_punctuations(text)
        try:
            text = text[:puncs_idx[-2] + 1]
        except:
            puncs_idx = find_punctuations(text, comma=True)
            try:
                text = text[:puncs_idx[-2] + 1]
            except:
                return text, ''
    
    new_len = len(text)
    diff = ori_len - new_len
    truncated = ori_text[new_len:]
    
    return text, truncated


def chunk_text(paragraphs, chunk_size):
    chunks = []
    curr_chunk = ''

    for p in tqdm(paragraphs, total=len(paragraphs)):
        new_chunk = '\n'.join([curr_chunk, p]) if len(curr_chunk) > 0 else p

        # if a single paragraph is too long, split it into smaller chunks
        if count_tokens(p) > chunk_size:
            curr_chunk, chunk_truncated = truncate(new_chunk, chunk_size)
            chunks.append(curr_chunk)
            while count_tokens(chunk_truncated) > chunk_size:
                curr_chunk, chunk_truncated = truncate(chunk_truncated, chunk_size)
                chunks.append(curr_chunk)
            curr_chunk = chunk_truncated
            continue
        
        if count_tokens(new_chunk) > chunk_size:
            chunks.append(curr_chunk)
            curr_chunk = p
        else:
            curr_chunk = new_chunk

    if len(curr_chunk) > 0:
        chunks.append(curr_chunk)

    return chunks


def process_books(books, chunk_size, output_path):
    new_data = {}
    if os.path.exists(output_path):
        new_data = pickle.load(open(output_path, 'rb'))
    for i, book in tqdm(enumerate(books), total=len(books)):
        if book in new_data:
            print("Already processed, skipping...")
            continue
        doc = books[book]
        paragraphs = doc.split("\n")
        if not args.include_empty_lines:
            paragraphs = [p for p in paragraphs if len(p) > 0]
        if count_tokens('\n'.join(paragraphs)) <= chunk_size:
            new_data[book] = ['\n'.join(paragraphs)]
        else:
            chunks = chunk_text(paragraphs, chunk_size)
            len_diff = count_tokens(''.join(paragraphs).replace('\n', '')) - count_tokens(''.join(chunks).replace('\n', ''))
            assert len_diff == 0, f"Information lost: {len_diff}"
            new_data[book] = chunks
        print(f"{book} chunk sizes: {[count_tokens(c) for c in new_data[book]]}")
        pickle.dump(new_data, open(output_path, 'wb'))
    pickle.dump(new_data, open(output_path, 'wb'))
    return new_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--include_empty_lines", action="store_true")
    args = parser.parse_args()

    books = pickle.load(open(args.input_path, 'rb'))
    process_books(books, args.chunk_size, args.output_path)

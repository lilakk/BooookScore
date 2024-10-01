# <img src="misc/title.png" alt="BooookScore" width="300" height="40"> [: A systematic exploration of book-length summarization in the era of LLMs](https://arxiv.org/pdf/2310.00785)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![arxiv](https://img.shields.io/badge/arXiv-2305.14251-b31b1b.svg)](https://arxiv.org/abs/2310.00785)

This repository hosts the official code and data release for our ICLR 2024 paper, [BooookScore: A systematic exploration of book-length summarization in the era of LLMs](https://arxiv.org/abs/2310.00785). There are 4 O's 😁

If you find BooookScore useful, please cite:
```
@inproceedings{
    chang2024booookscore,
    title={BooookScore: A systematic exploration of book-length summarization in the era of {LLM}s},
    author={Yapei Chang and Kyle Lo and Tanya Goyal and Mohit Iyyer},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://arxiv.org/pdf/2310.00785.pdf}
}
```

Also, if you're interested in faithfulness evaluation in book-length summarization, check out our follow-up work: *FABLES: Evaluating faithfulness and content selection in book-length summarization* ([paper](https://arxiv.org/pdf/2404.01261.pdf) | [repo](https://github.com/mungg/FABLES))!

Some TODO's for future updates are at the end of this README. We also welcome open-source contributions 🥰

# 📢 Announcements

- `2024/09/03` Added a Google form link for requesting the BooookScore dataset.
- `2024/04/01` BooookScore is now available as a Python package!
- `2024/02/27` We now have BooookScore v2, a version that batches sentences when obtaining model-generated annotations for summaries. Kudos to [@IlyaGusev](https://github.com/IlyaGusev) for implementing this!
- `2023/10/10` Initial data release: all summaries, GPT-4 annotations, and human annotations.

# 💿 Requesting a copy of the dataset

If you are interested in getting a copy of the BooookScore dataset, please fill out [this form](https://forms.gle/FfrJpHz54BdDEaj59). Note that we can only release the dataset to academic labs.

# ⬇️ Install BooookScore

```
pip install booookscore
```

# 🤩 Using BooookScore

## Getting chunked data

Before running the chunking script, you need to have a **pickle** file containing a dictionary, where keys are book names and values are full texts of the books. Refer to `data/example_all_books.pkl` for an example. Once you have this file ready, run the following command to chunk the data:

```
python -m booookscore.chunk --chunk_size {chunk_size} 
    --input_path {input_path} --output_path {output_path}
```

- `--chunk_size`: your desired chunk size (each chunk will not exceed this limit)
- `--input_path`: should be set to the path storing the pickle file described above
- `--output_path`: where to save the chunked data (pickle file)
- `--include_empty_lines` (optional): if specified, it does not remove the empty lines that may exist in the input texts

Example usage:

```
python -m booookscore.chunk --chunk_size 2048 
    --input_path all_books.pkl --output_path all_books_chunked_2048.pkl
```

## Obtain summaries

```
python -m booookscore.summ --book_path {book_path} --summ_path {summ_path} 
    --model {model} --api {api} --api_key {api_key} --method {method} --chunk_size {chunk_size} 
    --max_context_len {max_context_len} --max_summary_len {max_summary_len}
```

- `--book_path`: the path to the chunked data (pickle file)
- `--summ_path`: the path to save the generated summaries
- `--model`: name of the model to use, must be supported by the API you're using
- `--api`: which API to use, currently supports `openai`, `anthropic`, `together`
- `--api_key`: the path to the txt file storing your API key
- `--method`: the summarization method to use, "inc" for incremental updating, "hier" for hierarchical merging
- `--chunk_size`: the desired size of each chunk of text, must be consistent with your data in `book_path`
- `max_context_len`: the maximum context window of the model
- `max_summary_len`: the maximum number of tokens a summary can have

Example usage (GPT 4):

```
python -m booookscore.summ --book_path all_books_chunked_4096.pkl 
    --summ_path summaries.json --model gpt-4 --api openai --api_key api_key.txt 
    --method hier --chunk_size 4096 --max_context_len 8192
```

Example usage (Claude 3 Opus):

```
python -m booookscore.summ --book_path all_books_chunked_150000.pkl 
    --summ_path summaries.json --model claude-3-opus-20240229 
    --api anthropic --api_key api_key.txt --method hier 
    --chunk_size 150000 --max_context_len 200000
```

Example usage (Mixtral 8x7B):

```
python -m booookscore.summ --book_path all_books_chunked_30000.pkl 
    --summ_path summaries.json --model mistralai/Mixtral-8x7B-Instruct-v0.1
    --api together --api_key api_key.txt --method hier 
    --chunk_size 30000 --max_context_len 32000
```

### Checkpointing

**Incremental updating** saves progress every 10 chunks. **Hierarchical merging** saves progress every book. Improved checkpointing (and data structure as well) for hierarchical merging will be implemented in future versions!

## Post-processing summaries

After generating summaries with incremental updating or hierarchical merging, we create a json file with a dictionary that maps book names to their final summaries. If the input file is `summaries.json`, then the extracted final summaries will be saved to `summaries_cleaned.json`.

```
python -m booookscore.postprocess --input_path {input_path} 
    --model {model} --api {api} --api_key {api_key}
```

- `--input_path`: the path to the chunked data (pickle file)
- `--model` (optional): which model to use if you want a LLM to remove summary artifacts
- `--api` (optional): which API to use, currently supports `openai`, `anthropic`, `together`
- `--api_key` (optional): the path to the txt file storing your OpenAI API key
- `--remove_artifacts` (optional): if specified, it will ask a language model remove artifacts from merging (must also specify `model` and `api_key` in this case)

Example usage (without artifact removal):

```
python -m booookscore.postprocess --input_path summaries.json
```

Example usage (with artifact removal):

```
python -m booookscore.postprocess --input_path summaries.json --model gpt-4 
    --api openai --api_key api_key.txt --remove_artifacts
```

## Compute BooookScore

```
python -m booookscore.score --summ_path {summ_path} --annot_path {annot_path} 
    --model {model} --api {api} --api_key {api_key}
```

The input summaries must be stored in a json file that maps from book names to final book summaries.

- `--summ_path`: the path to all summaries (must specify if there are no annotations yet)
- `--annot_path`: the path to model-generated annotations
- `--model`: which model to use
- `--api`: which API to use, currently supports `openai`, `anthropic`, `together`
- `--api_key`: the path to the txt file storing your API key
- `--v2` (optional): if specified, it will generate annotations using v2 code and prompt, which uses sentence batching instead of evaluating sentence by sentence (contributed by [@IlyaGusev](https://github.com/IlyaGusev)!)
- `--batch_size` (optional): batch size to use if using v2

Example usage (original BooookScore):

```
python -m booookscore.score --summ_path summaries/chatgpt-2048-hier-cleaned.json 
    --annot_path annotations.json --model gpt-4 
    --api openai --api_key api_key.txt
```

Example usage (v2 BooookScore with sentence batching):

```
python -m booookscore.score --summ_path summaries/chatgpt-2048-hier-cleaned.json 
    --annot_path annotations.json --model gpt-4 --api openai 
    --api_key api_key.txt --v2 --batch_size 10
```

# ✅ TODO's for future versions

- Rework the data structure used for hierarchical summaries, it would be best to maintain a mapping between summaries that are one level apart.
- Improve checkpoint for hierarchical merging, currently it only saves outputs when it gets through the whole book.

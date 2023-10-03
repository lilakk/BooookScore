This repo hosts the code for [BooookScore: A systematic exploration of book-length summarization in the era of LLMs](https://arxiv.org/abs/2310.00785). More updates coming soon.

# Configure environment

1. Create virtual environment with `python3 -m virtualenv myenv`.
2. Activate environment with `. ./myenv/bin/activate`.
3. Install packages with `pip3 install -r requirements.txt`.

# Set up API function call

In `scripts/utils.py`, implement the `obtain_response` function so that given a prompt, it makes an API call and returns the output from the LLM. The returned result should be a string stripped of whitespace.

# Pre-process data

Before running the data pre-processing script, you need to have a `data/all_books.pkl` file with a dictionary, where keys are book names and values are full texts of the books.

`python3 scripts/chunk_data.py --chunk_size CHUNK_SIZE`

# Obtain summaries

## Incremental summaries

`python3 scripts/get_inc_summaries.py --model MODEL --max_context_len MAX_CONTEXT_LEN --chunk_size CHUNK_SIZE`

Summaries will be saved to `incremental_summary/incremental_summaries.json`. If the running script is interrupted and you want to pick up where you left off, simply run the command again.

## Hierarchical summaries

`python3 scripts/get_hier_summaries.py --model MODEL --max_context_len MAX_CONTEXT_LEN --chunk_size CHUNK_SIZE`

Summaries will be saved to `summaries/MODEL-CHUNK_SIZE-SUMMARY_STRATEGY`. If the running script is interrupted and you want to pick up where you left off, you'll need modify the output json file before you re-run. The json file is structured as follows:

- nested dictionary
    - (str) book name:
        - '0': (list) level-0 summaries
        - '1': (list) level-1 summaries
        ...
        - 'n': (list) level-n summaries
        - 'final_summary': (str) final summary, same as the summary in the level-n summaries list
    - ...
    - ...

If the last book in the current dictionary doesn't have a 'final_summary' key and the only summary list present is for level 0, simply run the script again to continue where you left off.

If the last book in the current dictionary doesn't have a 'final_summary' key and there are summary lists present for levels higher than 0, you'll need to remove the summary list with the highest key (level), since it's very likely that the model hasn't fully gotten through that level. We will restart from that level. After removing the list, you can run the script again.

# Compute BooookSore

`python3 scripts/get_booookscore.py --model MODEL --chunk_size CHUNK_SIZE --summary_strategy SUMMARY_STRATEGY`

- SUMMARY_STRATEGY should be either "hier" or "inc".

# Note

- GPT-4 annotations will be uploaded soon after some reformatting.
- Raw incremental summary json files contain full text from the books, they will be uploaded after some reformatting.

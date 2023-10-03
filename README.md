This repo hosts the code for [BooookScore: A systematic exploration of book-length summarization in the era of LLMs](https://arxiv.org/abs/2310.00785). More updates coming soon.

# Configure environment

1. Create virtual environment with `python3 -m virtualenv myenv`.
2. Activate environment with `. ./myenv/bin/activate`.
3. Install packages with `pip3 install -r requirements.txt`.

# Add your API key

In `scripts/utils.py`, add your OpenAI API key at the top. By default the model used is GPT-4, you can change it in the `get_response` function. If you want to use another API, re-implement `get_response` and `obtain_response` accordingly.

# Pre-process data

Before running the data pre-processing script, you need to have a `data/all_books.pkl` file with a dictionary, where keys are book names and values are full texts of the books. Refer to `data/example_all_books.pkl` for an example. Once you have this file ready, run the following command to chunk the data:

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

# Compute BooookScore

`python3 scripts/get_booookscore.py --model MODEL --chunk_size CHUNK_SIZE --summary_strategy SUMMARY_STRATEGY`

- SUMMARY_STRATEGY should be either "hier" or "inc".

# Note

- GPT-4 annotations will be uploaded soon after some reformatting.
- Files with intermediate summaries for incremental updating will be uploaded after some reformatting.

# Cite

`@misc{chang2023booookscore,
      title={BooookScore: A systematic exploration of book-length summarization in the era of LLMs}, 
      author={Yapei Chang and Kyle Lo and Tanya Goyal and Mohit Iyyer},
      year={2023},
      eprint={2310.00785},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}`

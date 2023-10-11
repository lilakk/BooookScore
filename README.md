This repo hosts the code for [BooookScore: A systematic exploration of book-length summarization in the era of LLMs](https://arxiv.org/abs/2310.00785). More updates coming soon.

# Updates

- 2023/10/10 Upload files containing intermediate incremental summaries; upload GPT-4 and human annotations.
- 2023/10/03 Initial commit.


# Configure environment

1. Create virtual environment with `python3 -m virtualenv myenv`.
2. Activate environment with `. ./myenv/bin/activate`.
3. Install packages with `pip3 install -r requirements.txt`.

# Add your API key

In `scripts/utils.py`, add your OpenAI API key at the top. By default the model used is GPT-4, you can change it in the `get_response` function. If you want to use another API, re-implement `get_response` and `obtain_response` accordingly.

# Pre-process data

Before running the data pre-processing script, you need to have a pickle file with a dictionary, where keys are book names and values are full texts of the books. Refer to `data/example_all_books.pkl` for an example. Once you have this file ready, run the following command to chunk the data:

`python3 scripts/chunk_data.py --input_path INPUT_PATH --chunk_size CHUNK_SIZE`

- `input_path` should be set to the pickle file described above.

# Obtain summaries

## Incremental summaries

`python3 scripts/get_inc_summaries.py --input_path INPUT_PATH --save_path SAVE_PATH --max_context_len MAX_CONTEXT_LEN --chunk_size CHUNK_SIZE`

- `input_path` should be set to a pickle file containing chunked data where chunk size = `chunk_size`.

If the running script is interrupted and you want to pick up where you left off, simply run the command again.

## Hierarchical summaries

`python3 scripts/get_hier_summaries.py --input_path INPUT_PATH --save_path SAVE_PATH --max_context_len MAX_CONTEXT_LEN --chunk_size CHUNK_SIZE`

- `input_path` should be set to a pickle file containing chunked data where chunk size = `chunk_size`.

If the running script is interrupted and you want to pick up where you left off, you'll need modify the output json file before you re-run. The json file is structured as follows:

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

`python3 scripts/get_booookscore.py --input_path INPUT_PATH`

- `input_path` should be set to a json file with book names as keys and final summaries as values (e.g., any file ending with `-cleaned.json` in the `summaries` folder).
- GPT-4 annotations will be saved to a file with the same name as the input file in the `gpt4_annotations` directory.

# Note

- GPT-4 and human annotations for existing summaries will be uploaded soon after some reformatting.

# Cite

```
@misc{chang2023booookscore,
      title={BooookScore: A systematic exploration of book-length summarization in the era of LLMs}, 
      author={Yapei Chang and Kyle Lo and Tanya Goyal and Mohit Iyyer},
      year={2023},
      eprint={2310.00785},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

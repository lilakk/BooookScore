import os
import traceback
import time

import openai
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def count_tokens(text):
    return len(tokenizer.encode(text))


def get_response(
    prompt: str,
    model_name: str,
    max_tokens: int,
    temperature: int,
):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response


def obtain_response(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    model_name: str = "gpt-4-turbo-preview",
    echo: bool = False
):
    response = None
    num_attemps = 0
    while response is None:
        try:
            response = get_response(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                model_name=model_name
            )
        except Exception as e:
            print(traceback.format_exc())
            num_attemps += 1
            print(f"Attempt {num_attemps} failed, trying again after 5 seconds...")
            time.sleep(5)

    return response["choices"][0]["message"]["content"].strip()

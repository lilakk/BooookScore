import os
import traceback
import time
import json
import pickle
import tiktoken
from openai import OpenAI

encoding = tiktoken.get_encoding('cl100k_base')


def count_tokens(text):
    return len(encoding.encode(text))


class OpenAIClient():
    def __init__(self, key_path, model):
        with open(key_path, "r") as f:
            key = f.read().strip()
        self.client = OpenAI(api_key=key)
        self.model = model

    def obtain_response(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        model_name: str = "gpt-4"
    ):
        response = None
        num_attemps = 0
        while response is None:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception as e:
                print(e)
                num_attemps += 1
                print(f"Attempt {num_attemps} failed, trying again after 5 seconds...")
                time.sleep(5)
        return response.choices[0].message.content.strip()

import os
import traceback
import time
import json
import pickle
import tiktoken
from openai import OpenAI
from anthropic import Anthropic

encoding = tiktoken.get_encoding('cl100k_base')


def count_tokens(text):
    return len(encoding.encode(text))


class APIClient():
    def __init__(self, api, key_path, model):
        assert key_path.endswith(".txt"), "api key path must be a txt file."
        self.api = api
        self.model = model
        if api == "openai":
            self.client = OpenAIClient(key_path, model)
        elif api == "anthropic":
            self.client = AnthropicClient(key_path, model)
        elif api == "together":
            self.client = TogetherClient(key_path, model)
        else:
            raise ValueError(f"API {api} not supported, custom implementation required.")

    def obtain_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ):
        return self.client.obtain_response(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )


class BaseClient:
    def __init__(self, key_path, model):
        with open(key_path, "r") as f:
            self.key = f.read().strip()
        self.model = model

    def obtain_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ):
        response = None
        num_attempts = 0
        while response is None:
            try:
                response = self.send_request(prompt, max_tokens, temperature)
            except Exception as e:
                print(e)
                num_attempts += 1
                print(f"Attempt {num_attempts} failed, trying again after 5 seconds...")
                time.sleep(5)
        return response

    def send_request(self, prompt, max_tokens, temperature):
        raise NotImplementedError("send_request method must be implemented by subclasses.")


class OpenAIClient(BaseClient):
    def __init__(self, key_path, model):
        super().__init__(key_path, model)
        self.client = OpenAI(api_key=self.key)

    def send_request(self, prompt, max_tokens, temperature):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


class AnthropicClient(BaseClient):
    def __init__(self, key_path, model):
        super().__init__(key_path, model)
        self.client = Anthropic(api_key=self.key)

    def send_request(self, prompt, max_tokens, temperature):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text


class TogetherClient(BaseClient):
    def __init__(self, key_path, model):
        super().__init__(key_path, model)
        self.client = OpenAI(api_key=self.key, base_url="https://api.together.xyz/v1")

    def send_request(self, prompt, max_tokens, temperature):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

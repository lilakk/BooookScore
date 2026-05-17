import os
import subprocess
import traceback
import time
import json
import pickle
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types

from xai_sdk import Client as xAIClient
from xai_sdk.chat import user, system

# Load a .env file (if present) so every client can read its key from the
# environment.
load_dotenv()

encoding = tiktoken.get_encoding('cl100k_base')


def count_tokens(text):
    return len(encoding.encode(text))


class APIClient():
    def __init__(self, api, model=None):
        self.api = api
        self.model = model
        if api == "openai":
            self.client = OpenAIClient(model)
        elif api == "anthropic":
            self.client = AnthropicClient(model)
        elif api == "together":
            self.client = TogetherClient(model)
        elif api == "gemini":
            self.client = GeminiClient(model)
        elif api == "xai":
            self.client = XAIClient(model)
        elif api == "claude-code":
            self.client = ClaudeCodeClient(model)
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
    def __init__(self, model=None):
        self.model = model

    def obtain_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0,
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
    def __init__(self, model=None):
        super().__init__(model)
        self.client = OpenAI()

    def send_request(self, prompt, max_tokens, temperature):
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        return response.output_text


class AnthropicClient(BaseClient):
    def __init__(self, model=None):
        super().__init__(model)
        # Anthropic() reads ANTHROPIC_API_KEY from the environment.
        self.client = Anthropic()

    def send_request(self, prompt, max_tokens, temperature):
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(e)
            return None


class TogetherClient(BaseClient):
    def __init__(self, model=None):
        super().__init__(model)
        self.client = OpenAI(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )

    def send_request(self, prompt, max_tokens, temperature):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


class GeminiClient(BaseClient):
    def __init__(self, model=None):
        super().__init__(model)
        self.client = genai.Client()

    def send_request(self, prompt, max_tokens, temperature):
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens*10)
        )
        return response.text

    def obtain_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0,
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

class XAIClient(BaseClient):
    def __init__(self, model=None):
        super().__init__(model)
        self.client = xAIClient(api_key=os.getenv("XAI_API_KEY"))

    def send_request(self, prompt, max_tokens, temperature):
        chat = self.client.chat.create(model=self.model, max_tokens=max_tokens, temperature=temperature)
        chat.append(user(prompt))
        response = chat.sample()
        return response.content

    def obtain_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0,
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


class ClaudeCodeClient:
    """Client that routes requests through the Claude Code CLI (`claude -p`)."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def obtain_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0,
    ) -> str:
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

    def send_request(self, prompt: str, max_tokens: int, temperature: float) -> str:
        cmd = ["claude", "-p", prompt, "--model", self.model, "--output-format", "text"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()

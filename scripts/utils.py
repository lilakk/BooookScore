from transformers import GPT2Tokenizer
import time
import openai

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

openai.api_key = "YOUR_API_KEY"


def count_tokens(text):
    return len(tokenizer.encode(text))


def get_response(prompt: str, max_tokens = 0, temperature = 0.1, top_p = 1, n = 1, logprobs = 1, stop = None, echo = False):
    response = openai.ChatCompletion.create(model="gpt-4",
                                        messages=[
                                            {"role": "user", "content": prompt}
                                        ],
                                        temperature=temperature,
                                        max_tokens=max_tokens)
    return response


def obtain_response(prompt: str, max_tokens = 0, temperature = 0.5, echo = False):
    response = None
    num_attemps = 0
    while response is None:
        try:
            response = get_response(prompt, max_tokens=max_tokens, temperature=temperature, echo=echo)
        except Exception as e:
            print(e)
            num_attemps += 1
            print(f"Attempt {num_attemps} failed, trying again after 5 seconds...")
            time.sleep(5)
    return response['choices'][0]['message']['content'].strip()

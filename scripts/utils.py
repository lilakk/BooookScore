from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def count_tokens(text):
    return len(tokenizer.encode(text))


def obtain_response(prompt):
    # implement this function so that it returns a response from the model
    # the returned value should be a string stripped of whitespace
    raise NotImplementedError

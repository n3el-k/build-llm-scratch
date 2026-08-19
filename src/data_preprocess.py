import torch
from torch.utils.data import DataLoader, Dataset
import re
import tiktoken

# Load raw data
with open('../the-verdict.txt') as file:
    raw = file.read()

word_array = re.split(r'([_,.!-();:?"\']|--|\s)', raw)
word_array = [item.lower() for item in word_array if item.strip()]
sorted_words = sorted(word_array)
sorted_words.extend(["<unk>", "<endoftext>"]) # Add special tokens

vocab = {token:idx for idx,token in enumerate(sorted_words)}
vocab_size = len(vocab)

# Create tokenizer
bpe_tokenizer = tiktoken.get_encoding('gpt2')

class LLMDataset(Dataset):

    def __init__(self):
        pass
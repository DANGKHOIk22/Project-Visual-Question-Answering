import os
import sys
from typing import Iterator, List
from PIL import Image
import numpy as np
import torch
import spacy
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator, Vocab


# Project structure handling
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '....'))
sys.path.append(PROJECT_ROOT)
from constants import IMAGE_DIRNAME

# Load Spacy tokenizer
eng = spacy.load("en_core_web_sm")

def get_tokens(train_iter: Iterator[dict]) -> Iterator[List[str]]:
    for sample in train_iter:
        question = sample['question']
        yield [token.text for token in eng(question)]  # Direct use of Spacy NLP object

def build_vocab(train_data: Iterator[dict], min_freq: int = 2) -> Vocab:
    vocab = build_vocab_from_iterator(
        get_tokens(train_data),
        specials=['<unk>', '<pad>'],
        min_freq=min_freq,
        special_first=True
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def tokenize(question: str, vocab: Vocab, max_seq_len: int) -> List[int]:
    tokens = [token.text for token in eng(question)]
    sequence = [vocab[token] for token in tokens]

    # Padding or truncation
    sequence = sequence[:max_seq_len] + [vocab['<pad>']] * max(0, max_seq_len - len(sequence))

    return sequence

class VQADataset(Dataset):
    def __init__(
        self,
        data,
        vocab,
        max_seq_len=20,
        transform=None,
        img_dir=IMAGE_DIRNAME
    ):
        self.data = data
        self.max_seq_len = max_seq_len
        self.vocab = vocab
        self.transform = transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image_path = sample['image_path']
        question = sample['question']
        answer = sample['answer']

        image = Image.open(os.path.join(self.img_dir, image_path)).convert('RGB')
        if self.transform :
            image = self.transform(image)

        question = tokenize(question,self.vocab, self.max_seq_len)
        question = torch.tensor(question, dtype=torch.long)

        label = 1 if answer.lower() == 'yes' else 0
        label = torch.tensor(label, dtype=torch.long)

        return image, question, label
import os
import sys
from typing import Iterator, List
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

# Project structure handling
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '....'))
sys.path.append(PROJECT_ROOT)
from constants import IMAGE_DIRNAME


class VQADataset(Dataset):
    def __init__(
        self,
        data,
        img_feature_extractor,
        text_tokenizer,
        device,
        transforms=None,
        img_dir=IMAGE_DIRNAME
    ):
        self.data = data
        self.img_dir = img_dir
        self.img_feature_extractor = img_feature_extractor
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.data[index]['image_path'])
        img = Image.open(img_path).convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        if self.img_feature_extractor:
            img = self.img_feature_extractor(images=img, return_tensors="pt")
            img = {k: v.to(self.device).squeeze(0) for k, v in img.items()}

        question = self.data[index]['question']
        if self.text_tokenizer:
            question = self.text_tokenizer(
                question,
                padding="max_length",
                max_length=20,
                truncation=True,
                return_tensors="pt"
            )
            question = {k: v.to(self.device).squeeze(0) for k, v in question.items()}


        answer = self.data[index]['answer']
        label = 1 if answer.lower() == 'yes' else 0
        label = torch.tensor(label, dtype=torch.long)

        sample = {
            'image': img,
            'question': question,
            'label': label
        }

        return sample
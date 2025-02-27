import os
import sys
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from torchvision import transforms
from VQA_dataset import VQADataset

# Project structure handling
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '....'))
sys.path.append(PROJECT_ROOT)
from constants import TRAIN_DIRNAME, TEST_DIRNAME, VAL_DIRNAME

def get_data(path: str) :
  
    data = []
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.strip().split('\t')
                if len(tmp) < 2:
                    print(f"Warning: Malformed line (skipped): {line}")
                    continue
                
                qa = tmp[1].split('?')
                if len(qa) >= 2:
                    question = qa[0] + '?'
                    answer = qa[1].strip() if len(qa) == 2 else qa[2].strip()
                    data.append({
                        'image_path': tmp[0].strip()[:-2],  
                        'question': question,
                        'answer': answer
                    })
                else:
                    print(f"Warning: Skipping malformed question-answer pair: {line}")
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
    
    return data

def get_dataloaders(
        train_dir: str = TRAIN_DIRNAME,
        val_dir: str = VAL_DIRNAME,
        test_dir: str = TEST_DIRNAME,
) -> Tuple[DataLoader, DataLoader, DataLoader]:


    train_batch_size = 256
    test_batch_size = 32

    # Load data
    train_data = get_data(train_dir)
    val_data = get_data(val_dir)
    test_data = get_data(test_dir)

    # Define image transformations
    data_transform = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(180),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    }

    # Create datasets
    train_dataset = VQADataset(data=train_data, transform=data_transform['train'])
    val_dataset = VQADataset(data=val_data, transform=data_transform['val'])
    test_dataset = VQADataset(data=test_data, transform=data_transform['val'])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

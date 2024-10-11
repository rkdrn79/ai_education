import torch
import numpy as np

from torch.utils.data import Dataset

from src.utils.tokentype import TokenType

class LLMDataset(Dataset):
    def __init__(self, args, instruction, image):
        self.args = args
        self.instruction = instruction
        self.image = image

    def __len__(self):
        return len(self.instruction)
    
    def __getitem__(self, idx):

        return {
            'input_ids': self.data['input_ids'][idx],
            'attention_mask': self.data['attention_mask'][idx],
            'image': self.data['oricorn'][idx],
            'token_type_ids': self.data['token_type_ids'][idx],
            'target_mask': self.data['target_mask'][idx]
        }
            
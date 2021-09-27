import torch
from torch.utils.data import Dataset

from utils import convert_excerpt_to_features

class CommonlitDataset(Dataset):
    
    def __init__(self, df, tokenizer, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        if not self.is_test:
            excerpt_sample = self.df.excerpt_processed.values[idx]
            excerpt_tokenized = convert_excerpt_to_features(excerpt_sample, self.tokenizer)
            target = self.df.target.values[idx]
            
            return {
                'input_ids': excerpt_tokenized["input_ids"][0],
                'token_type_ids': excerpt_tokenized["token_type_ids"][0],
                'attention_mask': excerpt_tokenized["attention_mask"][0],
                'target': torch.tensor(target, dtype=torch.float)
            }
        
        else:
            excerpt_sample = self.df.excerpt_processed.values[idx]
            excerpt_tokenized = convert_excerpt_to_features(excerpt_sample, self.tokenizer)
            return {
                'input_ids': excerpt_tokenized["input_ids"][0],
                'token_type_ids': excerpt_tokenized["token_type_ids"][0],
                'attention_mask': excerpt_tokenized["attention_mask"][0],
            }
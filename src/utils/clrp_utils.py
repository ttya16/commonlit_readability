import os
import time
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AdamW, AutoTokenizer, AutoModel, AutoConfig
from transformers import get_scheduler

from tqdm.notebook import tqdm

from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler

import gc
gc.enable()

import warnings
warnings.filterwarnings('ignore')


# SEED
def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True

# DATA PREPROCESS
def process_text(text):
    # remove newlines
    text = text.replace('\n', '')
    text = text.replace('\\', '')

    return text

def preprocess_train(df_train):
    drop_ids = ['da2dbbc70', '0684bb254']
    df_train = df_train.loc[~df_train.id.isin(drop_ids)]\
        [['id', 'excerpt', 'target', 'standard_error']].reset_index(drop=True)
    
    df_train["excerpt_processed"] = df_train.excerpt.apply(process_text)
    
    num_bins = int(np.floor(1 + np.log2(len(df_train))))
    df_train.loc[:, 'bins'] = pd.cut(df_train['target'], bins=num_bins, labels=False)
    bins = df_train.bins.to_numpy()

    return df_train, bins

# DATA CONVERT
def convert_excerpt_to_features(
    excerpt,
    tokenizer,
    is_test=False
):
    tokenized_excerpt = tokenizer.encode_plus(
        excerpt,
        truncation=True,
        max_length=256,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='pt',
        padding="max_length"
    )
    
    return tokenized_excerpt

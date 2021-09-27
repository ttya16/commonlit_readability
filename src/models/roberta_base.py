
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import AdamW, AutoTokenizer, AutoModel, AutoConfig
from transformers import get_scheduler

from tqdm.notebook import tqdm

import sys
sys.path.append("/clrp/src/models/")

import gc
gc.enable()

import warnings
warnings.filterwarnings('ignore')

from config.configs import Config as c
from heads import AttentionHead, WeightedLayerPooling, AttentionPooling

class CommonlitModel(nn.Module):
    
    def __init__(
        self, 
        checkpoint,
        dropout_p=0.3, 
        output_hidden_state=False
    ):
        super(CommonlitModel, self).__init__()
        
        self.model_config = AutoConfig.from_pretrained(checkpoint)
        self.model_config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})   
        self.encoder_model = AutoModel.from_pretrained(checkpoint, config=self.model_config)
        
        self.head = AttentionHead(self.model_config.hidden_size, 512)
#         self.pooler = WeightedLayerPooling(self.model_config.num_hidden_layers, layer_start=9, layer_weights=None)
#         self.pooler = AttentionPooling(self.model_config.num_hidden_layers, self.model_config.hidden_size, 128)

        self.dropout = nn.Dropout(dropout_p)
#         self.fc = nn.Linear(self.model_config.hidden_size, 64)
        self.out = nn.Linear(self.model_config.hidden_size, 1)
        
    def forward(
        self,
        batch
    ):
        
        input_ids = batch["input_ids"].to(c.device)
        attention_masks = batch["attention_mask"].to(c.device)


        enc_output = self.encoder_model(
            input_ids=input_ids,
            attention_mask=attention_masks
        )
            
        head_output = self.head(enc_output.last_hidden_state)
#         all_hidden_states = torch.stack(enc_output.hidden_states)
#         cls_embeddings = all_hidden_states[12, :, 0]
#         concat_pool = torch.cat(
#             (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), -1
#         )
#         concat_pool = concat_pool[:, 0]
#         pooler_output = self.pooler(all_hidden_states)
#         pooler_output = pooler_output[:, 0]
        
        output = F.relu(head_output)
#         output = self.dropout(output)
        output = self.out(output)
                
        return output
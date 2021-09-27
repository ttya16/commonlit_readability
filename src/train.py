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

from dataset.commonlit_dataset import CommonlitDataset
from models import CommonlitModel as CommonlitRobertaBaseModel

from config.configs import Config as cfg
from utils import set_random_seed, preprocess_train, convert_excerpt_to_features

# TODO: Refactoring
def eval_mse(model, val_dataloader):
    """
    Evaluate mse on validation data
    """
    
    model.eval()
    mse_sum = 0
    
    with torch.no_grad():
        for batch_num, batch in enumerate(val_dataloader):
            
            targets = batch["target"].to(DEVICE)
            
            pred = model(batch)
            
            mse_sum += nn.MSELoss(reduction='sum')(pred.flatten(), targets).item()
            
    return mse_sum / len(val_dataloader.dataset)

def train_loop(
    cfg,
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler=None,
    num_epochs=5,
    best_model_name="best_model.pth"
):

    
    best_val_rmse = None
    best_epoch = 0
    step = 0
    last_eval_step = 0
    eval_period = 16
    
    model.to(cfg.device)
    start = time.perf_counter()
    
    for epoch in range(num_epochs):
        val_rmse = None

        
        for batch_num, batch in enumerate(train_dataloader):
            targets = batch["target"].to(cfg.device)
            
            optimizer.zero_grad()
            
            model.train()
            
            pred = model(batch)
            
            mse = nn.MSELoss(reduction="mean")(pred.flatten(), targets)
            
            mse.backward()
            
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            if step >= last_eval_step + eval_period:
                elapsed_seconds = time.perf_counter() - start
                num_steps = step - last_eval_step
                print(f"\n{num_steps} steps took {elapsed_seconds:0.3} seconds")
                last_eval_step = step
                
                val_rmse = math.sqrt(eval_mse(model, val_dataloader))
                
                print(f"Epoch: {epoch} batch_num:{batch_num}", f"val_rmse: {val_rmse:0.4}")
                
                if not best_val_rmse or val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(cfg.output_path, f"{best_model_name}"))
                    print(f"New best_val_rmse: {best_val_rmse:0.4}")
                    
                else:
                    print(f"Still best_val_rmse: {best_val_rmse:0.4}", f"from epoch {best_epoch}")
                    
                start = time.perf_counter()
                
            step += 1
            
    return best_val_rmse

def create_optimizer(model):
    named_parameters = list(model.named_parameters())    
    
    roberta_parameters = named_parameters[:197]    
    attention_parameters = named_parameters[199:203]
    regressor_parameters = named_parameters[203:]
        
    attention_group = [params for (name, params) in attention_parameters]
    regressor_group = [params for (name, params) in regressor_parameters]

    parameters = []
    parameters.append({"params": attention_group})
    parameters.append({"params": regressor_group})

    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01

        lr = 2e-5

        if layer_num >= 69:        
            lr = 5e-5

        if layer_num >= 133:
            lr = 1e-4

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})

    return AdamW(parameters)

def create_wlp_optimizer(model):
    named_parameters = list(model.named_parameters())    
    
    roberta_parameters = named_parameters[:197]    
    attention_parameters = named_parameters[199]
    regressor_parameters = named_parameters[200:]
        
    attention_group = attention_parameters[1]
    regressor_group = [params for (name, params) in regressor_parameters]

    parameters = []
    parameters.append({"params": attention_group})
    parameters.append({"params": regressor_group})

    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01

        lr = 2e-5

        if layer_num >= 69:        
            lr = 5e-5

        if layer_num >= 133:
            lr = 1e-4

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})

    return AdamW(parameters)

if __name__ == "__main__":
    gc.collect()

    DEVICE = cfg.device
    BASE_DATA_PATH = cfg.base_data_path
    MODEL_PATH = cfg.output_path + "/clrp_roberta_base"
    TOKENIZER_PATH = cfg.output_path + "/clrp_roberta_base"

    NUM_FOLDS = cfg.num_folds
    NUM_EPOCHS = cfg.num_epochs
    BATCH_SIZE = cfg.batch_size

    SEED = cfg.seed


    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    df_train = pd.read_csv(BASE_DATA_PATH / "train.csv")
    df_train, bins = preprocess_train(df_train)


    # Stratified k-fold train
    kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    list_val_rmse = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(df_train, bins)):
        print(f"\nFold {fold + 1}/{NUM_FOLDS}")
        best_model_name = f"best_model_fold{fold+1}.pth"
        
        
        train_dataset = CommonlitDataset(df_train.loc[train_indices], tokenizer, is_test=False)
        val_dataset = CommonlitDataset(df_train.loc[val_indices], tokenizer, is_test=False)
        
        train_dataloader =  DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True)


        val_dataloader =  DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True)
        
        set_random_seed(SEED + fold)
        
        model = CommonlitRobertaBaseModel(MODEL_PATH)

        optimizer = create_optimizer(model)
    #     optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=50,
            num_training_steps=NUM_EPOCHS*len(train_dataloader)
        )

        val_rmse = train_loop(
            cfg,
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            scheduler=lr_scheduler,
            num_epochs=NUM_EPOCHS,
            best_model_name=best_model_name
        )
        
        list_val_rmse.append(val_rmse)
        
        del model
        gc.collect()
        
        print("\nPerformance estimates:")
        print(list_val_rmse)
        print("Mean:", np.array(list_val_rmse).mean())



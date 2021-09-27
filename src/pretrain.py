import pandas as pd

import os
import warnings
warnings.filterwarnings('ignore')

from transformers import (AutoModel,AutoModelForMaskedLM, 
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)


from config.configs import Config as c
from utils import process_text

if __name__ == "__main__":
    BASE_DATA_PATH = c.base_data_path
    OUTPUT_PATH = c.output_path + "/clrp_roberta_base"
    CHKPT_PATH = c.output_path + '/clrp_roberta_base_chk'

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    if not os.path.exists(CHKPT_PATH):
        os.makedirs(CHKPT_PATH)

    train_data = pd.read_csv(os.path.join(BASE_DATA_PATH, 'train.csv'))
    test_data = pd.read_csv(os.path.join(BASE_DATA_PATH, 'test.csv'))

    clrp_data = pd.concat([train_data, test_data])

    # Prepare data
    text_data = clrp_data["excerpt"].apply(process_text)
    text = '\n'.join(text_data.tolist())
    # Temporarily output as .txt file
    TEXT_OUTPUT_PATH = os.path.join(c.output_path, 'text.txt')
    with open(TEXT_OUTPUT_PATH, 'w') as f:
        f.write(text)

    # Load model and tokenizer
    model_name = 'roberta-base'
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(OUTPUT_PATH)

    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=TEXT_OUTPUT_PATH, #mention train text file here
        block_size=256)

    valid_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=TEXT_OUTPUT_PATH, #mention valid text file here
        block_size=256)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=os.path.join(CHKPT_PATH), #select model path for checkpoint
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy= 'steps',
        save_total_limit=2,
        eval_steps=200,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        load_best_model_at_end =True,
        prediction_loss_only=True,
        report_to = "none")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset)

    trainer.train()
    trainer.save_model(OUTPUT_PATH)
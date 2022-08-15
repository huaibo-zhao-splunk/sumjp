from pathlib import Path
import re
import math
import time
import copy
from tqdm import tqdm
import pandas as pd
import tarfile
import neologdn
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import settings

data_dir_path = Path("/home/ubuntu/20220812/data")

body_data = pd.read_csv(data_dir_path.joinpath('body_data.csv'))
summary_data = pd.read_csv(data_dir_path.joinpath('summary_data.csv'))

def join_text(x, add_char='。'):
    return add_char.join(x)

def preprocess_text(text):
    text = re.sub(r'[\r\t\n\u3000]', '', text)
    text = neologdn.normalize(text)
    text = text.lower()
    text = text.strip()
    return text

summary_data = summary_data.query('text.notnull()', engine='python').groupby(
    'article_id'
).agg({'text': join_text})

body_data = body_data.query('text.notnull()', engine='python')

data = pd.merge(
    body_data.rename(columns={'text': 'body_text'}),
    summary_data.rename(columns={'text': 'summary_text'}),
    on='article_id', how='inner'
).assign(
    body_text=lambda x: x.body_text.map(lambda y: preprocess_text(y)),
    summary_text=lambda x: x.summary_text.map(lambda y: preprocess_text(y))
)

print("data preparation done.")


def convert_batch_data(train_data, valid_data, tokenizer):

    def generate_batch(data):

        batch_src, batch_tgt = [], []
        for src, tgt in data:
            batch_src.append(src)
            batch_tgt.append(tgt)

        batch_src = tokenizer(
            batch_src, max_length=settings.max_length_src, truncation=True, padding="max_length", return_tensors="pt"
        )
        batch_tgt = tokenizer(
            batch_tgt, max_length=settings.max_length_target, truncation=True, padding="max_length", return_tensors="pt"
        )

        return batch_src, batch_tgt

    train_iter = DataLoader(train_data, batch_size=settings.batch_size_train, shuffle=True, collate_fn=generate_batch)
    valid_iter = DataLoader(valid_data, batch_size=settings.batch_size_valid, shuffle=True, collate_fn=generate_batch)

    return train_iter, valid_iter

tokenizer = T5Tokenizer.from_pretrained(settings.MODEL_NAME, is_fast=True)

print("Tokenizer initialized.")

X_train, X_test, y_train, y_test = train_test_split(
    data['body_text'], data['summary_text'], test_size=0.15, random_state=42, shuffle=True
)

train_data = [(src, tgt) for src, tgt in zip(X_train, y_train)]
valid_data = [(src, tgt) for src, tgt in zip(X_test, y_test)]

train_iter, valid_iter = convert_batch_data(train_data, valid_data, tokenizer)

print("data vectorization finished. Training data: " + str(len(train_data)) + ", Test data: " + str(len(valid_data)))

class T5FineTuner(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.model = T5ForConditionalGeneration.from_pretrained(settings.MODEL_NAME)

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None,
        decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

# Training function
def train(model, data, optimizer, PAD_IDX):
    
    model.train()
    
    loop = 1
    losses = 0
    pbar = tqdm(data)
    for src, tgt in pbar:
                
        optimizer.zero_grad()
        
        labels = tgt['input_ids'].to(settings.device)
        labels[labels[:, :] == PAD_IDX] = -100

        outputs = model(
            input_ids=src['input_ids'].to(settings.device),
            attention_mask=src['attention_mask'].to(settings.device),
            decoder_attention_mask=tgt['attention_mask'].to(settings.device),
            labels=labels
        )
        loss = outputs['loss']

        loss.backward()
        optimizer.step()
        losses += loss.item()
        
        pbar.set_postfix(loss=losses / loop)
        loop += 1
        
    return losses / len(data)

# VLoss function
def evaluate(model, data, PAD_IDX):
    
    model.eval()
    losses = 0
    with torch.no_grad():
        for src, tgt in data:

            labels = tgt['input_ids'].to(settings.device)
            labels[labels[:, :] == PAD_IDX] = -100

            outputs = model(
                input_ids=src['input_ids'].to(settings.device),
                attention_mask=src['attention_mask'].to(settings.device),
                decoder_attention_mask=tgt['attention_mask'].to(settings.device),
                labels=labels
            )
            loss = outputs['loss']
            losses += loss.item()
        
    return losses / len(data)


model = T5FineTuner()
model = model.to(settings.device)

print("model initialized.")

optimizer = optim.Adam(model.parameters())

PAD_IDX = tokenizer.pad_token_id
best_loss = float('Inf')
best_model = None
counter = 1

print("Start training.")

for loop in range(1, settings.epochs + 1):

    start_time = time.time()

    loss_train = train(model=model, data=train_iter, optimizer=optimizer, PAD_IDX=PAD_IDX)

    elapsed_time = time.time() - start_time

    loss_valid = evaluate(model=model, data=valid_iter, PAD_IDX=PAD_IDX)

    print('[{}/{}] train loss: {:.4f}, valid loss: {:.4f} [{}{:.0f}s] counter: {} {}'.format(
        loop, settings.epochs, loss_train, loss_valid,
        str(int(math.floor(elapsed_time / 60))) + 'm' if math.floor(elapsed_time / 60) > 0 else '',
        elapsed_time % 60,
        counter,
        '**' if best_loss > loss_valid else ''
    ))

    if best_loss > loss_valid:
        best_loss = loss_valid
        best_model = copy.deepcopy(model)
        counter = 1
    else:
        if counter > settings.patience:
            break

        counter += 1

print("finished training.")

# Saving model
model_dir_path = Path('/home/ubuntu/20220812/models/t5_dialog_jp')
tokenizer.save_pretrained(model_dir_path)
print("tokenizer saved.")
best_model.model.save_pretrained(model_dir_path)
print("model saved. Successfully finished.")
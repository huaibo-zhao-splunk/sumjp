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


X_train, X_test, y_train, y_test = train_test_split(
    data['body_text'], data['summary_text'], test_size=0.15, random_state=42, shuffle=True
)

train_data = [(src, tgt) for src, tgt in zip(X_train, y_train)]
valid_data = [(src, tgt) for src, tgt in zip(X_test, y_test)]

print("data preparation done.")




def generate_text_from_model(text, trained_model, tokenizer, num_return_sequences=1):

    trained_model.eval()
    
    text = preprocess_text(text)
    batch = tokenizer(
        [text], max_length=settings.max_length_src, truncation=True, padding="longest", return_tensors="pt"
    )

    outputs = trained_model.generate(
        input_ids=batch['input_ids'].to(settings.device),
        attention_mask=batch['attention_mask'].to(settings.device),
        max_length=settings.max_length_target,
        repetition_penalty=8.0,  
        # num_beams=10,
        # num_beam_groups=10, 
        num_return_sequences=num_return_sequences,  
    )

    generated_texts = [
        tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ids in outputs
    ]

    return generated_texts


model_dir_path = Path('/home/ubuntu/20220812/models/t5_dialog_jp')

tokenizer = T5Tokenizer.from_pretrained(model_dir_path)
trained_model = T5ForConditionalGeneration.from_pretrained(model_dir_path)

trained_model = trained_model.to(settings.device)


index = 100
body = valid_data[index][0]
summaries = valid_data[index][1]
generated_texts = generate_text_from_model(
    text=body, trained_model=trained_model, tokenizer=tokenizer, num_return_sequences=1
)
print('Summarization')
print('\n'.join(generated_texts[0].split('。')))
print()
print('ground truth')
print('\n'.join(summaries.split('。')))
print()
print('original text')
print(body)
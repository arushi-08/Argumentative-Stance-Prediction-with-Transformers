

import re
from collections import Counter

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

from google.colab import drive
drive.mount('/content/drive')

# Define hyperparameter grid
learning_rates = [1e-5]
batch_sizes = [32]

def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = tweet.lower()
    return tweet


def train_model(model_name, text, labels, dev_text, dev_labels):

    batch_size = 32
    learning_rate = 1e-5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading pretrained model: ', model_name)

    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    model = XLNetForSequenceClassification.from_pretrained(model_name)

    model.to(device)  # Move model to GPU

    encodings = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    train_dataset = torch.utils.data.TensorDataset(
        encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels)
        )    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    encodings = tokenizer(dev_text, return_tensors='pt', padding=True, truncation=True)
    val_dataset = torch.utils.data.TensorDataset(
        encodings['input_ids'], encodings['attention_mask'], torch.tensor(dev_labels)
        )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Compute class weights
    class_counts = Counter(labels)
    class_weights = 1.0 / torch.tensor(list(class_counts.values()), dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_val_loss = float('inf')
    best_model = None
    patience = 3
    counter = 0

    for epoch in range(50):
        # Train on training data
        epoch_loss = 0.0
        model.train()
        all_preds = []
        all_labels = []
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)  # Move data to GPU
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

            loss = loss_fn(outputs.logits, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1} train loss: {epoch_loss / len(train_loader)}')

        # Evaluate on validation data
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)  # Move data to GPU
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                val_loss += loss.item()
        print(f'Epoch {epoch + 1} val loss: {val_loss / len(val_loader)}')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                # Evaluation metrics
                f1 = f1_score(all_labels, all_preds, average='macro')
                print(f'Training data F1 score: {f1}')
                precision = precision_score(all_labels, all_preds, average='macro')
                print(f'Trainnig data Precision score: {precision}')
                recall = recall_score(all_labels, all_preds, average='macro')
                print(f'Training data Recall score: {recall}')
                break

    return best_val_loss, best_model



def main_training(dataset, model_name):

    print('Loading dataset: ', dataset)
    # ibm_df = pd.read_csv(f'/content/drive/MyDrive/{dataset}_train.csv')
    imagearg_df = pd.read_csv(f'/content/drive/MyDrive/ImageArg-Shared-Task/data/{dataset}_train.csv')
    # abortion_df = pd.read_csv(f'/content/drive/MyDrive/ImageArg-Shared-Task/data/abortion_train.csv')
    # ibm_df = ibm_df[['claims.claimCorrectedText', 'claims.stance']].rename(
    # columns={'claims.claimCorrectedText': 'tweet_text', 'claims.stance': 'stance'})
    # df = pd.concat([imagearg_df, abortion_df])

    df = imagearg_df
    text = df['tweet_text'].apply(preprocess_tweet).tolist()
    df['stance_encoded'] = df['stance'].replace({'support':1, 'oppose':0})
    labels = df['stance_encoded'].tolist()
    # labels = df['stance'].replace({'support':1, 'oppose':0, 'PRO':1, 'CON':0}).tolist()
    df_dev = pd.read_csv(f'/content/drive/MyDrive/ImageArg-Shared-Task/data/{dataset}_dev.csv')
    df_dev['tweet_text_cleaned'] = df_dev['tweet_text'].apply(preprocess_tweet)
    dev_text = df_dev['tweet_text_cleaned'].tolist()
    df_dev['stance_encoded'] = df_dev['stance'].replace({'support':1, 'oppose':0})
    dev_labels = df_dev['stance_encoded'].tolist()


    _, best_model = train_model(model_name, text, labels, dev_text, dev_labels)
    best_model.save_pretrained(f'/content/drive/MyDrive/ImageArg-Shared-Task/model_{dataset}_{model_name}_test_exp')

# ------# ------# ------# ------# ------# ------# ------# ------
def main_eval(dataset, model_name):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading model: ', model_name)
    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    model = XLNetForSequenceClassification.from_pretrained(f'/content/drive/MyDrive/ImageArg-Shared-Task/model_{dataset}_{model_name}_test_exp')

    model.to(device)

    print('Loading data: ', dataset)
    # df_dev = pd.read_csv(f'/content/drive/MyDrive/ImageArg-Shared-Task/data/{dataset}_dev.csv')
    # df_dev['stance_encoded'] = df_dev['stance'].replace({'support':1, 'oppose':0})
    df_dev = pd.read_csv(f'/content/drive/MyDrive/ImageArg-Shared-Task/data/{dataset}_dev.csv')
    df_dev['stance_encoded'] = df_dev['stance'].replace({'support':1, 'oppose':0})
    df_dev['tweet_text_cleaned'] = df_dev['tweet_text'].apply(preprocess_tweet)
    test_text = df_dev['tweet_text_cleaned'].tolist()
    test_labels = df_dev['stance_encoded'].tolist()

    test_encodings = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True)
    test_dataset = torch.utils.data.TensorDataset(
        test_encodings['input_ids'].to(device),
        test_encodings['attention_mask'].to(device),
        torch.tensor(test_labels).to(device))
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Make predictions on test data
    all_preds = []
    all_labels = []
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    # Evaluation metrics
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'F1 score: {f1}')
    precision = precision_score(all_labels, all_preds, average='macro')
    print(f'Precision score: {precision}')
    recall = recall_score(all_labels, all_preds, average='macro')
    print(f'Recall score: {recall}')


# xlnet-base
main_training(dataset='gun_control', model_name='xlnet-base-cased')
print('-'*50)
main_eval(dataset='gun_control', model_name='xlnet-base-cased')
print('-'*50)


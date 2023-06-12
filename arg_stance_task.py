import re

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

from google.colab import drive
drive.mount('/content/drive')

# Define hyperparameter grid
learning_rates = [1e-3, 1e-4, 1e-5]
batch_sizes = [8, 16, 32]

def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = tweet.lower()
    return tweet


def train_model(model_name, fold, text, labels, train_index, val_index, learning_rate, batch_size, check_eval):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading pretrained model: ', model_name)

    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # model = BertForSequenceClassification.from_pretrained(model_name)

    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    model = XLNetForSequenceClassification.from_pretrained(model_name)

    model.to(device)  # Move model to GPU
    
    encodings = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    dataset = torch.utils.data.TensorDataset(
        encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels)
        )

    # Split data into training and validation sets
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Compute class weights
    class_counts = torch.bincount(torch.tensor(labels)[train_index])
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_val_loss = float('inf')
    best_model = None
    patience = 3
    counter = 0
    
    for epoch in range(100):
        # Train on training data
        epoch_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)  # Move data to GPU
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Fold {fold} Epoch {epoch + 1} train loss: {epoch_loss / len(train_loader)}')

        if not check_eval:
          if epoch_loss < best_val_loss:
              best_val_loss = epoch_loss
              best_model = model
              counter = 0
          else:
              counter += 1
              if counter >= patience:
                  print(f'Early stopping at epoch {epoch + 1}')
                  break
          continue

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
        print(f'Fold {fold} Epoch {epoch + 1} val loss: {val_loss / len(val_loader)}')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Fold {fold} Early stopping at epoch {epoch + 1}')
                break

    return best_val_loss, best_model


def main_training(dataset, model_name):

    print('Loading dataset: ', dataset)
    df = pd.read_csv(f'/content/drive/MyDrive/ImageArg-Shared-Task/data/{dataset}_train.csv')
    text = df['tweet_text'].apply(preprocess_tweet).tolist()
    labels = df['stance'].replace({'support':1, 'oppose':0}).tolist()

    # Find best hyperparameters
    best_val_loss = float('inf')
    best_learning_rate = None
    best_batch_size = None

    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            print(f'Training with learning rate {learning_rate} and batch size {batch_size}')

            # Perform 5-fold cross-validation
            kfold = KFold(n_splits=5)
            val_losses = []
            for fold, (train_index, val_index) in enumerate(kfold.split(df)):
                print(f'Training fold {fold + 1}')
                val_loss, _ = train_model(model_name, fold + 1, text, labels, train_index, val_index, learning_rate, batch_size, check_eval=True)
                val_losses.append(val_loss)

            print(f'Average validation loss: {np.mean(val_losses)}')

            if np.mean(val_losses) < best_val_loss:
                best_val_loss = np.mean(val_losses)
                best_learning_rate = learning_rate
                best_batch_size = batch_size

    print(f'Best learning rate: {best_learning_rate}')
    print(f'Best batch size: {best_batch_size}')
    

    _, best_model = train_model(model_name, 0, text, labels, list(df.index), [], best_learning_rate, best_batch_size, check_eval=False)
    best_model.save_pretrained(f'/content/drive/MyDrive/ImageArg-Shared-Task/model_{dataset}_{model_name}_unfreeze_layers')


def main_eval(dataset, model_name):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading model: ', model_name)
    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    model = XLNetForSequenceClassification.from_pretrained(f'/content/drive/MyDrive/ImageArg-Shared-Task/model_{dataset}_{model_name}')

    model.to(device)

    print('Loading data: ', dataset)
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

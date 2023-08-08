# !pip install optuna transformers sentencepiece datasets accelerate -U

# final

import re
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = tweet.lower()
    return tweet

dataset = 'gun_control'
df_train = pd.read_csv(f'/content/drive/MyDrive/ImageArg-Shared-Task/data/{dataset}_train.csv')
df_train['tweet_text_cleaned'] = df_train['tweet_text'].apply(preprocess_tweet)
df_train['stance_encoded'] = df_train['stance'].replace({'support':1, 'oppose':0})
df_dev = pd.read_csv(f'/content/drive/MyDrive/ImageArg-Shared-Task/data/{dataset}_dev.csv')
df_dev['tweet_text_cleaned'] = df_dev['tweet_text'].apply(preprocess_tweet)
df_dev['stance_encoded'] = df_dev['stance'].replace({'support':1, 'oppose':0})
dev_labels = df_dev['stance_encoded'].tolist()

model = AutoModelForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")



# ----- 1. Preprocess data -----#
X_train = list(df_train["tweet_text_cleaned"])
y_train = list(df_train["stance_encoded"])
X_val = list(df_dev["tweet_text_cleaned"])
y_val = list(df_dev["stance_encoded"])
# X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.2)
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


args = TrainingArguments(
    output_dir=f"/content/drive/MyDrive/ImageArg-Shared-Task/text_model_trainer/{dataset}/",
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    seed=0,
    load_best_model_at_end=True,
)

# class_counts = torch.bincount(torch.tensor(y_train))
# class_weights = 1.0 / class_counts.float()
# class_weights = class_weights / class_weights.sum()
# class_weights = class_weights.to(device)


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        # extract labels from inputs
        labels = inputs.pop("labels")

        # compute model outputs
        outputs = model(**inputs)

        # compute loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# create trainer with custom compute_loss function
trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # class_weights=class_weights,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
trainer.train()
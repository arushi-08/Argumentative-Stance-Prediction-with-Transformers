import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import re
import pandas as pd
from transformers import FlavaProcessor, FlavaModel
from PIL import Image
import torch
import numpy as np
import random
from torch import nn
import pdb
import torch.nn.functional as F


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


processor = FlavaProcessor.from_pretrained("facebook/flava-full")

dataset = 'gun_control'

def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = tweet.lower()
    return tweet

class CustomTrainDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get item
        item = self.df.iloc[idx]
        text = item['tweet_text']
        # label = item['label']
        if item['stance'] == 'oppose':
          label = 0
        else:
          label = 1
        image = np.array(Image.open(f"/content/drive/MyDrive/ImageArg-Shared-Task/data/images/{dataset}/{item['tweet_id']}.jpg").convert('RGB'))
        # Sometimes PIL returns a 2D tensor (for black-white images),
        # which is not supported by ViLT
        if len(image.shape) == 2:
            image = np.stack([image] * 3, -1)

        # encode text
        encoding = self.processor(text = text, images=image, padding="max_length", truncation=True, return_tensors="pt",)
        # remove batch dimension which the tokenizer automatically adds
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        # add label
        encoding["labels"] = torch.tensor(label)
        
        return encoding


from transformers import BertTokenizer
import pandas as pd

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_df = pd.read_csv(f"/content/drive/MyDrive/ImageArg-Shared-Task/data/{dataset}_train.csv")
train_df['tweet_text'] = train_df['tweet_text'].apply(preprocess_tweet)
train_dataset = CustomTrainDataset(df=train_df, processor=processor)

df = pd.read_csv(f"/content/drive/MyDrive/ImageArg-Shared-Task/data/{dataset}_dev.csv")
df['tweet_text'] = df['tweet_text'].apply(preprocess_tweet)
val_dataset = CustomTrainDataset(df=df, processor=processor)

df = pd.read_csv(f"/content/drive/MyDrive/ImageArg-Shared-Task/data/{dataset}_test_w_labels.csv")
df['tweet_text'] = df['tweet_text'].apply(preprocess_tweet)
test_dataset = CustomTrainDataset(df=df, processor=processor)


def preprocess_function(examples):
    batch_size = 1
    num_images = 1

    tweet_text = preprocess_tweet(examples['tweet_text'])
    if examples['stance'] == 'oppose':
      stance = 0
    else:
      stance = 1

    image = np.array(Image.open(f"/content/drive/MyDrive/ImageArg-Shared-Task/data/images/{dataset}/{examples['tweet_id']}.jpg").convert('RGB'))
    # Sometimes PIL returns a 2D tensor (for black-white images),
  
    if len(image.shape) == 2:
        image = np.stack([image] * 3, -1)

    inputs = processor(text=tweet_text, images=image, padding="max_length", truncation=True, return_tensors="pt",)

    inputs['labels'] = torch.tensor(stance)

    for k in ["input_ids", "token_type_ids", "attention_mask"]:
        inputs[k] = inputs[k].squeeze()

    pixel_value_shape = inputs["pixel_values"].shape
    num_channels = pixel_value_shape[1]
    height = pixel_value_shape[2]
    width = pixel_value_shape[3]
    
    inputs["pixel_values"] = torch.tensor(inputs["pixel_values"].reshape(
        [
            num_channels,
            height,
            width,
        ]
    ))

    return inputs


train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

flava = FlavaModel.from_pretrained("facebook/flava-full")

# Freeze the FLAVA model parameters
for param in flava.parameters():
    param.requires_grad = False

# Define a custom classification head
class ClassificationHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flava = FlavaModel.from_pretrained("facebook/flava-full")
        self.fc = nn.Linear(self.flava.config.hidden_size, num_classes)

    def forward(self, input_ids, pixel_values, attention_mask=None, token_type_ids=None):
        # Pass inputs through Flava model
        flava_output = self.flava(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # Extract the last hidden state of the text encoder
        text_last_hidden_state = flava_output[0]

        # Extract the last hidden state of the [CLS] token
        cls_output = text_last_hidden_state[:, 0]
        
        # Pass [CLS] token output through linear layer
        logits = self.fc(cls_output)

        return logits


num_classes = 2 
model = ClassificationHead(num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
train_df['labels'] = train_df['stance'].replace({'support':1, 'oppose':0})
class_counts = torch.bincount(torch.tensor(train_df['labels']))
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)
criterion = nn.BCEWithLogitsLoss(weight=class_weights)
model.to(device)

best_val_loss = float('inf')
best_model = None
patience = 3
counter = 0


for epoch in range(20):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        labels = F.one_hot(labels, num_classes=num_classes).to(torch.float32)
        labels = labels.to(torch.float32)
        loss = criterion(outputs, labels)
        
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = {k:v.to(device) for k,v in batch.items()}
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            labels = F.one_hot(labels, num_classes=num_classes).to(torch.float32)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
                  
    print(f"Validation loss after epoch {epoch}:", val_loss/len(val_dataloader))


    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

torch.save(best_model.state_dict(), f'/content/drive/MyDrive/ImageArg-Shared-Task/flava_model_{dataset}')


import torch


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
        image = np.array(Image.open(f"/content/drive/MyDrive/ImageArg-Shared-Task/data/images/gun_control/{item['tweet_id']}.jpg").convert('RGB'))
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

import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/ImageArg-Shared-Task/data/gun_control_test_w_labels.csv")
df['tweet_text'] = df['tweet_text'].apply(preprocess_tweet)
test_dataset = CustomTrainDataset(df=df, processor=processor)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
tweet_ids = df['tweet_id'].tolist()
test_predictions = {}


best_model = torch.load(f'/content/drive/MyDrive/ImageArg-Shared-Task/flava_model_{dataset}')
best_model.eval()

for batch, tweet in zip(test_dataloader, tweet_ids):
  inputs = {k:v.to(device) for k,v in batch.items()}
  labels = inputs.pop("labels")
  # Perform inference
  with torch.no_grad():
      logits = best_model(**inputs)
      probabilities = torch.softmax(logits, dim=1).numpy()

  test_predictions[tweet] = np.argmax(probabilities)


test_results_df = pd.DataFrame(test_predictions.items())
test_results_df.columns = ['tweet_id', 'stance']
test_results_df['stance'] = test_results_df['stance'].replace({0:'oppose',1:'support'})
test_results_df.to_csv('Pitt Pixel Persuaders.FlavaModel.TaskA.1.csv', index=False)

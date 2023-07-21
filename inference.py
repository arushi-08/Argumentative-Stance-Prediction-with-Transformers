# %tb
# inference

from transformers import Trainer, TrainingArguments
from datasets import load_dataset

import re
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
from datasets import load_metric
from transformers import TrainingArguments, Trainer, GroupViTModel, ViltForImagesAndTextClassification,ViltProcessor
from PIL import Image
import torch
import numpy as np

import pdb

from google.colab import drive
drive.mount('/content/drive')

model_name='xlm_roberta_xlarge'
dataset = 'abortion'

model = AutoModelForSequenceClassification.from_pretrained(
    f'/content/drive/MyDrive/ImageArg-Shared-Task/text_model_trainer/{dataset}/model_name/checkpoint-400', # Path to the directory containing the model.bin file
    local_files_only=True # Only load from local files
)
tokenizer = AutoTokenizer.from_pretrained(
    "xlm-roberta-large"
    )

def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = tweet.lower()
    return tweet


test_dataset = pd.read_csv(f'/content/drive/MyDrive/ImageArg-Shared-Task/data/{dataset}_test.csv')

test_predictions = {}
for idx, row in test_dataset.iterrows():

  tweet_text = preprocess_tweet(row['tweet_text'])
  input_ids = tokenizer.encode(tweet_text, return_tensors='pt')

  # Perform inference
  with torch.no_grad():
      logits = model(input_ids)[0]
      probabilities = torch.softmax(logits, dim=1).numpy()

  test_predictions[row['tweet_id']] = np.argmax(probabilities)
  # break

pd.DataFrame.from_dict(
    test_predictions, orient='index'
    ).reset_index().to_csv(
        f'/content/drive/MyDrive/ImageArg-Shared-Task/data/{dataset}_test_predictions_{model_name}_text_model.csv'
        )
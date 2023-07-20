# !pip install optuna transformers sentencepiece datasets accelerate -U

# %tb
from transformers import AutoModelForSequenceClassification, AutoProcessor, Trainer, TrainingArguments
from datasets import load_dataset

import re
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, TrainingArguments, Trainer, GroupViTModel, ViltForImagesAndTextClassification,ViltProcessor
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import numpy as np
import random

import pdb

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")

def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = tweet.lower()
    return tweet


def preprocess_function(examples):
    batch_size = 1
    num_images = 1

    tweet_text = preprocess_tweet(examples['tweet_text'])
    if examples['stance'] == 'oppose':
      stance = 0
    else:
      stance = 1

    image = np.array(Image.open(f"/content/drive/MyDrive/ImageArg-Shared-Task/data/images/gun_control/{examples['tweet_id']}.jpg").convert('RGB'))
    # Sometimes PIL returns a 2D tensor (for black-white images),
    # which is not supported by ViLT
    if len(image.shape) == 2:
        image = np.stack([image] * 3, -1)

    inputs = processor(text=tweet_text, images=image, padding="max_length", truncation=True, return_tensors="pt",)

    inputs['labels'] = stance

    for k in ["input_ids", "token_type_ids", "attention_mask"]:
        inputs[k] = inputs[k].squeeze()

    pixel_value_shape = inputs["pixel_values"].shape
    num_channels = pixel_value_shape[1]
    height = pixel_value_shape[2]
    width = pixel_value_shape[3]

    inputs["pixel_values"] = inputs["pixel_values"].reshape(
        [
            num_images,
            num_channels,
            height,
            width,
        ]
    )
    inputs["pixel_mask"] = inputs["pixel_mask"].reshape(
        [num_images, height, width]
    )

    return inputs

train_dataset =  load_dataset('csv', data_files='/content/drive/MyDrive/ImageArg-Shared-Task/data/gun_control_train.csv', split='train')
train_dataset = train_dataset.map(remove_columns=["tweet_url", "persuasiveness", "split"])

train_dataset = train_dataset.map(preprocess_function)
train_dataset = train_dataset.map(remove_columns=["tweet_text","tweet_id", "stance"])

val_dataset =  load_dataset('csv', data_files='/content/drive/MyDrive/ImageArg-Shared-Task/data/gun_control_dev.csv', split='train')
val_dataset = val_dataset.map(remove_columns=["tweet_url", "persuasiveness", "split"])
val_dataset = val_dataset.map(preprocess_function)
val_dataset = val_dataset.map(remove_columns=["tweet_text", "tweet_id", "stance"])


# Obtain label2id mappings
label_list = list(set(train_dataset["labels"]))
num_labels = len(label_list)

label2id = {v: i for i, v in enumerate(label_list)}
id2label = {id: label for label, id in label2id.items()}


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = balanced_accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}



class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        # extract labels from inputs
        labels = inputs.pop("labels")

        # compute model outputs
        outputs = model(**inputs)

        # compute loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# define class weights
# weights = torch.tensor([0.5, 2.0]) # example weights for a 2-class problem
class_counts = torch.bincount(torch.tensor(train_dataset['labels']))
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)



def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [1]),
    }



def model_init(trial):
    return ViltForImagesAndTextClassification.from_pretrained(
        "dandelin/vilt-b32-finetuned-nlvr2",
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        num_images=1,
        modality_type_vocab_size=2,
        ignore_mismatched_sizes=True,
    )


args = TrainingArguments(
    output_dir="/content/drive/MyDrive/ImageArg-Shared-Task/multi_modal_trainer/gun_control_optuna_hp",
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    seed=0,
    load_best_model_at_end=True,
)

# create trainer with custom compute_loss function
trainer = CustomTrainer(
    model=None,
    model_init=model_init,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    class_weights=class_weights,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20,
    # compute_objective=compute_objective,
)


trainer.train()




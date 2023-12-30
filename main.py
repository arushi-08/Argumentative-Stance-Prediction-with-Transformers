import os
import re
import logging
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    ViltProcessor,
    ViltForImagesAndTextClassification,
    FlavaProcessor,
    FlavaModel,
)
from PIL import Image
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from config import *
from classification_head import ClassificationHead

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def preprocess_tweet(tweet):
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"#", "", tweet)
    tweet = re.sub(r"[^\w\s]", "", tweet)
    tweet = re.sub(r"\s+", " ", tweet)
    tweet = tweet.lower()
    return tweet


parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    type=str,
    help="Enter dataset: gun_control, abortion",
    default="gun_control",
)
parser.add_argument(
    "model_ckpt",
    type=str,
    help=f"Enter model: e.g. text models like {MODEL_CKPT}, vilt, flava",
    default=MODEL_CKPT,
)
parser.add_argument(
    "model_type",
    type=str,
    help=f"Enter model: e.g. {MODEL_TYPE}",
    default=MODEL_TYPE,
)
args = parser.parse_args()
dataset = args.dataset
model_ckpt = args.model_ckpt
model_type = args.model_type

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model_type == "multimodal":
    label2id_dict = {v: i for i, v in enumerate(range(NUM_LABELS))}
    id2label_dict = {id: label for label, id in label2id_dict.items()}

    if "vilt" in model_ckpt:
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
        model = ViltForImagesAndTextClassification.from_pretrained(
            "dandelin/vilt-b32-finetuned-nlvr2",
            num_labels=NUM_LABELS,
            label2id=label2id_dict,
            id2label=id2label_dict,
            num_images=1,
            modality_type_vocab_size=2,
            ignore_mismatched_sizes=True,
        )
    elif "flava" in model_ckpt:
        processor = FlavaProcessor.from_pretrained("facebook/flava-full")
        flava = FlavaModel.from_pretrained("facebook/flava-full")
        # flava model doesn't have a classification head in huggingface
        model = ClassificationHead(flava)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=NUM_LABELS
    ).to(device)


def tokenize(batch):
    return tokenizer(batch["tweet_text"], padding=True, truncation=True, max_length=512)


def preprocess_data(data, model_type="text"):
    data["tweet_text"] = preprocess_tweet(data["tweet_text"])

    if model_type == "text":
        data_encoded = tokenize(data)

    elif model_type == "multimodal":
        image = np.array(
            Image.open(
                os.path.join(DATA_PATH, f"images/{dataset}/{data['tweet_id']}.jpg")
            ).convert("RGB")
        )
        # Sometimes PIL returns a 2D tensor (for black-white images),
        # which is not supported by ViLT
        if len(image.shape) == 2:
            image = np.stack([image] * 3, -1)
        data_encoded = processor(
            text=data["tweet_text"],
            images=image,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        data_encoded["labels"] = stance

        for k in ["input_ids", "token_type_ids", "attention_mask"]:
            data_encoded[k] = data_encoded[k].squeeze()

        pixel_value_shape = data_encoded["pixel_values"].shape
        num_channels = pixel_value_shape[1]
        height = pixel_value_shape[2]
        width = pixel_value_shape[3]

        num_images = 1
        data_encoded["pixel_values"] = data_encoded["pixel_values"].reshape(
            [
                num_images,
                num_channels,
                height,
                width,
            ]
        )
        data_encoded["pixel_mask"] = data_encoded["pixel_mask"].reshape(
            [num_images, height, width]
        )

    if data_encoded["stance"] == "oppose":
        stance = 0
    else:
        stance = 1
    data_encoded["labels"] = stance

    return data_encoded


def prepare_data(dataset, dtype):
    dataset = load_dataset(
        "csv", data_files=os.path.join(DATA_PATH, f"{dataset}_{dtype}.csv"), split=dtype
    )
    dataset.map(remove_columns=["tweet_url", "persuasiveness", "split"])
    dataset_encoded = dataset.map(
        preprocess_data,
        batched=True,
        batch_size=None,
        fn_kwargs={"model_type": model_type},
    )
    dataset_encoded.map(remove_columns=["tweet_text", "tweet_id", "stance"])
    dataset_encoded.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return dataset_encoded


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, data_encoded, return_outputs=False):
        labels = data_encoded.pop("labels")
        outputs = model(**data_encoded)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(
            outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def main():
    logging.info(f"Preparing dataset: {dataset}")
    train_dataset = prepare_data(dataset, "train")
    dev_dataset = prepare_data(dataset, "dev")

    output_dir = os.path.join(TRAINER_PATH, dataset)

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=EVAL_STRATEGY,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        seed=0,
        load_best_model_at_end=True,
    )

    class_counts = torch.bincount(torch.tensor(train_dataset["labels"]))
    class_weights = 1.0 / class_counts.float()
    class_weights = (class_weights / class_weights.sum()).to(device)

    logging.info(f"Preparing training step")
    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        class_weights=class_weights,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    logging.info(f"Training complete. Model stored in {output_dir}")

    logging.info(f"Preparing testing step")
    test_dataset = prepare_data(dataset, "test")
    preds_output = trainer.predict(test_dataset)

    logging.info("Model performance:")
    logging.info(preds_output.metrics)

    y_preds = np.argmax(preds_output.predictions, axis=1)
    test_dataset.set_format("pandas")
    test_dataset["predicted_label"] = y_preds

    pred_output_dir = os.path.join(DATA_PATH, "predictions")
    if not os.path.exists(pred_output_dir):
        os.makedirs(pred_output_dir)

    pred_output_dir = os.path.join(
        pred_output_dir, f"{dataset}_predictions_{model_ckpt}.csv"
    )
    test_dataset.to_csv(pred_output_dir)
    logging.info(f"Inference complete. Predictions stored in {pred_output_dir}")


if __name__ == "__main__":
    main()

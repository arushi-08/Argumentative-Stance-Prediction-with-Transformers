from torch import nn

from config import NUM_LABELS


class ClassificationHead(nn.Module):
    def __init__(self, model_ckpt):
        super().__init__()
        self.pretrained_model_ckpt = model_ckpt
        self.fc = nn.Linear(self.pretrained_model_ckpt.config.hidden_size, NUM_LABELS)

    def forward(
        self, input_ids, pixel_values, attention_mask=None, token_type_ids=None
    ):
        # Pass inputs through Flava model
        flava_output = self.pretrained_model_ckpt(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract the last hidden state of the text encoder
        text_last_hidden_state = flava_output[0]

        # Extract the last hidden state of the [CLS] token
        cls_output = text_last_hidden_state[:, 0]

        # Pass [CLS] token output through linear layer
        logits = self.fc(cls_output)

        return logits

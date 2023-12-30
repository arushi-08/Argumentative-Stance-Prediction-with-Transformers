# EMNLP-ImageArgTask-PittPixelPersuaders

Finetuning HuggingFace Transformer models for the EMNLP'23 ImageArg Shared Task.  
The task comprised of Argumentative Stance Prediction subtask using 2 models: gun_control and abortion (see the details on the [website](https://imagearg.github.io)).

To run the experiments:
1. Install requirements.txt
```
pip install -r requirements.txt
```
3. Run the main.py
```
python main.py --dataset gun_control --model_ckpt xlnet-base-cased --model_type text
```

The `main.py` file will finetune the model_checkpoint (`model_ckpt`) and also run inference on test dataset and generate results.

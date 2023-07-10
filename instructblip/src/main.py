import os, sys
sys.path.append('/mnt/Volume1/ImgArg/ImageArg-Shared-Task/instructblip')
os.CUDA_VISIBLE_DEVICES = '0,1'
import pdb

# from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
# import deepspeed
# from utils.extract_descriptions import extract_descriptions
from utils.data import initialize_data
from utils.roberta import train

nepochs = 50
train_loader, valid_loader, test_loader = initialize_data(batchsize=16, promts='Can you summarize the image in brief?')
train(train_loader, valid_loader, test_loader, nepochs)

# prompts = ['What does the text on image say?', 'How many people are in the image?', 'Can you summarize the image in brief?']
# img_dir = '../data/images/abortion'
# filename = '../data/abortion_instructblip_descriptions.csv'
# task_name = 'abortion'


# extract_descriptions(prompts=prompts, img_dir=img_dir, filename=filename, task_name=task_name)


# model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", load_in_8bit=True)
# ds_engine = deepspeed.init_inference(model,
#                                  mp_size=2,
#                                 #  dtype=torch.half,
#                                  checkpoint=None,
#                                  replace_with_kernel_inject=True)
# model = ds_engine.module
# processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")

# #device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
# model.to(device)

# def inference(image, prompt):
#     inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
#     outputs = model.generate(
#         **inputs,
#         do_sample=False,
#         num_beams=5,
#         max_length=256,
#         min_length=1,
#         top_p=0.9,
#         repetition_penalty=1.5,
#         length_penalty=1.0,
#         temperature=1,
#     )
#     generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
#     print(prompt+'\n'+generated_text)

#     return generated_text

# 4 scarfed people with firearms and text on image
# image = Image.open('../data/images/gun_control/1346718979587452929.jpg')

# image with text "Senator: we dissent"
# image = Image.open('../data/images/gun_control/1314258017018355713.jpg')

# US supreme court with text and hashtags
# image = Image.open('../data/images/gun_control/1314260551183327234.jpg')

# prompt = "What is unusual about this image?"
# inference(image, prompt)
# print('-'*25)

# prompt = "What does the text on image say?"
# inference(image, prompt)
# print('-'*25)

# prompt = "How many people are in the image?"
# inference(image, prompt)
# print('-'*25)

# prompt = "Can you summarize the image in brief?"
# inference(image, prompt)
# print('-'*25)

# prompt = "What is the gender of the people in the image?"
# inference(image, prompt)
# print('-'*25)

# prompt = "What is the ethnicity of people in the image?"
# inference(image, prompt)
# print('-'*25)

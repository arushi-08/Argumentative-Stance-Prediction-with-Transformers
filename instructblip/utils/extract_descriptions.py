import os, pdb
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
# device = 'cuda'
device = 'cpu'
# model.to(device)

def extract_descriptions(prompts, img_dir, filename, task_name):
    print(f'Extracting descriptions for task: {task_name}')
    output = dict()
    output['img'] = []
    for prompt in prompts:
        output[prompt.replace(' ', '_')] = []
    for img in tqdm(os.listdir(img_dir)):
        output['img'].append(img)
        image = Image.open(os.path.join(img_dir, img))
        for prompt in prompts:
            description = inference(image, prompt)
            # output['prompts'].append({'prompt': prompt, 'description': description})
            output[prompt.replace(' ', '_')].append(description)
    output['task'] = task_name
    write_descriptions_to_csv(output, filename)
    return

def inference(image, prompt, verbose=False):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    if verbose:
        print(prompt+'\n'+generated_text)

    return generated_text

def write_descriptions_to_csv(descriptions, filename):
    pdb.set_trace()
    return pd.DataFrame.from_dict(descriptions).to_csv(filename, index=False)
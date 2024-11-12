import json
import os

import dotenv
from huggingface_hub import snapshot_download
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline

def argument_detection(data, HF_TOKEN):
    hf_repo_url = "armaniii/llama-3-8b-argument-detection"
    local_directory = "./models/wiba/llama-3-8b-argument-detection"

    snapshot_download(repo_id=hf_repo_url,local_dir=local_directory)

    with open('./models/wiba/system_message_arg.txt', 'r') as file:
        system_message = file.read()

    tokenizer = AutoTokenizer.from_pretrained(local_directory)

    try:
        import peft
    except ImportError:
        raise ImportError("PEFT is not installed. Please install it by running 'pip install peft'")

    model = AutoModelForSequenceClassification.from_pretrained(local_directory, num_labels=2, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN)

    model.eval()

    # Using Pipeline
    pipe = pipeline(task="text-classification", model=model, tokenizer=tokenizer, padding=True, truncation=True, device_map="auto", max_length=2048, torch_dtype=torch.float16)

    data['sentence'] = data['text'].map(lambda x: f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\nText: '{x}' [/INST] ")
    inputs = data['sentence']
    prompts_generator = (p for p in inputs)
    results = []
    for out in tqdm(pipe(prompts_generator,batch_size=1)):          
        if out['label'] == 'LABEL_1':
            results.append('Argument')
        else:
            results.append('NoArgument')

    model.to('cpu')
    del model
    del pipe
    torch.cuda.empty_cache()

    data['is_argument'] = results
    return data

def target_extraction(data, HF_TOKEN):
    model = AutoModelForCausalLM.from_pretrained("armaniii/llama-3-8b-claim-topic-extraction",torch_dtype=torch.float16,device_map="auto",low_cpu_mem_usage = True, token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained("armaniii/llama-3-8b-claim-topic-extraction",use_fast = False)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id= 18610
    tokenizer.padding_side = "left"

    with open('./models/wiba/system_message_cte.txt', 'r') as file:
        system_message = file.read()

    # Using Pipeline
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer,max_new_tokens=8,device_map="auto",torch_dtype=torch.float16,pad_token_id=128009)
    data['sentence'] = data['text'].map(lambda x: f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message.strip()}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{x}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
    inputs = data['sentence']
    prompts_generator = (p for p in inputs)
    results = []
    for out in tqdm(pipe(prompts_generator,batch_size=4)):          
        result=out[0]['generated_text']
        result = result.split("assistant<|end_header_id|>\n\n")[1]
        results.append(result)

    model.to('cpu')
    del model
    del pipe
    torch.cuda.empty_cache()

    data['topic'] = results
    return data

def stance_detection(data, HF_TOKEN):
    hf_repo_url = "armaniii/llama-stance-classification"
    local_directory = "./models/wiba/llama-stance-classification"
    tokenizer_local_directory = "./models/wiba/llama-3-8b-argument-detection"

    snapshot_download(repo_id=hf_repo_url,local_dir=local_directory)

    with open('./models/wiba/system_message_arg.txt', 'r') as file:
        system_message = file.read()

    with open(os.path.join(local_directory, 'adapter_config.json'), 'r') as file:
        adapter_config = json.load(file)

    base_model = adapter_config['base_model_name_or_path']

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(local_directory,num_labels=3, torch_dtype=torch.float16,device_map="auto",token=HF_TOKEN)
    model.eval()

    # Using Pipeline
    pipe = pipeline(task="text-classification", model=model, tokenizer=tokenizer,padding=True,truncation=True,device_map="auto",max_length=2048,torch_dtype=torch.float16)

    data['sentence'] = data[['topic', 'text']].apply(lambda r: f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\nTarget: '{r['topic']}' Text: '{r['text']}' [/INST] ", axis=1)
    inputs = data['sentence']
    prompts_generator = (p for p in inputs)
    results = []
    for out in tqdm(pipe(prompts_generator,batch_size=1)):          
        if out['label'] == 'LABEL_0':
            results.append('No Argument')
        elif out['label'] == 'LABEL_1':
            results.append('Argument in Favor')
        else:
            results.append('Argument Against')
    data['stance'] = results

    model.to('cpu')
    del model
    del pipe
    torch.cuda.empty_cache()

    return data

class Wiba:
    def __init__(self):
        pass

    def fit_transform(self, docs):
        dotenv.load_dotenv()
        HF_TOKEN = os.environ['HF_TOKEN']

        # https://github.com/Armaniii/WIBA
        data = pd.DataFrame(docs, columns=['text'])
        
        data = argument_detection(data, HF_TOKEN)
        data = target_extraction(data, HF_TOKEN)
        data = stance_detection(data, HF_TOKEN)
        data = data[['text', 'is_argument', 'topic', 'stance']]

        docs = data['text'].to_list()
        doc_targets = data['topic'].to_list()
        self.all_targets = list(set(doc_targets))
        target_to_idx = {target: idx for idx, target in enumerate(self.all_targets)}
        probs = np.zeros((len(docs), len(self.all_targets)))
        for idx, target in enumerate(doc_targets):
            probs[idx, target_to_idx[target]] = 1
        polarity = np.zeros((len(docs), len(self.all_targets)))
        for idx, target in enumerate(doc_targets):
            if data['stance'][idx] == 'Argument in Favor':
                polarity[idx, target_to_idx[target]] = 1
            elif data['stance'][idx] == 'Argument Against':
                polarity[idx, target_to_idx[target]] = -1
        return doc_targets, probs, polarity
    
    def get_target_info(self):
        return pd.DataFrame({'ngram': self.all_targets})

    

    
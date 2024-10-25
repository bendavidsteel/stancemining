import os

import dotenv
from huggingface_hub import snapshot_download
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
    model = AutoModelForSequenceClassification.from_pretrained(local_directory, num_labels=2, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN, local_files_only=True)

    model.eval()

    # Using Pipeline
    pipe = pipeline(task="text-classification", model=model, tokenizer=tokenizer, padding=True, truncation=True, device_map="auto", max_length=2048, torch_dtype=torch.float16)

    system_message = ""
    data= data.map(lambda x: {"sentence":[ f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Text: '" + sentence +"' [/INST] " for sentence in x['text']]}, batched=True)
    inputs = data['sentence']
    prompts_generator = (p for p in inputs)
    results = []
    for out in tqdm(pipe(prompts_generator,batch_size=4)):          
        if out['label'] == 'LABEL_1':
            results.append('Argument')
        else:
            results.append('NoArgument')

    data['argument_predictions'] = results
    return data

def target_extraction(data, HF_TOKEN):
    model = AutoModelForCausalLM.from_pretrained("armaniii/llama-3-8b-claim-topic-extraction",torch_dtype=torch.float16,device_map="auto",low_cpu_mem_usage = True, token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained("armaniii/llama-3-8b-claim-topic-extraction",use_fast = False)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id= 18610
    tokenizer.padding_side = "left"

    # Using Pipeline
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer,max_new_tokens=8,device_map="auto",torch_dtype=torch.float16,pad_token_id=128009)
    data= data.map(lambda x: {"sentence":[ f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message.strip()}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n' + sentence +'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' for sentence in x['text']]}, batched=True)
    inputs = data['sentence']
    prompts_generator = (p for p in inputs)
    results = []
    for out in tqdm(pipe(prompts_generator,batch_size=4)):          
        result=out[0]['generated_text']
        result = result.split("assistant<|end_header_id|>\n\n")[1]
        results.append(result)

    data['argument_predictions'] = results
    return data

def stance_detection(data, HF_TOKEN):
    hf_repo_url = "armaniii/llama-stance-classification"
    local_directory = "./models/wiba/llama-stance-classification"

    snapshot_download(repo_id=hf_repo_url,local_dir=local_directory)

    system_message

    tokenizer = AutoTokenizer.from_pretrained(local_directory)
    model = AutoModelForSequenceClassification.from_pretrained(local_directory,num_labels=3, torch_dtype=torch.float16,device_map="auto",token=HF_TOKEN,local_files_only=True)
    model.eval()

    # Using Pipeline
    pipe = pipeline(task="text-classification", model=model, tokenizer=tokenizer,padding=True,truncation=True,device_map="auto",max_length=2048,torch_dtype=torch.float16)

    data= data.map(lambda x: {"sentence":[ f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Target: '" + topic +"' Text: '" +inputs  + "' [/INST] " for topic,inputs in zip(x['topic'],x['text'])]}, batched=True)
    inputs = data['sentence']
    prompts_generator = (p for p in inputs)
    results = []
    for out in tqdm(pipe(prompts_generator,batch_size=4)):          
        if out['label'] == 'LABEL_0':
            results.append('No Argument')
        elif out['label'] == 'LABEL_1':
            results.append('Argument in Favor')
        else:
            results.append('Argument Against')
    data['argument_predictions'] = results

    return data

def wiba(docs):
    dotenv.load_dotenv()
    HF_TOKEN = os.environ['HF_TOKEN']

    # https://github.com/Armaniii/WIBA
    data = pd.DataFrame(docs, columns=['text'])
    
    data = argument_detection(data, HF_TOKEN)
    data = target_extraction(data, HF_TOKEN)
    data = stance_detection(data, HF_TOKEN)
    return data

    

    
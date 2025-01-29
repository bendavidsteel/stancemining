import os

import dotenv
import huggingface_hub
import numpy as np
import pandas as pd
import polars as pl
import torch

from stancemining.finetune import get_predictions

def argument_detection(data, config, model_path, token):
    if model_path is None:
        hf_repo_url = "armaniii/llama-3-8b-argument-detection"
        local_directory = "./models/wiba/llama-3-8b-argument-detection"

        huggingface_hub.snapshot_download(repo_id=hf_repo_url,local_dir=local_directory)

    # with open('./models/wiba/system_message_arg.txt', 'r') as file:
    #     system_message = file.read()

    # tokenizer = AutoTokenizer.from_pretrained(local_directory)

    # try:
    #     import peft
    # except ImportError:
    #     raise ImportError("PEFT is not installed. Please install it by running 'pip install peft'")

    # model = AutoModelForSequenceClassification.from_pretrained(local_directory, num_labels=2, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN)

    # model.eval()

    # # Using Pipeline
    # pipe = pipeline(task="text-classification", model=model, tokenizer=tokenizer, padding=True, truncation=True, device_map="auto", max_length=2048, torch_dtype=torch.float16)

    # data['sentence'] = data['text'].map(lambda x: f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\nText: '{x}' [/INST] ")
    # inputs = data['sentence']
    # prompts_generator = (p for p in inputs)
    # results = []
    # for out in tqdm(pipe(prompts_generator,batch_size=1)):          
    #     if out['label'] == 'LABEL_1':
    #         results.append('Argument')
    #     else:
    #         results.append('NoArgument')
    results = get_predictions("argument-classification", data, model_path, config, hf_token=token)

    model.to('cpu')
    del model
    del pipe
    torch.cuda.empty_cache()

    data = data.with_columns(pl.Series(name='is_argument', values=results))
    return data

def target_extraction(data, config, model_path, token):
    if model_path is None:
        model_path = "armaniii/llama-3-8b-claim-topic-extraction"

    print(f"Extracting topics using model: {model_path}")
    results = get_predictions("topic-extraction", data, model_path, config, hf_token=token)

    data = data.with_columns(pl.Series(name='topic', values=results))
    return data

def stance_detection(data, config, model_path, token):
    if model_path is None:
        hf_repo_url = "armaniii/llama-stance-classification"
        local_directory = "./models/wiba/llama-stance-classification"
        tokenizer_local_directory = "./models/wiba/llama-3-8b-argument-detection"

        huggingface_hub.snapshot_download(repo_id=hf_repo_url,local_dir=local_directory)

    print(f"Detecting stance using model: {model_path}")
    results = get_predictions("stance-classification", data, model_path, config, hf_token=token)
    data = data.with_columns(pl.Series(name='stance', values=results))

    return data

class Wiba:
    def __init__(self):
        pass

    def fit_transform(self, docs, config, argument_detection_path=None, topic_extraction_path=None, stance_classification_path=None):
        dotenv.load_dotenv()
        HF_TOKEN = os.environ['HF_TOKEN']

        # https://github.com/Armaniii/WIBA
        data = pl.DataFrame({'text': docs})
        
        if argument_detection_path is not None:
            data = argument_detection(data, config, argument_detection_path, token=HF_TOKEN)
        else:
            # just set all as arguments
            data = data.with_columns(pl.lit('Argument').alias('is_argument'))

        
        data = target_extraction(data, config, topic_extraction_path, HF_TOKEN)
        data = stance_detection(data, config, stance_classification_path, HF_TOKEN)
        data = data.select(['text', 'is_argument', 'topic', 'stance'])

        docs = data['text'].to_list()
        doc_targets = data['topic'].to_list()
        doc_targets = [[t] if not isinstance(t, list) else t for t in doc_targets]
        self.all_targets = list(set(data['topic'].unique()))
        target_to_idx = {target: idx for idx, target in enumerate(self.all_targets)}
        probs = np.zeros((len(docs), len(self.all_targets)))
        for idx, targets in enumerate(doc_targets):
            for target in targets:
                probs[idx, target_to_idx[target]] = 1
        polarity = np.full((len(docs), len(self.all_targets)), np.nan)
        for idx, targets in enumerate(doc_targets):
            for target in targets:
                if data['stance'][idx] == 'favor':
                    polarity[idx, target_to_idx[target]] = 1
                elif data['stance'][idx] == 'against':
                    polarity[idx, target_to_idx[target]] = -1
                elif data['stance'][idx] == 'neutral':
                    polarity[idx, target_to_idx[target]] = 0
                else:
                    raise ValueError(f"Unknown stance: {data['stance'][idx]}")
        return doc_targets, probs, polarity
    
    def get_target_info(self):
        return pl.DataFrame({'noun_phrase': self.all_targets})

    

    
from collections.abc import Iterable
from dataclasses import dataclass, field
import gc
import json
import multiprocessing
import os
import pathlib
import re
from typing import Optional, Dict, List, Any, Union

import accelerate
import datasets
import evaluate
import huggingface_hub
import numpy as np
import pandas as pd
import peft
import polars as pl
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import tqdm
import transformers
import wandb

import stancemining.datasets

CLASSIFICATION_TASKS = ['stance-classification', 'argument-classification', 'claim-entailment-2way', 'claim-entailment-3way', 'claim-entailment-4way', 'claim-entailment-5way', 'claim-entailment-7way']
GENERATION_TASKS = ['topic-extraction', 'claim-extraction']

def load_split_data(dataset_name: str, split: str, task: str, generation_method: str) -> pl.DataFrame:
    return stancemining.datasets.load_dataset(
        dataset_name, 
        split=split, 
        group=(generation_method=='list') and (task in GENERATION_TASKS), 
        remove_synthetic_neutral=task not in CLASSIFICATION_TASKS,
        task=task
    )

def load_training_data(dataset_name: str, task: str, generation_method: str) -> pl.DataFrame:
    return load_split_data(dataset_name, "train", task, generation_method)

def load_validation_data(dataset_name: str, task: str, generation_method: str) -> pl.DataFrame:
    return load_split_data(dataset_name, "val", task, generation_method)

def load_test_data(dataset_name: str, task: str, generation_method: str) -> pl.DataFrame:
    return load_split_data(dataset_name, "test", task, generation_method)

def save_predictions(predictions: List[Any], df: pd.DataFrame, save_path: str) -> None:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df['predictions'] = predictions
    df.to_csv(save_path + "/predictions.csv", index=False)

def print_metrics(metrics: Dict[str, float]) -> None:
    for metric, value in metrics.items():
        if metric in ['accuracy', 'f1', 'precision', 'recall', 'bertscore_f1', 'bertscore_p', 'bertscore_r', 'bleu_f1', 'bleu_p', 'bleu_r']:
            value = round(value, 3)
        print(f"{metric}: {value}")

def get_model_save_path(task, model_path_dir, model_name, dataset_name, output_type):
    if task == "argument-classification":
        model_path_name = model_path_dir + "/" + model_name.replace('/', '-') + "-argument-detection"
    else:
        model_path_name = model_path_dir + "/" + model_name.replace('/', '-') + "-" + task
    if isinstance(dataset_name, str):
        model_path_name = model_path_name + f"-{dataset_name}"
    elif isinstance(dataset_name, Iterable):
        d_name = '-'.join(dataset_name)
        model_path_name = model_path_name + f"-{d_name}"
    model_path_name += f"-{output_type}"

    return model_path_name

def load_prompt(task: str, prompting_method: str, generation_method: str = None) -> str:
    top_dir = pathlib.Path(__file__).parent.parent
    if task == "stance-classification" or task == "argument-classification":
        if prompting_method == 'wiba':
            file_path = top_dir / 'models/wiba/system_message_arg.txt'
        elif prompting_method == 'stancemining':
            file_path = top_dir / 'models/prompts/prompt_stance.txt'
        else:
            raise ValueError("Prompting method not found")
    elif task == "topic-extraction":
        if prompting_method == 'wiba':
            file_path = top_dir / 'models/wiba/system_message_cte.txt'
        elif prompting_method == 'stancemining':
            if generation_method == 'beam':
                file_path = top_dir / 'models/prompts/prompt_stance_target.txt'
            elif generation_method == 'list':
                file_path = top_dir / 'models/prompts/prompt_stance_target_list.txt'
            else:
                raise ValueError("Generation method not found")
        else:
            raise ValueError("Prompting method not found")
    elif task == "claim-extraction":
        file_path = top_dir / 'models/prompts/prompt_claim_extraction.txt'
    elif task == "claim-entailment-2way":
        file_path = top_dir / 'models/prompts/prompt_claim_entailment_2_way.txt'
    elif task == "claim-entailment-3way":
        file_path = top_dir / 'models/prompts/prompt_claim_entailment.txt'
    elif task == "claim-entailment-4way":
        file_path = top_dir / 'models/prompts/prompt_claim_entailment_4_way.txt'
    elif task == "claim-entailment-5way":
        file_path = top_dir / 'models/prompts/prompt_claim_entailment_5_way.txt'
    elif task == "claim-entailment-7way":
        file_path = top_dir / 'models/prompts/prompt_claim_entailment_7_way.txt'
    else:
        raise ValueError("Task not found")
    with open(file_path, 'r') as file:
        system_message = file.read()
    return system_message

def load_parent_prompt(task: str, prompting_method: str) -> str:
    top_dir = pathlib.Path(__file__).parent.parent
    if task == "stance-classification" or task == "argument-classification":
        if prompting_method == 'wiba':
            return ''
        elif prompting_method == 'stancemining':
            file_path = top_dir / 'models/prompts/prompt_parent_stance.txt'
        else:
            raise ValueError("Prompting method not found")
    elif task == "topic-extraction":
        if prompting_method == 'wiba':
            return ''
        elif prompting_method == 'stancemining':
            file_path = top_dir / 'models/prompts/prompt_parent_stance_target.txt'
        else:
            raise ValueError("Prompting method not found")
    elif task == "claim-extraction":
        file_path = top_dir / 'models/prompts/prompt_claim_extraction.txt'
    elif task == "claim-entailment-2way":
        file_path = top_dir / 'models/prompts/prompt_parent_claim_entailment_2_way.txt'
    elif task == "claim-entailment-3way":
        file_path = top_dir / 'models/prompts/prompt_parent_claim_entailment.txt'
    elif task == "claim-entailment-4way":
        file_path = top_dir / 'models/prompts/prompt_parent_claim_entailment_4_way.txt'
    elif task == "claim-entailment-5way":
        file_path = top_dir / 'models/prompts/prompt_parent_claim_entailment_5_way.txt'
    elif task == "claim-entailment-7way":
        file_path = top_dir / 'models/prompts/prompt_parent_claim_entailment_7_way.txt'
    else:
        raise ValueError("Task not found")
    with open(file_path, 'r') as file:
        system_message = file.read()
    return system_message

def load_context_prompt(task: str, prompting_method: str) -> str:
    top_dir = pathlib.Path(__file__).parent.parent
    if task == 'claim-entailment-2way':
        file_path = top_dir / 'models/prompts/prompt_context_claim_entailment_2_way.txt'
    elif task == 'claim-entailment-4way':
        file_path = top_dir / 'models/prompts/prompt_context_claim_entailment_4_way.txt'
    elif task == "claim-entailment-5way":
        file_path = top_dir / 'models/prompts/prompt_context_claim_entailment_5_way.txt'
    elif task == "stance-classification":
        file_path = top_dir / 'models/prompts/prompt_context_stance.txt'
    else:
        raise ValueError("Task not found")
    with open(file_path, 'r') as file:
        system_message = file.read()
    return system_message

def stance_example_to_prompt(examples, i, prompt_template: str, parent_prompt_template: str, context_prompt_template: str):
    text = examples['text'][i]
    target = examples['topic'][i]
    if 'parenttexts' in examples:
        parenttexts = examples['parenttexts'][i]
    else:
        parenttexts = None
    if 'context' in examples:
        context = examples['context'][i]
    else:
        context = None
    kwargs = {
        'target': target,
        'text': text,
    }
    if context:
        kwargs['context'] = context
        prompt_template = context_prompt_template
    elif parenttexts and len(parenttexts) > 0:
        parent_chain = []
        for i, p_text in enumerate(parenttexts):
            if i == 0:
                parent_chain.append(f"1. [Original Post]: '{p_text}'")
            else:
                parent_chain.append(f"{i+1}. [Reply to {i}]: '{p_text}'")

        parent_chain = '\n'.join(parent_chain)
        kwargs['parent_chain'] = parent_chain
        prompt_template = parent_prompt_template
    else:
        prompt_template = prompt_template

    if isinstance(prompt_template, str):
        prompt = prompt_template.format(**kwargs)
    elif isinstance(prompt_template, list):
        prompt = [{k: v.format(**kwargs) for k, v in p.items()} for p in prompt_template]
    else:
        raise ValueError("Prompt template must be str or list")
    return prompt

def stance_examples_to_prompt(prompt_template: str, parent_prompt_template: str, context_prompt_template: str, examples):
    prompts = []
    for i in range(len(examples['text'])):
        prompt = stance_example_to_prompt(examples, i, prompt_template, parent_prompt_template, context_prompt_template)
        prompts.append(prompt)
    return prompts

def convert_list_to_quoted_str(topic):
    topic = sorted(topic, key=lambda t: len(t))
    topic = [f'"{t}"' for t in topic]
    return ', '.join(topic)

def stance_target_examples_to_prompt(prompt_template: str, examples):
    prompts = []
    for i in range(len(examples['text'])):
        text = examples['text'][i]
        prompt = prompt_template.format(text=text)
        prompts.append(prompt)
    return prompts

def to_message_format(text, label=None, system_message="You are a helpful assistant."):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": text},
    ]
    if label is not None:
        messages.append({"role": "assistant", "content": label})
    return messages

def activate_neftune(model, accelerator, neftune_noise_alpha):
    r"""
    Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper:
    https://arxiv.org/abs/2310.05914
    """
    unwrapped_model = accelerator.unwrap_model(model)

    if transformers.trainer._is_peft_model(unwrapped_model):
        embeddings = unwrapped_model.base_model.model.get_input_embeddings()
    else:
        embeddings = unwrapped_model.get_input_embeddings()

    del unwrapped_model

    embeddings.neftune_noise_alpha = neftune_noise_alpha
    hook_handle = embeddings.register_forward_hook(transformers.trainer_utils.neftune_post_forward_hook)
    return model, hook_handle

def deactivate_neftune(model, accelerator, neftune_hook_handle):
    """
    Deactivates the neftune method. Make sure to call `_activate_neftune` first.
    """
    if not neftune_hook_handle:
        raise ValueError("Neftune is not activated make sure to call `trainer._activate_neftune()` first")

    unwrapped_model = accelerator.unwrap_model(model)

    if transformers.trainer._is_peft_model(unwrapped_model):
        embeddings = unwrapped_model.base_model.model.get_input_embeddings()
    else:
        embeddings = unwrapped_model.get_input_embeddings()

    neftune_hook_handle.remove()
    del embeddings.neftune_noise_alpha, unwrapped_model

def get_labels_2_id(task: str):
    if task == 'stance-classification':
        num_labels = 3
        labels2id = STANCE_LABELS_2_ID
    elif task == 'argument-classification':
        num_labels = 2
        labels2id = {'Argumentative': 1, 'Non-Argumentative': 0}
    elif task == 'claim-entailment-2way':
        num_labels = 2
        labels2id = CLAIM_ENTAILMENT_2_WAY_LABELS_2_ID
    elif task == 'claim-entailment-3way':
        num_labels = 3
        labels2id = CLAIM_ENTAILMENT_3_WAY_LABELS_2_ID
    elif task == 'claim-entailment-4way':
        num_labels = 4
        labels2id = CLAIM_ENTAILMENT_4_WAY_LABELS_2_ID
    elif task == 'claim-entailment-5way':
        num_labels = 5
        labels2id = CLAIM_ENTAILMENT_5_WAY_LABELS_2_ID
    elif task == 'claim-entailment-7way':
        num_labels = 7
        labels2id = CLAIM_ENTAILMENT_7_WAY_LABELS_2_ID
    else:
        num_labels = None
        labels2id = None
    return num_labels, labels2id

@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"

class ModelConfig:
    model_name: str
    task: str
    num_labels: Optional[int]
    prompt: str
    device_map: Dict[str, int] = 'auto'
    parent_prompt: Optional[str] = None
    context_prompt: Optional[str] = None
    tokenizer: Optional[transformers.PreTrainedTokenizer] = None
    model: Optional[transformers.PreTrainedModel] = None
    classification_method: Optional[str] = "head"
    generation_method: Optional[str] = "beam"
    attn_implementation: Optional[str] = "flash_attention_2"
    torch_dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16
    quantization: Optional[str] = None
    labels2id: Dict[str, int] = field(default_factory=lambda: STANCE_LABELS_2_ID)
    lora_config: Optional[LoraConfig] = None

    def __init__(
        self,
        model_name: str,
        task: str,
        prompt: str,
        device_map: Dict[str, int] = 'auto',
        parent_prompt: Optional[str] = None,
        context_prompt: Optional[str] = None,
        tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        model: Optional[transformers.PreTrainedModel] = None,
        classification_method: Optional[str] = "head",
        generation_method: Optional[str] = "beam",
        attn_implementation: Optional[str] = "flash_attention_2",
        torch_dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        quantization: Optional[str] = None,
        lora_config: Optional[LoraConfig] = None,
    ):
        self.model_name = model_name
        self.task = task
        self.prompt = prompt
        self.device_map = device_map
        self.parent_prompt = parent_prompt
        self.context_prompt = context_prompt
        self.tokenizer = tokenizer
        self.model = model
        self.classification_method = classification_method
        self.generation_method = generation_method
        self.attn_implementation = attn_implementation
        self.torch_dtype = torch_dtype
        self.quantization = quantization
        self.num_labels, self.labels2id = get_labels_2_id(task)
        self.lora_config = lora_config
        
class ChatTemplateTokenizer:
    def __init__(self, model_config: ModelConfig):
        self.tokenizer = model_config.tokenizer
        self.generation_method = model_config.generation_method
        self.task = model_config.task
        if self.tokenizer.chat_template is None:
            if model_config.task == 'stance-classification':
                self.base_continuation_prompt = '. Stance: The stance is '
            else:
                raise NotImplementedError()
        else:
            self.base_continuation_prompt = None

    def create_input_sequence_for_generation(self, sample):
        if self.tokenizer.chat_template is not None:
            if isinstance(sample['text'][0], list) and isinstance(sample['text'][0][0], dict) and set(sample['text'][0][0].keys()) == {'role', 'content'}:
                messages = sample['text']
            else:
                if isinstance(sample['text'], str):
                    messages = to_message_format(sample['text'])
                elif isinstance(sample['text'], list):
                    messages = [to_message_format(text) for text in sample['text']]
            inputs = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                truncation=True,
                max_length=2048,
                padding='max_length',
                return_token_type_ids=False, 
                return_tensors='pt',
                return_dict=True,
                enable_thinking=False
            )
        else:
            if isinstance(sample['text'], str):
                texts = sample['text'] + self.base_continuation_prompt
            elif isinstance(sample['text'], list):
                texts = [text + self.base_continuation_prompt for text in sample['text']]
            inputs = self.tokenizer(texts, truncation=True, max_length=2048, padding='max_length', return_tensors='pt')
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        return inputs
    
    def create_input_sequence_for_training(self, sample):
        text = sample['text']
        if self.task in CLASSIFICATION_TASKS or (self.task in GENERATION_TASKS and self.generation_method == 'beam'):
            label = sample['labels'].strip()
        elif self.task in GENERATION_TASKS and self.generation_method == 'list':
            label = convert_list_to_quoted_str(sample['topic'])
        else:
            raise ValueError(f"Unknown task: {self.task}")

        if self.tokenizer.chat_template is not None:
            messages = to_message_format(text, label)
        
            inputs = self.tokenizer.apply_chat_template(
                messages,
                truncation=True,
                max_length=2048,
                padding='max_length',
                return_tensors="pt",
                return_dict=True
            )
            response_tokens = self.tokenizer.encode(label, add_special_tokens=False)
        else:
            texts = text + self.base_continuation_prompt + label
            inputs = self.tokenizer(
                texts,
                truncation=True,
                max_length=2048,
                padding='max_length',
                return_tensors='pt'
            )
            response_tokens = self.tokenizer.encode(f" {label}", add_special_tokens=False)
        
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        
        # Get the chat template format without the response
        # Find the assistant's response start
        
        response_start = None
        # Find where the assistant's response starts in the tokenized input
        for i in range(len(inputs['input_ids']) - len(response_tokens), 0, -1):
            if inputs['input_ids'][i:i+len(response_tokens)].tolist() == response_tokens:
                response_start = i
                break
        else:
            raise ValueError("Response not found in input")
        # Create labels tensor with -100s before the response
        labels = inputs['input_ids'].clone()
        labels[:response_start] = -100
        return {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "labels": labels
        }

STANCE_LABELS_2_ID = {
    "neutral": 0,
    "favor": 1,
    "against": 2
}

CLAIM_ENTAILMENT_2_WAY_LABELS_2_ID = {
    'supporting': 1,
    'other': 0
}

CLAIM_ENTAILMENT_3_WAY_LABELS_2_ID = {
    'supporting': 1,
    'refuting': 2,
    'neutral': 0
}

CLAIM_ENTAILMENT_4_WAY_LABELS_2_ID = {
    'supporting': 0,
    'refuting': 1,
    'discussing': 2,
    'irrelevant': 3,
}

CLAIM_ENTAILMENT_5_WAY_LABELS_2_ID = {
    'supporting': 0,
    'refuting': 1,
    'querying': 2,
    'irrelevant': 3,
    'discussing': 4
}

CLAIM_ENTAILMENT_7_WAY_LABELS_2_ID = {
    'supporting': 0,
    'refuting': 1,
    'leaning refuting': 2,
    'leaning supporting': 3,
    'querying': 4,
    'irrelevant': 5,
    'discussing': 6
}

@dataclass
class DataConfig:
    dataset_name: str

class DataProcessor:
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        self.model_config = model_config
        self.data_config = data_config
        
    def process_data(self, df: pd.DataFrame, classification_method: str, generation_method: str, train: bool = True, tokenize=True, truncate_beyond=None) -> datasets.Dataset:
        """Process dataframe into a format suitable for model input"""
        if self.model_config.task in CLASSIFICATION_TASKS:
            df = self._process_stance_classification(df, classification_method)
        elif self.model_config.task in GENERATION_TASKS:
            df = self._process_topic_extraction(df, generation_method)
        else:
            raise ValueError(f"Unknown task: {self.model_config.task}")
            
        dataset = datasets.Dataset.from_polars(df)
        dataset = self._add_prompts(dataset)
        if tokenize:
            dataset = self._tokenize_dataset(dataset, classification_method, train=train)
            columns = ['input_ids', 'attention_mask']
            if 'labels' in dataset.column_names:
                columns.append('labels')
            dataset.set_format(type='torch', columns=columns)
        if train:
            dataset.shuffle(seed=42)
        return dataset

    def get_loader(self, dataset: datasets.Dataset, loader_kwargs={}) -> torch.utils.data.DataLoader:
        if 'labels' in dataset.column_names:
            cols = ['input_ids', 'attention_mask', 'labels']
        else:
            cols = ['input_ids', 'attention_mask']
        return torch.utils.data.DataLoader(
            dataset.select_columns(cols),
            **loader_kwargs
        )
    
    def _process_stance_classification(self, df: pl.DataFrame, classification_method: str) -> pl.DataFrame:
        cols = ['text', 'topic']
        df = df.rename({
            'Target': 'topic',
            'Text': 'text',
            'Dataset': 'dataset',
            'Stance': 'class',
            'ParentTexts': 'parenttexts',
            'Context': 'context'
        }, strict=False)
        if 'dataset' in df.columns:
            cols.append('dataset')
        if 'class' in df.columns:
            if classification_method == 'head':
                df = df.with_columns(pl.col('class').replace_strict(self.model_config.labels2id))
            cols.append('class')
            
        if 'parenttexts' in df.columns:
            cols.append('parenttexts')
        if 'context' in df.columns:
            cols.append('context')
        return df.select(cols)
    
    def _process_topic_extraction(self, df: pl.DataFrame, generation_method: str) -> pl.DataFrame:
        if 'Text' in df.columns and 'text' not in df.columns:
            df = df.rename({"Text": "text"})
        if 'Target' in df.columns and 'topic' not in df.columns:
            df = df.rename({"Target": "topic"})
        if 'Dataset' in df.columns and 'dataset' not in df.columns:
            df = df.rename({'Dataset': 'dataset'})
        cols = ['text']
        if 'dataset' in df.columns:
            cols.append('dataset')
        if 'topic' in df.columns:
            cols.append('topic')
        return df.select(cols)

    def _add_prompts(self, dataset: datasets.Dataset) -> datasets.Dataset:
        prompt = self.model_config.prompt
        parent_prompt = self.model_config.parent_prompt
        context_prompt = self.model_config.context_prompt
        if self.model_config.task in CLASSIFICATION_TASKS:
            return dataset.map(
                lambda examples: {
                    "text": stance_examples_to_prompt(prompt, parent_prompt, context_prompt, examples)
                },
                batched=True
            )
        elif self.model_config.task in GENERATION_TASKS:
            return dataset.map(
                lambda examples: {
                    "text": stance_target_examples_to_prompt(prompt, examples)
                }, batched=True
            )
        else:
            raise ValueError("Task not found")
        
            
    def _tokenize_dataset(self, dataset: datasets.Dataset, classification_method: str, train: bool = True) -> datasets.Dataset:
        tokenizer = ChatTemplateTokenizer(self.model_config)
        if len(dataset) > 10000:
            num_proc = multiprocessing.cpu_count() // 2
        else:
            num_proc = 1
        if self.model_config.task in CLASSIFICATION_TASKS:
            if 'class' in dataset.column_names:
                dataset = dataset.rename_column("class", "labels")
            if train:
                if classification_method == 'head':
                    dataset = dataset.map(tokenizer.create_input_sequence_for_generation, batched=True, num_proc=num_proc)
                elif classification_method == 'generation':
                    dataset = dataset.map(tokenizer.create_input_sequence_for_training, num_proc=num_proc)
            else:
                dataset = dataset.map(tokenizer.create_input_sequence_for_generation, batched=True, num_proc=num_proc)

        elif self.model_config.task in GENERATION_TASKS:
            if train:
                dataset = dataset.map(tokenizer.create_input_sequence_for_training, num_proc=num_proc)
            else:
                dataset = dataset.map(tokenizer.create_input_sequence_for_generation, batched=len(dataset) > 1, num_proc=num_proc)
        else:
            raise ValueError(f"Unknown task: {self.model_config.task}")

        return dataset

def _evaluate_generation_set(predictions, references):
    import stancemining.metrics
    bertscore_f1, bertscore_p, bertscore_r = stancemining.metrics.bertscore_f1_targets(predictions, references)
    bleu_f1, bleu_p, bleu_r = stancemining.metrics.bleu_targets(predictions, references)
    max_claim_length = max(max(len(claim) for claim in preds) for preds in predictions)
    repetition_score = stancemining.metrics.max_ngram_repetition(predictions)
    return {
        'bertscore_f1': bertscore_f1,
        'bertscore_p': bertscore_p,
        'bertscore_r': bertscore_r,
        'bleu_f1': bleu_f1,
        'bleu_p': bleu_p,
        'bleu_r': bleu_r,
        'max_claim_length': max_claim_length,
        'max_ngram_repetition': repetition_score
    }

class ModelEvaluator:
    def __init__(self, task):
        self.task = task
        _, self.labels2id = get_labels_2_id(task)
        self.metrics = self._setup_metrics()
    
    def _setup_metrics(self) -> Dict[str, Any]:
        if self.task in CLASSIFICATION_TASKS:
            return {
                'accuracy': evaluate.load("accuracy"),
                'f1': evaluate.load("f1"),
                'precision': evaluate.load("precision"),
                'recall': evaluate.load("recall")
            }
        elif self.task in GENERATION_TASKS:
            return {
                'bertscore': evaluate.load("bertscore"),
                'bleu': evaluate.load("bleu")
            }
        raise ValueError(f"Unknown task: {self.model_config.task}")
    
    def evaluate(self, predictions: List[Any], references: List[Any], datasets: List[Any]) -> Dict[str, float]:
        if self.task in CLASSIFICATION_TASKS:
            assert self.labels2id is not None, "Must provide labels2id for classification evaluation"
            if isinstance(predictions[0], str):
                predictions = [self.labels2id[p] for p in predictions]
                references = [self.labels2id[r] for r in references]
            return self._evaluate_classification(predictions, references, datasets)
        elif self.task in GENERATION_TASKS:
            return self._evaluate_generation(predictions, references, datasets)
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _evaluate_classification(self, predictions: List[Any], references: List[Any], datasets: List[Any]) -> Dict[str, float]:
        pred_metrics = {
            'accuracy': self.metrics['accuracy'].compute(predictions=predictions, references=references)['accuracy'],
            'f1_macro': self.metrics['f1'].compute(predictions=predictions, references=references, average='macro')['f1'],
            'precision': self.metrics['precision'].compute(predictions=predictions, references=references, average='macro')['precision'],
            'recall': self.metrics['recall'].compute(predictions=predictions, references=references, average='macro')['recall']
        }

        pred_metrics['confusion_matrix'] = confusion_matrix(references, predictions)

        num_labels, labels2id = get_labels_2_id(self.task)
        id_2_label = {v: k for k, v in labels2id.items()}

        label_metrics = {
            'f1': self.metrics['f1'].compute(predictions=predictions, references=references, average=None)['f1'],
            'precision': self.metrics['precision'].compute(predictions=predictions, references=references, average=None)['precision'],
            'recall': self.metrics['recall'].compute(predictions=predictions, references=references, average=None)['recall']
        }
        for label, label_id in labels2id.items():
            if label_id in label_metrics['f1']:
                pred_metrics[label] = {
                    'f1': float(label_metrics['f1'][label_id]),
                    'precision': float(label_metrics['precision'][label_id]),
                    'recall': float(label_metrics['recall'][label_id])
                }

        df = pl.DataFrame({'prediction': predictions, 'reference': references, 'dataset': datasets})
        for key, dataset_df in df.partition_by('dataset', as_dict=True).items():
            dataset = key[0]
            d_predictions = dataset_df['prediction'].to_numpy()
            d_references = dataset_df['reference'].to_numpy()
            pred_metrics[dataset] = {
                'accuracy': self.metrics['accuracy'].compute(predictions=d_predictions, references=d_references)['accuracy'],
                'f1_macro': self.metrics['f1'].compute(predictions=d_predictions, references=d_references, average='macro')['f1'],
                'precision': self.metrics['precision'].compute(predictions=d_predictions, references=d_references, average='macro')['precision'],
                'recall': self.metrics['recall'].compute(predictions=d_predictions, references=d_references, average='macro')['recall']
            }

            label_metrics = {
                'f1': self.metrics['f1'].compute(predictions=d_predictions, references=d_references, average=None)['f1'],
                'precision': self.metrics['precision'].compute(predictions=d_predictions, references=d_references, average=None)['precision'],
                'recall': self.metrics['recall'].compute(predictions=d_predictions, references=d_references, average=None)['recall']
            }
            for label, label_id in labels2id.items():
                if label_id in label_metrics['f1']:
                    pred_metrics[dataset][label] = {
                        'f1': float(label_metrics['f1'][label_id]),
                        'precision': float(label_metrics['precision'][label_id]),
                        'recall': float(label_metrics['recall'][label_id])
                    }

        return pred_metrics

    def _evaluate_generation(self, predictions: List[List[str]], references: List[List[str]], datasets: List[str]) -> Dict[str, float]:
        pred_metrics = _evaluate_generation_set(predictions, references)

        df = pl.DataFrame({'prediction': predictions, 'reference': references, 'dataset': datasets})
        for key, dataset_df in df.partition_by('dataset', as_dict=True).items():
            dataset = key[0]
            d_predictions = dataset_df['prediction'].to_list()
            d_references = dataset_df['reference'].to_list()
    
            pred_metrics[dataset] = _evaluate_generation_set(d_predictions, d_references)
        return pred_metrics


@dataclass
class TrainingConfig:
    num_epochs: int
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_accum_steps: int = 8
    batch_size: int = 1
    warmup_steps: int = 500
    neftune_noise_alpha: float = 5

class ModelTrainer:
    def __init__(
        self, 
        model_config: ModelConfig,
        training_config: TrainingConfig
    ):
        self.model_config = model_config
        self.training_config = training_config
        
        if self.model_config.torch_dtype == torch.bfloat16:
            mixed_precision = 'bf16'
        elif self.model_config.torch_dtype == torch.float16:
            mixed_precision = 'fp16'
        else:
            mixed_precision = 'no'
        
        self.accelerator = accelerate.Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=training_config.grad_accum_steps
        )

    def set_model_and_tokenizer(self, model, tokenizer) -> None:
        """Set model and tokenizer"""
        self.model_config.model = model
        self.model_config.tokenizer = tokenizer

    def prepare_for_training(self) -> None:
        """Prepare model for training with LoRA"""
        if self.training_config.grad_accum_steps > 1:
            self.model_config.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        if (self.model_config.task in CLASSIFICATION_TASKS and self.model_config.classification_method == 'head' and self.training_config.grad_accum_steps > 1) \
            or (self.model_config.quantization is not None):
            self.model_config.model = peft.prepare_model_for_kbit_training(self.model_config.model)
        
        # Setup LoRA
        modules = self._find_all_linear_names()
        lora_kwargs = {}
        if self.model_config.task in CLASSIFICATION_TASKS:
            if self.model_config.classification_method == 'head':
                lora_kwargs['task_type'] = "SEQ_CLS"
                lora_kwargs['modules_to_save'] = ['score']
            elif self.model_config.classification_method == 'generation':
                lora_kwargs['task_type'] = "CAUSAL_LM"
        elif self.model_config.task in GENERATION_TASKS:
            lora_kwargs['task_type'] = "CAUSAL_LM"
        else:
            raise ValueError(f"Unknown task: {self.model_config.task}")

        if self.model_config.quantization is None:
            lora_config = peft.LoraConfig(
                r=self.model_config.lora_config.r,
                lora_alpha=self.model_config.lora_config.lora_alpha,
                target_modules=modules,
                lora_dropout=self.model_config.lora_config.lora_dropout,
                bias="none",
                **lora_kwargs
            )
        elif self.model_config.quantization is not None:
            lora_config = peft.LoraConfig(
                r=self.model_config.lora_config.r,
                lora_alpha=self.model_config.lora_config.lora_alpha,
                target_modules=modules,
                lora_dropout=self.model_config.lora_config.lora_dropout,
                bias="none",
                **lora_kwargs
            )
        
        self.model_config.model = peft.get_peft_model(self.model_config.model, lora_config)

    def _find_all_linear_names(self) -> list:
        """Find all linear layer names in model"""
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in self.model_config.model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def train(
        self, 
        train_dataset, 
        eval_dataset, 
        model_save_path: str,
        evaluator: ModelEvaluator
    ) -> None:
        """Train the model"""
        # Setup training components
        optimizer = torch.optim.AdamW(
            self.model_config.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        num_steps = self.training_config.num_epochs * len(train_dataset) // (self.training_config.batch_size * self.training_config.grad_accum_steps)
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, int(0.05 * num_steps), num_steps)
        
        # Prepare dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset.select_columns(['input_ids', 'attention_mask', 'labels']),
            batch_size=self.training_config.batch_size,
            shuffle=True,
            # pin_memory=True,
            # pin_memory_device=self.model_config.model.device
        )
        eval_cols = ['input_ids', 'attention_mask']
        if 'labels' in eval_dataset.column_names:
            eval_cols.append('labels')
        elif 'class' in eval_dataset.column_names:
            eval_cols.append('labels')
            eval_dataset = eval_dataset.rename_column('class', 'labels')
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset.select_columns(eval_cols),
            batch_size=self.training_config.batch_size,
            # pin_memory=True,
            # pin_memory_device=self.model_config.model.device
        )
        
        # Prepare training components
        if self.model_config.quantization is None:
            (
                self.model_config.model,
                optimizer,
                train_loader,
                eval_loader,
                scheduler
            ) = self.accelerator.prepare(
                self.model_config.model,
                optimizer,
                train_loader,
                eval_loader,
                scheduler,
                device_placement=[False] * 5
            )
        
        self._training_loop(
            train_loader,
            eval_loader,
            train_dataset,
            eval_dataset,
            optimizer,
            scheduler,
            evaluator,
            model_save_path
        )

    def _training_loop(
        self,
        train_loader,
        eval_loader,
        train_dataset,
        eval_dataset,
        optimizer,
        scheduler,
        evaluator,
        model_save_path
    ):
        """Main training loop"""
        best_eval_metric = -float('inf')

        chosen_metric = 'f1_macro' if self.model_config.task in CLASSIFICATION_TASKS else 'bertscore_f1'

        if self.model_config.task in CLASSIFICATION_TASKS and self.model_config.classification_method == 'head':
            batch_keys = ['input_ids', 'attention_mask']
            train_labels = np.array(train_dataset['labels'])
            class_wts = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
            loss_func = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(
                    class_wts, 
                    dtype=torch.float32,
                    device=self.model_config.model.device)
            )
        else:
            batch_keys = ['input_ids', 'attention_mask', 'labels']
            loss_func = None
        
        self.model_config.model, neftune_hook = activate_neftune(
            self.model_config.model,
            self.accelerator,
            self.training_config.neftune_noise_alpha
        )
        for epoch in range(self.training_config.num_epochs):
            print(f"Epoch {epoch + 1}/{self.training_config.num_epochs}")
            self._train_step(
                train_loader,
                optimizer,
                scheduler,
                loss_func,
                batch_keys
            )
                    
            # Evaluation
            deactivate_neftune(
                self.model_config.model,
                self.accelerator,
                neftune_hook
            )
            metrics = self._validation_step(eval_loader, eval_dataset, evaluator, loss_func)
            state_str = ""
            if self.model_config.task in CLASSIFICATION_TASKS:
                report_keys = ['f1_macro', 'precision', 'recall', 'loss']
            elif self.model_config.task in GENERATION_TASKS:
                report_keys = ['bertscore_f1', 'bleu_f1']
            else:
                raise ValueError("Task not found")
            for key in report_keys:
                if key in metrics:
                    val = metrics[key]
                    state_str += f" {key.title()}: {val:.4f},"
            print(state_str)
            eval_metrics = {f"eval/{key}": val for key, val in metrics.items()}
            wandb.log(eval_metrics)
            
            # Save best model
            if metrics[chosen_metric] > best_eval_metric:
                best_eval_metric = metrics[chosen_metric]
                self.model_config.model.save_pretrained(model_save_path)
                self.model_config.tokenizer.save_pretrained(model_save_path)
                
            self.model_config.model, neftune_hook = activate_neftune(
                self.model_config.model,
                self.accelerator,
                self.training_config.neftune_noise_alpha
            )

    def _train_step(self, train_loader, optimizer, scheduler, loss_func, batch_keys):
        self.model_config.model.train()
        loss = float('inf')
        pbar = tqdm.tqdm(total=len(train_loader), desc=f"Training round, loss: {loss:.4f}")
        for step, batch in enumerate(train_loader):
            model_batch = {k: batch[k].to(self.model_config.model.device) for k in batch_keys}
            outputs = self.model_config.model(**model_batch)
            if loss_func is not None:
                labels = batch['labels'].to(self.model_config.model.device)
                loss = loss_func(outputs.logits, labels) / self.training_config.grad_accum_steps
            else:
                loss = outputs.loss / self.training_config.grad_accum_steps
            self.accelerator.backward(loss)

            wandb.log({"train/loss": loss.item()})
            pbar.set_description(f"Training round, loss: {loss.item():.4e}")
            pbar.update(1)
            
            if (step + 1) % self.training_config.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

    def _validation_step(self, eval_loader, eval_dataset, evaluator: ModelEvaluator, loss_func=None):
        """Run validation step"""
        self.model_config.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm.tqdm(total=len(eval_loader), desc="Validation round")
        with torch.no_grad():
            for batch in eval_loader:
                pbar.update(1)

                # For head classification, do single forward pass for both loss and predictions
                if self.model_config.task in CLASSIFICATION_TASKS \
                        and self.model_config.classification_method == 'head':
                    inputs = {k: v.to(self.model_config.model.device) for k, v in batch.items()}
                    outputs = self.model_config.model(**inputs)

                    # Compute loss if loss_func and labels are available
                    if loss_func is not None and 'labels' in batch:
                        labels = batch['labels'].to(self.model_config.model.device)
                        loss = loss_func(outputs.logits, labels)
                        total_loss += loss.item()
                        num_batches += 1

                    # Extract predictions from same forward pass
                    predicted_class = torch.argmax(outputs.logits, dim=1)
                    preds = predicted_class.cpu().tolist()
                else:
                    # For generation methods, use existing get_prediction
                    preds = get_prediction(
                        batch,
                        self.model_config.task,
                        self.model_config.model,
                        self.model_config.tokenizer,
                        self.model_config.classification_method,
                        self.model_config.generation_method
                    )

                all_preds.extend(preds)

                # delete batch to reduce memory usage
                del batch
        pbar.close()

        if self.model_config.task in CLASSIFICATION_TASKS:
            eval_df = eval_dataset.to_polars()
            label_col = 'labels' if 'labels' in eval_df.columns else 'class'
            all_labels = eval_df[label_col].to_list()
        elif self.model_config.task in GENERATION_TASKS:
            all_labels = eval_dataset.to_polars()['topic'].to_list()
        else:
            raise ValueError("Task not found")
        datasets = eval_dataset.to_polars()['dataset'].to_list()

        metrics = evaluator.evaluate(all_preds, all_labels, datasets)

        # Add validation loss to metrics if computed
        if num_batches > 0:
            metrics['loss'] = total_loss / num_batches

        # clear memory
        torch.cuda.empty_cache()
        gc.collect()

        return metrics

def setup_model_and_tokenizer(model_config: ModelConfig, model_kwargs={}, model_save_path=None, model_name=None, full_saved_model=False):
    """Initialize model and tokenizer based on config"""
    model_path = model_save_path if model_save_path else model_name
    tokenizer_kwargs = {}
    if 'hf_token' in model_kwargs:
        tokenizer_kwargs['hf_token'] = model_kwargs['hf_token']
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, 
        **tokenizer_kwargs
    )

    if model_config.task in CLASSIFICATION_TASKS:
        if model_config.classification_method == 'head':
            if model_save_path and not full_saved_model:
                create_fn = peft.AutoPeftModelForSequenceClassification.from_pretrained
            else:
                create_fn = transformers.AutoModelForSequenceClassification.from_pretrained
            model = create_fn(
                model_path,
                num_labels=model_config.num_labels,
                **model_kwargs
            )
        elif model_config.classification_method == 'generation':
            if model_save_path:
                create_fn = peft.AutoPeftModelForCausalLM.from_pretrained
            else:
                create_fn = transformers.AutoModelForCausalLM.from_pretrained
            model = create_fn(
                model_path,
                **model_kwargs
            )
        else:
            raise ValueError("Classification method not found")
        
        # set labels2id and id2labels
        model.config.update({
            'label2id': model_config.labels2id,
            'id2label': {v: k for k, v in model_config.labels2id.items()}
        })
        if model_config.classification_method == 'head':
            model.config.pad_token_id = tokenizer.pad_token_id
        elif model_config.classification_method == 'generation':
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    elif model_config.task in GENERATION_TASKS:
        if model_save_path:
            create_fn = peft.AutoPeftModelForCausalLM.from_pretrained
        else:
            create_fn = transformers.AutoModelForCausalLM.from_pretrained
        model = create_fn(
            model_path,
            **model_kwargs
        )

        model.generation_config.pad_token_id = tokenizer.pad_token_id
    else:
        raise ValueError("Task not found")
    
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def get_prediction(inputs, task, model, tokenizer, classification_method, generation_method, generate_kwargs={}):
    """Get model predictions"""
    if task in CLASSIFICATION_TASKS:
        if classification_method == 'head':
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            output = model(**inputs)
            predicted_class = torch.argmax(output.logits, dim=1)
            return predicted_class.cpu().tolist()
        elif classification_method == 'generation':
            if 'labels' in inputs:
                # does not work with batch size > 1
                assert inputs['input_ids'].shape[0] == 1, "Batch size must be 1"
                prompt = {
                    "input_ids": inputs["input_ids"][inputs['labels'] == -100].unsqueeze(0),
                    "attention_mask": inputs["attention_mask"][inputs['labels'] == -100].unsqueeze(0),
                }
            else:
                prompt = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
            prompt = {k: v.to(model.device) for k, v in prompt.items()}
            if 'max_new_tokens' not in generate_kwargs:
                generate_kwargs['max_new_tokens'] = 1
            outputs = model.generate(**prompt, **generate_kwargs)
            completions = [tokenizer.decode(
                output[prompt['input_ids'].shape[1]:],
                skip_special_tokens=True
            ) for output in outputs]
            def get_label(c):
                for label in ['neutral', 'in favor', 'against']:
                    if label in c.lower():
                        return label
                else:
                    if c.lower() in ['f', 'in'] and generate_kwargs['max_new_tokens'] == 1:
                        return 'favor'
                    return 'neutral'
            return [get_label(c) for c in completions]
    elif task in GENERATION_TASKS:
        if 'labels' in inputs:
            prompt = {
                "input_ids": inputs["input_ids"][inputs['labels'] == -100].unsqueeze(0),
                "attention_mask": inputs["attention_mask"][inputs['labels'] == -100].unsqueeze(0),
            }
        else:
            prompt = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
        prompt = {k: v.to(model.device) for k, v in prompt.items()}
        if 'max_new_tokens' not in generate_kwargs:
            generate_kwargs['max_new_tokens'] = 40
        if 'stop_strings' not in generate_kwargs:
            generate_kwargs['stop_strings'] = ['\n', '<|endoftext|>', '<|im_end|>']
            generate_kwargs['tokenizer'] = tokenizer
        outputs = model.generate(**prompt, **generate_kwargs)
        completions = [tokenizer.decode(
            output[prompt['input_ids'].shape[1]:],
            skip_special_tokens=True
        ) for output in outputs]
        if generation_method == 'list':
            return parse_list_completions(completions)
        elif generation_method == 'beam':
            batched_completions = []
            num_return_sequences = generate_kwargs.get('num_return_sequences', 3)
            for i in range(prompt['input_ids'].shape[0]):
                batched_completions.append(completions[i*num_return_sequences:(i+1)*num_return_sequences])
            return batched_completions
        else:
            raise NotImplementedError("Generation method not found.")
    else:
        raise ValueError("Task not found")
            

def get_predictions(task, df, config, model_kwargs={}, generate_kwargs={}):
    
    output_type = config['classification_method'] if task in CLASSIFICATION_TASKS else config['generation_method']
    if 'hf_model' in config:
        model_save_path = config['hf_model']
        file_path = huggingface_hub.hf_hub_download(repo_id=model_save_path, filename='metadata.json')
        with open(file_path, 'r') as f:
            metadata = json.load(f)
        prompt = metadata['prompt']
        parent_prompt = metadata['parent_prompt'] if 'parent_prompt' in metadata else None
    else:
        model_save_path = config['model_path']
        prompt = load_prompt(task, config['prompting_method'], generation_method=config['generation_method'] if 'generation_method' in config else None)
        parent_prompt = load_parent_prompt(task, prompting_method=config['prompting_method'])

    if task in ['stance-classification', 'claim-entailment-3-way']:
        num_labels = 3
    elif task == 'argument-classification':
        num_labels = 2
    elif task in ['claim-entailment-7-way']:
        num_labels = 7
    else:
        num_labels = None

    # Setup configurations
    model_config = ModelConfig(
        model_name=None,
        task=task,
        num_labels=num_labels,
        device_map=model_kwargs['device_map'],
        prompt=prompt,
        parent_prompt=parent_prompt,
        classification_method=config['classification_method'] if task in CLASSIFICATION_TASKS else None,
        generation_method=config['generation_method'] if task in GENERATION_TASKS else None,
        output_type=output_type,
    )
    
    data_config = DataConfig(
        dataset_name=None
    )
    
    # Initialize components
    model, tokenizer = setup_model_and_tokenizer(model_config.task, model_config.classification_method, model_config.num_labels, model_kwargs=model_kwargs, model_save_path=model_save_path)
    model_config.model, model_config.tokenizer = model, tokenizer
    processor = DataProcessor(model_config, data_config)
    test_dataset = processor.process_data(df, model_config.classification_method, model_config.generation_method, train=False)
    
    # optimize for inference
    # model.generation_config.cache_implementation = 'static'
    # model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=True)

    if model_config.generation_method == 'beam':
        num_samples = generate_kwargs.get('num_return_sequences', 3)
        if num_samples > 1:
            generate_kwargs['num_beams'] = num_samples * 2
            generate_kwargs['num_beam_groups'] = num_samples
            generate_kwargs['diversity_penalty'] = 10.0
            generate_kwargs['no_repeat_ngram_size'] = 2
            generate_kwargs['do_sample'] = False
            generate_kwargs['trust_remote_code'] = True

    predictions = []
    test_loader = processor.get_loader(test_dataset, loader_kwargs={"batch_size": config.get('batch_size', 1)})
    for inputs in tqdm.tqdm(test_loader, desc="Evaluating"):
        predictions.extend(get_prediction(
            inputs, 
            task, 
            model, 
            tokenizer, 
            model_config.classification_method,
            model_config.generation_method,
            generate_kwargs=generate_kwargs
        ))

    if task in CLASSIFICATION_TASKS:
        if model_config.classification_method == 'head':
            id2labels = {v: k for k, v in model_config.labels2id.items()}
            predictions = [id2labels[p] for p in predictions]

    return predictions

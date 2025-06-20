import json
import os

import huggingface_hub
import numpy as np
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from stancemining.finetune import (
    DataConfig, 
    ModelConfig, 
    DataProcessor, 
    get_model_save_path, 
    load_prompt, 
    load_parent_prompt,
    parse_list_completions,
    to_message_format
)

class BaseLLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, prompt, max_new_tokens=100, num_samples=3):
        raise NotImplementedError
    
    
class Transformers(BaseLLM):
    def __init__(self, model_name, model_kwargs={}, tokenizer_kwargs={}, lazy=False):
        super().__init__(model_name)
        
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs

        self.lazy = lazy
        if not lazy:
            self.load_model()
        else:
            self.model = None
            self.tokenizer = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **self.model_kwargs
        )
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
    
    def generate(self, prompts, max_new_tokens=100, num_samples=3, add_generation_prompt=True, continue_final_message=False):
        conversations = []
        for prompt in prompts:
            if isinstance(prompt, str):
                conversation = [
                    {'role': 'user', 'content': prompt}
                ]
            elif isinstance(prompt, list):
                conversation = []
                conversation.append({'role': 'system', 'content': prompt[0]})
                role = 'user'
                for i, p in enumerate(prompt[1:]):
                    conversation.append({'role': role, 'content': p})
                    role = 'assistant' if role == 'user' else 'user'
            else:
                raise ValueError('Prompt must be a string or list of strings')
            conversations.append(conversation)
        
        if self.lazy:
            self.load_model()
        all_outputs = []
        if len(conversations) == 1:
            iterator = conversations
        else:
            iterator = tqdm.tqdm(conversations)
        for conversation in iterator:
            inputs = self.tokenizer.apply_chat_template(conversation, return_dict=True, return_tensors='pt', add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generate_kwargs = {}
            if num_samples > 1:
                generate_kwargs['num_beams'] = num_samples * 5
                generate_kwargs['num_return_sequences'] = num_samples
                generate_kwargs['num_beam_groups'] = num_samples
                generate_kwargs['diversity_penalty'] = 0.5
                generate_kwargs['no_repeat_ngram_size'] = 2
                generate_kwargs['do_sample'] = False

            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **generate_kwargs)
            outputs = [self.tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True) for output in outputs]
            all_outputs.append(outputs)
        if self.lazy:
            self.unload_model()
        return all_outputs

    def unload_model(self):
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()

class VLLM(BaseLLM):
    def __init__(self, model_name, lazy=False, verbose=False):
        super().__init__(model_name)
        
        self.model_name = model_name
        self.verbose = verbose

        self.lazy = lazy
        if not lazy:
            self.load_model()
        else:
            self.model = None

    def load_model(self):
        import vllm
        self.model = vllm.LLM(model=self.model_name)
        self.sampling_params = vllm.SamplingParams(temperature=0.0)
    
    def generate(self, prompts, max_new_tokens=100, num_samples=3, add_generation_prompt=True, continue_final_message=False):
        conversations = []
        for prompt in prompts:
            if isinstance(prompt, str):
                conversation = [
                    {'role': 'user', 'content': prompt}
                ]
            elif isinstance(prompt, list):
                conversation = []
                conversation.append({'role': 'system', 'content': prompt[0]})
                role = 'user'
                for i, p in enumerate(prompt[1:]):
                    conversation.append({'role': role, 'content': p})
                    role = 'assistant' if role == 'user' else 'user'
            else:
                raise ValueError('Prompt must be a string or list of strings')
            conversations.append(conversation)
        
        if self.lazy:
            self.load_model()

        self.sampling_params.max_tokens = max_new_tokens
        self.sampling_params.n = num_samples

        outputs = self.model.chat(
            messages=conversations, 
            sampling_params=self.sampling_params, 
            use_tqdm=self.verbose, 
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message
        )
        all_outputs = [o.outputs[0].text for o in outputs]
        if self.lazy:
            self.unload_model()
        return all_outputs

    def unload_model(self):
        self.model = None
        torch.cuda.empty_cache()


def get_vllm_predictions(task, df, config, verbose=False):
    import vllm
    import vllm.lora.request
    
    output_type = config['classification_method'] if task == "stance-classification" else config['generation_method']
    if 'hf_model' in config:
        model_save_path = config['hf_model']
        file_path = huggingface_hub.hf_hub_download(repo_id=model_save_path, filename='metadata.json')
        with open(file_path, 'r') as f:
            metadata = json.load(f)
        prompt = metadata['prompt']
        parent_prompt = metadata['parent_prompt'] if 'parent_prompt' in metadata else None
    else:
        model_save_path = get_model_save_path(task, config['save_model_path'], config['model_name'], config['data_name'], output_type)
        prompt=load_prompt(task, config['prompting_method'], generation_method=config['generation_method'] if 'generation_method' in config else None),
        parent_prompt=load_parent_prompt(task, prompting_method=config['prompting_method'])
    
    # Setup configurations
    model_config = ModelConfig(
        model_name=None,
        task=task,
        num_labels=2 if task == "argument-classification" else 3,
        prompt=prompt,
        parent_prompt=parent_prompt,
        classification_method=config['classification_method'] if task == 'stance-classification' else None,
        generation_method=config['generation_method'] if task == 'topic-extraction' else None,
    )
    
    data_config = DataConfig(
        dataset_name=None
    )
    
    # Initialize components
    processor = DataProcessor(model_config, data_config)
    test_dataset = processor.process_data(df, model_config.classification_method, model_config.generation_method, train=False, tokenize=False, truncate_beyond=10000)
    prompts = test_dataset['text']
    prompts = [to_message_format(p) for p in prompts]

    if model_config.generation_method == 'beam':
        raise NotImplementedError()

    llm_kwargs = {}
    if task == 'topic-extraction' or (task == 'stance-classification' and model_config.classification_method == 'generation'):
        llm_kwargs['task'] = 'generate'
        llm_kwargs['generation_config'] = 'auto'

        file_path = huggingface_hub.hf_hub_download(repo_id=model_save_path, filename='adapter_config.json')
        with open(file_path, 'r') as f:
            adapter_config = json.load(f)
        adapter_path = huggingface_hub.snapshot_download(repo_id=model_save_path)
        model_name = adapter_config['base_model_name_or_path']

    elif task == 'stance-classification' and model_config.classification_method == 'head':
        llm_kwargs['task'] = 'classify'
        llm_kwargs['enforce_eager'] = True
        model_name = config['hf_model']
    else:
        raise ValueError()
    
    file_path = huggingface_hub.hf_hub_download(repo_id=model_save_path, filename='tokenizer_config.json')
    with open(file_path, 'r') as f:
        tokenizer_config = json.load(f)
    if 'chat_template' in tokenizer_config and task == 'stance-classification' and model_config.classification_method == 'head':
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        prompts = tokenizer.apply_chat_template(
            prompts, 
            add_generation_prompt=True,
            truncation=True,
            max_length=2048,
            padding='max_length',
            return_token_type_ids=False, 
            enable_thinking=False,
            tokenize=False
        )

    try:
        llm = vllm.LLM(
            model=model_name,
            enable_lora=True,
            **llm_kwargs
        )
    except NotImplementedError as ex:
        # this sometimes works without the env var, not sure why
        if str(ex) == 'VLLM_USE_V1=1 is not supported with --task classify.':
            os.environ['VLLM_USE_V1'] = '0'
            llm = vllm.LLM(
                model=model_name,
                enable_lora=True,
                **llm_kwargs
            )
        else:
            raise

    # greedy decoding
    sampling_params = vllm.SamplingParams(temperature=0.0)
    if task == 'topic-extraction':
        lora_request = vllm.lora.request.LoRARequest(
            f"{task}_adapter",
            1,
            adapter_path
        )

    if task == 'topic-extraction' or (task == 'stance-classification' and model_config.classification_method == 'generation'):
        outputs = llm.chat(messages=prompts, sampling_params=sampling_params, use_tqdm=verbose, lora_request=lora_request)
        predictions = [o.outputs[0].text for o in outputs]
        predictions = parse_list_completions(predictions)
    elif task == 'stance-classification' and model_config.classification_method == 'head':
        outputs = llm.classify(prompts, use_tqdm=verbose)
        probs = [o.outputs.probs for o in outputs]
        predictions = [np.argmax(p) for p in probs]
        id2labels = {v: k for k, v in data_config.labels2id.items()}
        predictions = [id2labels[p] for p in predictions]
    else:
        raise ValueError()

    return predictions
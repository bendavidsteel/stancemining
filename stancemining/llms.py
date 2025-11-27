import json
import logging
import os
import re

import huggingface_hub
import numpy as np
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from stancemining.finetune import (
    CLASSIFICATION_TASKS,
    GENERATION_TASKS,
    DataConfig, 
    ModelConfig, 
    DataProcessor, 
    get_model_save_path, 
    load_prompt, 
    load_parent_prompt,
    parse_list_completions,
    to_message_format
)

logger = logging.getLogger(__name__)

def prompts_to_conversations(prompts, system_prompt_allowed=True):
    conversations = []
    for prompt in prompts:
        if isinstance(prompt, str):
            conversation = [
                {'role': 'user', 'content': prompt}
            ]
        elif isinstance(prompt, list):
            conversation = []
            if system_prompt_allowed:
                conversation.append({'role': 'system', 'content': prompt[0]})
            role = 'user'
            for p in prompt[1:]:
                conversation.append({'role': role, 'content': p})
                role = 'assistant' if role == 'user' else 'user'
        else:
            raise ValueError('Prompt must be a string or list of strings')
        conversations.append(conversation)
    return conversations

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

        self.load_model()

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
        conversations = prompts_to_conversations(prompts)
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
        
        return all_outputs

    def unload_model(self):
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()

def load_vllm_model(model_name, model_kwargs, sampling_param_kwargs):
    import vllm
    model = None
    while model is None:
        try:
            model = vllm.LLM(
                model=model_name,
                **model_kwargs
            )
        except (NotImplementedError, ValueError) as ex:
            # this sometimes works without the env var, not sure why
            if str(ex) == 'VLLM_USE_V1=1 is not supported with --task classify.':
                logger.warning("Disabling VLLM_USE_V1 for compatibility")
                os.environ['VLLM_USE_V1'] = '0'
            elif 'Prefix caching is not supported' in str(ex):
                logger.warning("Disabling prefix caching due to model incompatibility")
                model_kwargs['enable_prefix_caching'] = False
            elif 'Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine' in str(ex):
                # parse max seq length from error message
                logger.warning("Adjusting max_model_len due to GPU memory constraints")
                max_seq_len = re.search(r'max seq len \((\d+)\)', str(ex))
                max_kv_cache_tokens = re.search(r'that can be stored in KV cache \((\d+)\)', str(ex))
                if max_seq_len and max_kv_cache_tokens:
                    max_seq_len = int(max_seq_len.group(1))
                    max_kv_cache_tokens = int(max_kv_cache_tokens.group(1))
                    new_max_seq_len = min(max_seq_len, max_kv_cache_tokens)
                    model_kwargs['max_model_len'] = new_max_seq_len
                else:
                    raise
            else:
                raise
    sampling_params = vllm.SamplingParams(**sampling_param_kwargs)
    return model, sampling_params

class VLLM(BaseLLM):
    def __init__(self, model_name, model_kwargs, verbose=False):
        super().__init__(model_name)
        self.model_kwargs = model_kwargs
        self.verbose = verbose

        sampling_param_kwargs = {
            'temperature': 0.0,
            'stop': ['\n', '<|endoftext|>', '<|im_end|>']
        }

        self.model, self.sampling_params = load_vllm_model(model_name, model_kwargs, sampling_param_kwargs)

    def generate(self, prompts, max_new_tokens=100, num_samples=3, add_generation_prompt=True, continue_final_message=False):
        conversations = prompts_to_conversations(prompts)
        
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
        
        return all_outputs

    def unload_model(self):
        self.model = None
        torch.cuda.empty_cache()

class Anthropic(BaseLLM):
    def __init__(self, model_name, model_kwargs):
        super().__init__(model_name)
        import anthropic
        assert 'api_key' in model_kwargs, "Anthropic API key must be provided in model_kwargs"
        self.client = anthropic.Anthropic(api_key=model_kwargs['api_key'])

    def generate(self, prompts, max_new_tokens=100, num_samples=3, add_generation_prompt=True, continue_final_message=False):
        conversations = prompts_to_conversations(prompts, system_prompt_allowed=False)

        system_prompts = [p[0] for p in prompts]
        assert len(set(system_prompts)) == 1, "All system prompts must be the same for Anthropic"
        system_prompt = system_prompts[0]

        all_outputs = []
        if len(conversations) == 1:
            iterator = conversations
        else:
            iterator = tqdm.tqdm(conversations)
        for conversation in iterator:
            if continue_final_message:
                conversation[-1]['content'] = conversation[-1]['content'].rstrip()
            message = self.client.messages.create(
                max_tokens=max_new_tokens,
                messages=conversation,
                model=self.model_name,
                system=system_prompt
            )
            all_outputs.append([c.text.strip() for c in message.content])

        return all_outputs

def get_vllm_predictions(task, df, config, verbose=False, model_kwargs={}, generate_kwargs={}):
    import vllm
    import vllm.lora.request
    
    output_type = config['classification_method'] if task in CLASSIFICATION_TASKS else config['generation_method']
    if 'hf_model' in config:
        model_save_path = config['hf_model']
        file_path = huggingface_hub.hf_hub_download(repo_id=model_save_path, filename='metadata.json')
        with open(file_path, 'r') as f:
            metadata = json.load(f)
        prompt = metadata['prompt']
        parent_prompt = metadata['parent_prompt'] if 'parent_prompt' in metadata else None
    else:
        if 'data_name' in config:
            model_save_path = get_model_save_path(task, config['save_model_path'], config['model_name'], config['data_name'], output_type)
        else:
            model_save_path = config['model_path']
        prompt=load_prompt(task, config['prompting_method'], generation_method=config['generation_method'] if 'generation_method' in config else None)
        parent_prompt=load_parent_prompt(task, prompting_method=config['prompting_method'])
    
    # Setup configurations
    model_config = ModelConfig(
        model_name=None,
        task=task,
        prompt=prompt,
        parent_prompt=parent_prompt,
        classification_method=config['classification_method'] if task in CLASSIFICATION_TASKS else None,
        generation_method=config['generation_method'] if task in GENERATION_TASKS else None,
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
        raise NotImplementedError("Beam search is not supported with VLLM yet.")

    chat_template_kwargs = {}
    if task in GENERATION_TASKS or (task in CLASSIFICATION_TASKS and model_config.classification_method == 'generation'):
        model_kwargs['task'] = 'generate'
        model_kwargs['generation_config'] = 'auto'
        model_kwargs['enable_lora'] = True
        chat_template_kwargs['enable_thinking'] = False

        if 'hf_model' in config:
            file_path = huggingface_hub.hf_hub_download(repo_id=model_save_path, filename='adapter_config.json')
            with open(file_path, 'r') as f:
                adapter_config = json.load(f)
            adapter_path = huggingface_hub.snapshot_download(repo_id=model_save_path)
            model_name = adapter_config['base_model_name_or_path']
        else:
            adapter_path = model_save_path
            model_name = config['base_model_name']

    elif task in CLASSIFICATION_TASKS and model_config.classification_method == 'head':
        model_kwargs['task'] = 'classify'
        # model_kwargs['enforce_eager'] = True
        model_name = model_save_path
    else:
        raise ValueError()
    
    if 'hf_model' in config:
        tokenizer_file_path = huggingface_hub.hf_hub_download(repo_id=model_save_path, filename='tokenizer_config.json')
    else:
        tokenizer_file_path = os.path.join(model_save_path, 'tokenizer_config.json')
    with open(tokenizer_file_path, 'r') as f:
        tokenizer_config = json.load(f)
    if task in CLASSIFICATION_TASKS and model_config.classification_method == 'head':
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

    # turn off verbose logging
    os.environ['VLLM_CONFIGURE_LOGGING'] = '0'

    model_kwargs['enable_prefix_caching'] = True

    if task in CLASSIFICATION_TASKS and model_config.classification_method == 'generation':
        max_new_tokens = 1
    elif task == 'topic-extraction':
        max_new_tokens = 30
    elif task == 'claim-extraction':
        max_new_tokens = 200
    else:
        max_new_tokens = None

    # greedy decoding
    sampling_param_kwargs = {
        'temperature': 0.0,
        'stop': ['\n', '<|endoftext|>', '<|im_end|>'] if task in GENERATION_TASKS else None,
        'max_tokens': max_new_tokens,
        'repetition_penalty': 1.2
    }

    llm, sampling_params = load_vllm_model(model_name, model_kwargs, sampling_param_kwargs)

    if task in GENERATION_TASKS:
        lora_request = vllm.lora.request.LoRARequest(
            f"{task}_adapter",
            1,
            adapter_path
        )

    if task in GENERATION_TASKS or (task in CLASSIFICATION_TASKS and model_config.classification_method == 'generation'):
        outputs = llm.chat(messages=prompts, sampling_params=sampling_params, use_tqdm=verbose, lora_request=lora_request, chat_template_kwargs=chat_template_kwargs, **generate_kwargs)
        predictions = [o.outputs[0].text for o in outputs]
        predictions = parse_list_completions(predictions)
    elif task in CLASSIFICATION_TASKS and model_config.classification_method == 'head':
        outputs = llm.classify(prompts, use_tqdm=verbose, **generate_kwargs)
        probs = [o.outputs.probs for o in outputs]
        predictions = [np.argmax(p) for p in probs]
        id2labels = {v: k for k, v in model_config.labels2id.items()}
        predictions = [id2labels[p] for p in predictions]
    else:
        raise ValueError()

    return predictions
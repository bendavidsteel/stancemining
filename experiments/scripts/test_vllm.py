import datetime
import json
import os

import huggingface_hub
import numpy as np
import polars as pl
import torch
import vllm
import vllm.lora.request

from stancemining import finetune, llms

def print_results(task, method, results, references, datasets, start_time, end_time):
    total_secs = (end_time - start_time).total_seconds()

    labels2id = finetune.LABELS_2_ID
    evaluator = finetune.ModelEvaluator(task, labels2id)

    metrics = evaluator.evaluate(
        results,
        references,
        datasets
    )
    print(f"Method: {method}")
    print(f"Total time: {total_secs}")
    if task == 'topic-extraction':
        print(f"Metrics: bertscore f1: {metrics['bertscore_f1']}, bleu f1: {metrics['bleu_f1']}")
    elif task == 'stance-classification':
        print(f"Metrics: F1: {metrics['f1_macro']}")

def main():
    torch.cuda.init()

    task = "stance-classification"

    dataset_name = ['vast', 'ezstance', 'ctsdt']
    generation_method = 'list'

    test_data = finetune.load_test_data(dataset_name, task, generation_method)
    # test_data = test_data.sample(100)
    if task == 'topic-extraction':
        references = test_data['Target'].to_list()
    elif task == 'stance-classification':
        references = test_data['Stance'].to_list()
    datasets = test_data['Dataset']

    select_cols = ['Text']
    if task == 'stance-classification':
        select_cols.append('Target')
    df = test_data.select(select_cols)

    if task == 'topic-extraction':
        config = {
            'generation_method': 'list',
            'hf_model': 'bendavidsteel/SmolLM2-360M-Instruct-stance-target-extraction'
        }
    elif task == 'stance-classification':
        config = {
            'classification_method': 'head',
            'hf_model': 'bendavidsteel/SmolLM2-135M-Instruct-stance-detection'
        }
    else:
        raise ValueError()
    
    output_type = config['classification_method'] if task == "stance-classification" else config['generation_method']
    model_save_path = config['hf_model']
    file_path = huggingface_hub.hf_hub_download(repo_id=model_save_path, filename='metadata.json')
    with open(file_path, 'r') as f:
        metadata = json.load(f)
    prompt = metadata['prompt']
    parent_prompt = metadata['parent_prompt'] if 'parent_prompt' in metadata else None

    # Setup configurations
    model_config = finetune.ModelConfig(
        model_name=None,
        task=task,
        num_labels=2 if task == "argument-classification" else 3,
        prompt=prompt,
        parent_prompt=parent_prompt,
        classification_method=config['classification_method'] if task == 'stance-classification' else None,
        generation_method=config['generation_method'] if task == 'topic-extraction' else None,
    )
    
    data_config = finetune.DataConfig(
        dataset_name=None
    )
    
    # Initialize components
    processor = finetune.DataProcessor(model_config, data_config)
    test_dataset = processor.process_data(df, model_config.classification_method, model_config.generation_method, train=False, tokenize=False)
    prompts = test_dataset['text']
    prompts = [finetune.to_message_format(p) for p in prompts]

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
        # os.environ['VLLM_USE_V1'] = '0' # https://github.com/vllm-project/vllm/pull/16188 remove when this is merged
    else:
        raise ValueError()

    predictions = []
    llm = vllm.LLM(
        model=model_name,
        enable_lora=True,
        **llm_kwargs
    )
    # greedy decoding
    sampling_params = vllm.SamplingParams(temperature=0.0)
    if task == 'topic-extraction':
        lora_request = vllm.lora.request.LoRARequest(
            f"{task}_adapter",
            1,
            adapter_path
        )

    model_kwargs = {
        'device_map': {'': 1}
    }
    model_config = finetune.ModelConfig(
        model_name=None,
        task=task,
        device_map=model_kwargs['device_map'],
        num_labels=2 if task == "argument-classification" else 3,
        prompt=prompt,
        parent_prompt=parent_prompt,
        classification_method=config['classification_method'] if task == 'stance-classification' else None,
        generation_method=config['generation_method'] if task == 'topic-extraction' else None,
    )
    # Initialize components
    model, tokenizer = finetune.setup_model_and_tokenizer(
        model_config.task, 
        model_config.classification_method, 
        model_config.num_labels, 
        model_kwargs=model_kwargs, 
        model_save_path=model_save_path,
        full_saved_model=True
    )
    model_config.model, model_config.tokenizer = model, tokenizer
    processor = finetune.DataProcessor(model_config, data_config)
    test_dataset = processor.process_data(df, model_config.classification_method, model_config.generation_method, train=False)
    
    id2labels = {v: k for k, v in data_config.labels2id.items()}

    vllm_preds = []
    hf_preds = []
    for prompt, inputs in zip(prompts, test_dataset):
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0),
            'attention_mask': inputs['attention_mask'].unsqueeze(0)
        }
        
        pred_id = finetune.get_prediction(
            inputs, 
            task, 
            model, 
            tokenizer, 
            model_config.classification_method,
            model_config.generation_method,
            generate_kwargs={}
        )
        pred = id2labels[pred_id[0]]

        if task == 'topic-extraction' or (task == 'stance-classification' and model_config.classification_method == 'generation'):
            outputs = llm.chat(messages=[prompt], sampling_params=sampling_params, use_tqdm=False, lora_request=lora_request)
            predictions = [o.outputs[0].text for o in outputs]
            predictions = finetune.parse_list_completions(predictions)
        elif task == 'stance-classification' and model_config.classification_method == 'head':
            prompt = tokenizer.apply_chat_template(
                prompt, 
                add_generation_prompt=True,
                truncation=True,
                max_length=2048,
                padding='max_length',
                return_token_type_ids=False, 
                enable_thinking=False,
                tokenize=False
            )
            outputs = llm.classify([prompt], use_tqdm=False)
            probs = [o.outputs.probs for o in outputs]
            predictions = [np.argmax(p) for p in probs]
            
            predictions = [id2labels[p] for p in predictions]
        else:
            raise ValueError()
        
        vllm_preds.append(predictions[0])
        hf_preds.append(pred)

    print_results(task, 'vllm', vllm_preds, references, datasets, datetime.datetime.now(), datetime.datetime.now())
    print_results(task, 'hf', hf_preds, references, datasets, datetime.datetime.now(), datetime.datetime.now())

    

if __name__ == '__main__':
    main()
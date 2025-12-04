import json
import os

import huggingface_hub
import hydra
import numpy as np
import omegaconf
import polars as pl
import torch
import vllm
import vllm.lora.request
import wandb

from stancemining.finetune import (
    ModelConfig, 
    DataConfig, 
    DataProcessor, 
    ModelEvaluator, 
    load_prompt,
    load_parent_prompt,
    load_context_prompt,
    load_test_data,
    get_model_save_path,
    print_metrics,
    to_message_format,
    CLASSIFICATION_TASKS,
    GENERATION_TASKS
)
import stancemining.prompting
from stancemining.llms import get_max_new_tokens, parse_list_completions, parse_category_completions, parse_answer_from_thinking

def print_results(task, method, results, references, datasets):
    evaluator = ModelEvaluator(task)

    metrics = evaluator.evaluate(
        results,
        references,
        datasets
    )
    print(f"Method: {method}")
    if task in GENERATION_TASKS:
        print(f"Metrics: bertscore f1: {metrics['bertscore_f1']}, bleu f1: {metrics['bleu_f1']}")
    elif task in CLASSIFICATION_TASKS:
        print(f"Metrics: F1: {metrics['f1_macro']}")

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    args = config.finetune

    torch.cuda.init()

    if args.task == "stance-classification":
        project_name = 'stance-detection'
    elif args.task == "topic-extraction":
        project_name = 'stance-target-extraction'
    else:
        project_name = args.task

    wandb_config = omegaconf.OmegaConf.to_object(args)
    wandb_config.update(omegaconf.OmegaConf.to_object(config.data))

    wandb.init(project=project_name, config=wandb_config)

    if config.test.in_context_prompt:
        if args.task == 'stance-classification':
            prompt = stancemining.prompting.NOUN_PHRASE_STANCE_DETECTION
            context_prompt = None
        elif args.task == 'claim-entailment-4way':
            prompt = stancemining.prompting.CLAIM_STANCE_DETECTION_4_LABELS
            context_prompt = stancemining.prompting.CLAIM_CONTEXT_STANCE_DETECTION_4_LABELS
        else:
            raise ValueError(f"In-context prompting not supported for task {args.task}")
        parent_prompt = None
    else:
        prompt = load_prompt(args.task, args.prompting_method, args.generation_method)
        parent_prompt = load_parent_prompt(args.task, args.prompting_method)
        context_prompt = load_context_prompt(args.task, args.prompting_method)

    # Setup configurations
    model_config = ModelConfig(
        model_name=args.model_name,
        quantization=args.quantization if args.quantization in ['8bit', '4bit'] else None,
        task=args.task,
        classification_method=args.classification_method,
        generation_method=args.generation_method,
        device_map=config.device_map,
        prompt=prompt,
        parent_prompt=parent_prompt,
        context_prompt=context_prompt,
        attn_implementation=args.attn_implementation,
    )
    
    data_config = DataConfig(
        dataset_name=config.data.dataset
    )

    # Initialize components
    processor = DataProcessor(model_config, data_config)
    evaluator = ModelEvaluator(model_config.task)
    
    # Load HF token
    hf_token = config.hf_token
    os.environ['HF_TOKEN'] = hf_token
    
    # Setup model path
    output_type = model_config.classification_method if model_config.task in CLASSIFICATION_TASKS else model_config.generation_method

    if config.test.finetuned_model:
        model_save_path = get_model_save_path(args.task, args.save_model_path, args.model_name, data_config.dataset_name, output_type)

    test_data = load_test_data(data_config.dataset_name, model_config.task, model_config.generation_method)

    # drop long text
    test_data = test_data.with_columns(pl.col('Text').str.len_chars().alias('text_len')).filter(pl.col('text_len') < 30000).drop('text_len')

    test_dataset = processor.process_data(test_data, model_config.classification_method, model_config.generation_method, train=False, tokenize=False)
    
    # Initialize components
    prompts = list(test_dataset['text'])
    if isinstance(prompts[0], str):
        prompts = [to_message_format(p) for p in prompts]

    llm_kwargs = {
        'gpu_memory_utilization': 0.90,
        'max_model_len': 8192,
        'enable_prefix_caching': True,
    }
    generate_kwargs = {}
    if model_config.task in GENERATION_TASKS or (model_config.task in CLASSIFICATION_TASKS and model_config.classification_method == 'generation'):
        llm_kwargs['task'] = 'generate'
        llm_kwargs['generation_config'] = 'auto'

        if config.test.finetuned_model:
            file_path = huggingface_hub.hf_hub_download(repo_id=model_save_path, filename='adapter_config.json')
            with open(file_path, 'r') as f:
                adapter_config = json.load(f)
            adapter_path = huggingface_hub.snapshot_download(repo_id=model_save_path)
            model_name = adapter_config['base_model_name_or_path']
            llm_kwargs['enable_lora'] = True
            generate_kwargs['lora_request'] = vllm.lora.request.LoRARequest(
                f"{model_config.task}_adapter",
                1,
                adapter_path
            )

    elif model_config.task in CLASSIFICATION_TASKS and model_config.classification_method == 'head':
        llm_kwargs['task'] = 'classify'
        llm_kwargs['enforce_eager'] = True
        model_name = config['hf_model']
        # os.environ['VLLM_USE_V1'] = '0' # https://github.com/vllm-project/vllm/pull/16188 remove when this is merged
    else:
        raise ValueError()
    
    
    
    if prompts[0][-1]['role'] == 'assistant':
        if config.test.enable_thinking:
            # remove final assistant message to allow model to generate the "thinking" step
            prompts = [p[:-1] for p in prompts]
        
        else:
            generate_kwargs['add_generation_prompt'] = False
            generate_kwargs['continue_final_message'] = True

    generate_kwargs['chat_template_kwargs'] = { 'enable_thinking': config.test.enable_thinking }

    max_new_tokens = get_max_new_tokens(model_config.task, model_config)

    # greedy decoding
    generate_kwargs['sampling_params'] = vllm.SamplingParams(
        temperature=0.0,
        max_tokens=None
    )

    llm = vllm.LLM(
        model=model_config.model_name,
        **llm_kwargs
    )

    prompts = prompts[:100]
    test_dataset = test_dataset.take(100)

    if model_config.task in GENERATION_TASKS or (model_config.task in CLASSIFICATION_TASKS and model_config.classification_method == 'generation'):
        outputs = llm.chat(messages=prompts, **generate_kwargs, use_tqdm=True)
        predictions = [o.outputs[0].text for o in outputs]
        if config.test.enable_thinking:
            predictions = [parse_answer_from_thinking(p) for p in predictions]
        if model_config.task in GENERATION_TASKS:
            predictions = parse_list_completions(predictions)
        elif model_config.task in CLASSIFICATION_TASKS:
            predictions = parse_category_completions(predictions, model_config.task)
    elif model_config.task in CLASSIFICATION_TASKS and model_config.classification_method == 'head':
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
        outputs = llm.classify(prompts, use_tqdm=False)
        probs = [o.outputs.probs for o in outputs]
        predictions = [np.argmax(p) for p in probs]
        id2labels = {v: k for k, v in model_config.labels2id.items()}
        predictions = [id2labels[p] for p in predictions]
    else:
        raise ValueError()
    
    datasets = test_dataset.to_polars()['dataset'].to_list()
    if model_config.task in CLASSIFICATION_TASKS:
        references = test_dataset.to_polars()['class'].to_list()  # for some insane reason test_dataset['class'] does not work
    else:
        references = test_data['Target'].to_list()

    metrics = evaluator.evaluate(
        predictions,
        references,
        datasets
    )
    
    print_metrics(metrics)
    test_metrics = {f"test/{k}": v for k, v in metrics.items()}
    wandb.run.summary.update(test_metrics)
    wandb.finish()

    

if __name__ == '__main__':
    main()
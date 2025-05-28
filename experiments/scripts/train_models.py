import argparse
import os

import dotenv
import hydra
import omegaconf
import transformers
import tqdm
import wandb

from stancemining.finetune import (
    ModelConfig, 
    DataConfig, 
    TrainingConfig, 
    ModelTrainer, 
    DataProcessor, 
    ModelEvaluator, 
    load_prompt,
    load_parent_prompt,
    load_training_data,
    load_validation_data,
    load_test_data,
    get_model_save_path,
    print_metrics,
    get_prediction,
    setup_model_and_tokenizer
)
from experiments.metrics import bertscore_f1_targets, bleu_targets

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    _main(config, config.finetune)

def _main(config, args):

    if args.task == "stance-classification":
        project_name = 'stance-detection'
    elif args.task == "topic-extraction":
        project_name = 'stance-target-extraction'
    else:
        raise ValueError(f"Invalid task: {args.task}")

    wandb_config = omegaconf.OmegaConf.to_object(args)
    wandb_config.update(omegaconf.OmegaConf.to_object(config.data))

    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,

        # track hyperparameters and run metadata
        config=wandb_config
    )

    # Setup configurations
    model_config = ModelConfig(
        model_name=args.model_name,
        quantization=args.quantization if args.quantization in ['8bit', '4bit'] else None,
        task=args.task,
        num_labels=2 if args.task == "argument-classification" else 3,
        classification_method=args.classification_method,
        generation_method=args.generation_method,
        device_map={"": config.device_id},
        prompt=load_prompt(args.task, args.prompting_method),
        parent_prompt=load_parent_prompt(args.task, args.prompting_method)
    )
    
    data_config = DataConfig(
        dataset_name=config.data.dataset,
        labels2id={
            "neutral": 0,
            "favor": 1,
            "against": 2
        }
    )
    
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        eval_steps=args.eval_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate
    )

    # Initialize components
    trainer = ModelTrainer(model_config, training_config)
    processor = DataProcessor(model_config, data_config)
    evaluator = ModelEvaluator(model_config, data_config)
    
    # Load HF token
    dotenv.load_dotenv()
    hf_token = os.environ['HF_TOKEN']
    
    # Setup model path
    output_type = model_config.classification_method if model_config.task == "stance-classification" else model_config.generation_method
    model_save_path = get_model_save_path(args.task, args.save_model_path, args.model_name, data_config.dataset_name, output_type)

    model_kwargs = {
        'device_map': model_config.device_map,
        'token': hf_token,
        'attn_implementation': model_config.attn_implementation,
        'torch_dtype': model_config.torch_dtype
    }

    if model_config.quantization is not None:
        if model_config.quantization == '8bit':
            quant_config = transformers.BitsAndBytesConfig(
                load_in_8bit=True
            )
        elif model_config.quantization == '4bit':
            quant_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=model_config.torch_dtype,
            )
        else:
            raise ValueError(f"Unknown quantization type: {model_config.quantization}")
        model_kwargs['quantization_config'] = quant_config

    if args.do_train:
        # Process training data
        train_data = load_training_data(data_config.dataset_name, model_config.task, model_config.generation_method)
        val_data = load_validation_data(data_config.dataset_name, model_config.task, model_config.generation_method)

        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(model_config.task, model_config.classification_method, model_config.num_labels, model_kwargs=model_kwargs, model_name=model_config.model_name)
        trainer.set_model_and_tokenizer(model, tokenizer)

        train_dataset = processor.process_data(train_data, model_config.classification_method, model_config.generation_method)
        val_dataset = processor.process_data(val_data, model_config.classification_method, model_config.generation_method, train=False)

        trainer.prepare_for_training()
        
        # Train model
        trainer.train(train_dataset, val_dataset, model_save_path, evaluator)
    
    if args.do_eval:
        model, tokenizer = setup_model_and_tokenizer(model_config.task, model_config.classification_method, model_config.num_labels, model_kwargs=model_kwargs, model_save_path=model_save_path)
        trainer.set_model_and_tokenizer(model, tokenizer)

        test_data = load_test_data(data_config.dataset_name, model_config.task, model_config.generation_method)
        test_dataset = processor.process_data(test_data, model_config.classification_method, model_config.generation_method, train=False)
        
        predictions = []
        test_loader = processor.get_loader(test_dataset, {"batch_size": training_config.batch_size})

        generate_kwargs = {}
        if trainer.model_config.generation_method == 'beam':
            num_samples = 3
            generate_kwargs['num_beams'] = num_samples * 5
            generate_kwargs['num_return_sequences'] = num_samples
            generate_kwargs['num_beam_groups'] = num_samples
            generate_kwargs['diversity_penalty'] = 0.5
            generate_kwargs['no_repeat_ngram_size'] = 2
            generate_kwargs['do_sample'] = False
        
        for inputs in tqdm.tqdm(test_loader, desc="Evaluating"):
            predictions.extend(get_prediction(
                inputs, 
                trainer.model_config.task, 
                trainer.model_config.model, 
                trainer.model_config.tokenizer, 
                trainer.model_config.classification_method,
                trainer.model_config.generation_method,
                generate_kwargs=generate_kwargs
            ))
        
        if model_config.task in ["argument-classification", "stance-classification"]:
            references = test_dataset['class']
            datasets = test_dataset['dataset']
            metrics = evaluator.evaluate(
                predictions,
                references,
                datasets
            )
        else:
            references = test_data['Target'].to_list()
            b_f1, b_precision, b_recall = bertscore_f1_targets(predictions, references)
            # bleu_score = bleu_targets(predictions, references)
            metrics = {
                'bertscore_f1': b_f1,
                'bertscore_precision': b_precision,
                'bertscore_recall': b_recall,
                # 'bleu_score': bleu_score
            }
        
        print_metrics(metrics)
        test_metrics = {f"test/{k}": v for k, v in metrics.items()}
        wandb.run.summary.update(test_metrics)
    wandb.finish()

if __name__ == "__main__":
    main()
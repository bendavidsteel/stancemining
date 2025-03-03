import argparse
import os

import dotenv
import hydra
import omegaconf
import torch
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
        task=args.task,
        num_labels=2 if args.task == "argument-classification" else 3,
        classification_method=args.classification_method,
        generation_method=args.generation_method,
        device_map={"": config.device_id},
        prompt=load_prompt(args.task, args.prompting_method)
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

    if args.do_train:
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(model_config.task, model_config.classification_method, model_config.num_labels, model_config.device_map, model_name=model_config.model_name, hf_token=hf_token)
        trainer.set_model_and_tokenizer(model, tokenizer)

        trainer.prepare_for_training()
        
        # Process training data
        train_data = load_training_data(data_config.dataset_name, model_config.task, model_config.generation_method)
        train_dataset = processor.process_data(train_data, model_config.classification_method, model_config.generation_method)
        val_data = load_validation_data(data_config.dataset_name, model_config.task, model_config.generation_method)
        val_dataset = processor.process_data(val_data, model_config.classification_method, model_config.generation_method, train=False)
        
        # Train model
        trainer.train(train_dataset, val_dataset, model_save_path, evaluator)
    
    if args.do_eval:
        model, tokenizer = setup_model_and_tokenizer(model_config.task, model_config.classification_method, model_config.num_labels, model_config.device_map, model_save_path=model_save_path, hf_token=hf_token)
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
            metrics = evaluator.evaluate(
                predictions,
                references
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
        wandb.run.summary.update(metrics)
    wandb.finish()

if __name__ == "__main__":
    main()
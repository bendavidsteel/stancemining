import argparse
import os

import dotenv
import hydra
import torch
import tqdm

from vectopic.finetune import (
    ModelConfig, 
    DataConfig, 
    TrainingConfig, 
    ModelTrainer, 
    DataProcessor, 
    ModelEvaluator, 
    load_system_message,
    load_training_data,
    load_validation_data,
    load_test_data,
    get_model_save_path,
    print_metrics,
    get_prediction,
    setup_model_and_tokenizer
)

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    args = config.finetune
    # Setup configurations
    model_config = ModelConfig(
        model_name=args.model_name,
        task=args.task,
        num_labels=2 if args.task == "argument-classification" else 3,
        device_map={"": 0},
        system_message=load_system_message(args.task)
    )
    
    data_config = DataConfig(
        dataset_name=config.data.dataset,
        add_system_message=args.add_system_message,
        id2labels={
            "neutral": 0,
            "favor": 1,
            "against": 2
        }
    )
    
    training_config = TrainingConfig(
        num_epochs=args.num_epochs
    )

    # Initialize components
    trainer = ModelTrainer(model_config, training_config)
    processor = DataProcessor(model_config, data_config)
    evaluator = ModelEvaluator(model_config)
    
    # Load HF token
    dotenv.load_dotenv()
    hf_token = os.environ['HF_TOKEN']
    
    # Setup model path
    model_save_path = get_model_save_path(args.task, args.save_model_path, args.model_name, data_config.dataset_name)
    
    if args.do_train:
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(model_config.task, model_config.num_labels, model_config.device_map, model_name=model_config.model_name, hf_token=hf_token)
        trainer.set_model_and_tokenizer(model, tokenizer)
        trainer.prepare_for_training()
        
        # Process training data
        train_data = load_training_data(data_config.dataset_name, model_config.task)
        train_dataset = processor.process_data(train_data)
        val_dataset = processor.process_data(load_validation_data(data_config.dataset_name, model_config.task))
        
        # Train model
        trainer.train(train_dataset, val_dataset, model_save_path, evaluator)
    
    if args.do_eval:
        if not args.do_train:
            model, tokenizer = setup_model_and_tokenizer(model_config.task, model_config.num_labels, model_config.device_map, model_save_path=model_save_path, hf_token=hf_token)
            trainer.set_model_and_tokenizer(model, tokenizer)

        test_data = load_test_data(data_config.dataset_name, model_config.task)
        test_dataset = processor.process_data(test_data, train=False)
        
        predictions = []
        test_loader = processor.get_loader(test_dataset, {"batch_size": 1})
        for inputs in tqdm.tqdm(test_loader, desc="Evaluating"):
            predictions.extend(get_prediction(inputs, trainer.model_config.task, trainer.model_config.model, trainer.model_config.tokenizer))
        
        if model_config.task in ["argument-classification", "stance-classification"]:
            references = test_dataset['class']
        else:
            references = test_data['Target']
        metrics = evaluator.evaluate(
            predictions,
            references
        )
        print_metrics(metrics)

if __name__ == "__main__":
    main()
import argparse
import os

import accelerate
import datasets
import dotenv
import evaluate
import numpy as np
import pandas as pd
import peft
from sklearn.preprocessing import LabelEncoder
import torch
import tqdm as tqdm
import transformers


def find_all_linear_names(model):
    cls = torch.nn.Linear #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def to_message_format(text, label):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]
    if label is not None:
        messages.append({"role": "assistant", "content": label})
    return messages

def setup_tokens(model, tokenizer, task):
    if task == "stance-classification" or task == "argument-classification":
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        # model.resize_token_embeddings(len(tokenizer))
    elif task == "topic-extraction":
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    else:
        raise ValueError("Task not found")
    
    return model, tokenizer

class ChatTemplateTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def create_input_sequence_for_classification(self, sample):
        messages = to_message_format(sample['text'], None)
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            truncation=True,
            max_length=2048,
            padding='max_length',
            return_token_type_ids=False, 
            return_tensors='pt',
            return_dict=True
        )
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        return inputs
    
    def create_input_sequence_for_causal_lm(self, sample):
        messages = to_message_format(sample['text'], sample['labels'])
        
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            truncation=True, 
            max_length=2048, 
            padding='max_length', 
            return_tensors="pt",
            return_dict=True
        )

        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        
        # Find the assistant's response start
        response_tokens = self.tokenizer.encode(sample['labels'], add_special_tokens=False)
        response_start = None
        
        # Find where the assistant's response starts in the tokenized input
        for i in range(len(inputs['input_ids']) - len(response_tokens)):
            if inputs['input_ids'][i:i+len(response_tokens)].tolist() == response_tokens:
                response_start = i
                break
        
        # Create labels tensor with -100s before the response
        labels = inputs['input_ids'].clone()
        labels[:response_start] = -100
        
        return {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "labels": labels
        }

def train(model_name, hf_token, training_data, task, add_data, add_system_message, add_topic, num_epochs, model_path_name, num_labels, id2labels, system_message):
    
    device_map = {'': 0}
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if task == "stance-classification" or task == "argument-classification":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, torch_dtype=torch.float16, device_map=device_map, attn_implementation='flash_attention_2', token=hf_token)
    elif task == "topic-extraction":
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device_map, attn_implementation='flash_attention_2', token=hf_token)
    else:
        raise ValueError("Task not found")

    model, tokenizer = setup_tokens(model, tokenizer, task)
      

    ### LIMA TRAIN ### 
    ### Using the Less is More approach ### 
    if training_data == "semeval":
        data = pd.read_csv('./data/semeval_train.csv')

        # TODO to remove
        data = data.iloc[:50]

        data = data[['Target','Tweet','Stance']]
        
        if task == "stance-classification":
            data = data.rename(columns={"Stance": "class","Tweet":"text", "Target":"topic"})
            data["class"] = data["class"].map(lambda l: {'AGAINST': 'Argument_against', 'FAVOR': 'Argument_for', 'NONE': 'NoArgument'}[l])
            #map labels
            data["class"] = data["class"].map(id2labels)
            data = data[["text","class","topic"]]

        elif task == "argument-classification":
            data['text'] = data['sentence']
            data = data.rename(columns={"annotation_real": "class"})
            data["text"] = data["text"].astype(str)
            if add_topic:
                data = data[['text','class','topic']]
            else:
                data = data[["text","class"]]

            data["class"] = data["class"].apply(lambda x: 1 if x != "NoArgument" else 0) 
        elif task == "topic-extraction":
            data = data.rename(columns={"Stance": "class","Tweet":"text", "Target":"topic"})
            data = data[['text','class','topic']]
        else:
            raise ValueError("Task not found")
    
    if task == "stance-classification":
        if training_data == "semeval":
            val_split = 0.2
            data = data.sample(frac=1, random_state=42).reset_index(drop=True)
            data, val = data.iloc[:int(len(data) * (1 - val_split))], data.iloc[int(len(data) * (1 - val_split)):]
            print(val['class'].value_counts())
        else:
            raise ValueError("Training data not found")
        
    elif task == "argument-classification":
        raise NotImplementedError("Argument Classification not implemented")
        
    elif task == "topic-extraction":
        if training_data == "semeval":
            val_split = 0.2
            data = data.sample(frac=1, random_state=42).reset_index(drop=True)
            data, val = data.iloc[:int(len(data) * (1 - val_split))], data.iloc[int(len(data) * (1 - val_split)):]
            print(val['topic'].value_counts())
        else:
            raise ValueError("Training data not found")

    else:
        raise ValueError("Task not found")

    train_dataset = datasets.Dataset.from_pandas(data)

    if add_system_message:
        if task == "stance-classification":
            train_dataset = train_dataset.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Topic: '" + topic +"' Text: '" +sentence  + "' [/INST] " for topic,sentence in zip(examples['topic'],examples['text'])]}, batched=True)
        elif task == "argument-classification" and add_topic == False:
            train_dataset = train_dataset.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Text: '" + prompt +"' [/INST] " for prompt in examples['text']]}, batched=True)
        elif task == "argument-classification" and add_topic == True:
            train_dataset = train_dataset.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Topic: '" + topic +"' Text: '" +sentence  + "' [/INST] " for topic,sentence in zip(examples['topic'],examples['text'])]}, batched=True)
        elif task == "topic-extraction":
            train_dataset = train_dataset.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Text: '" + prompt +"' [/INST] " for prompt in examples['text']]}, batched=True)
        else:
            raise ValueError("Task not found")
    else:
        if task == "stance-classification":
            train_dataset = train_dataset.map(lambda examples: {"text": [f"Target: '" + topic +" Text:'" + sentence+"'" for topic,sentence in zip(examples['topic'],examples['text'])]}, batched=True)
        else:
            train_dataset = train_dataset.map(lambda examples: {"text": [f"Text:'" + sentence+"'" for sentence in examples['text']]}, batched=True)   
    
    t = ChatTemplateTokenizer(tokenizer)
    if task == "stance-classification" or task == "argument-classification":
        train_dataset = train_dataset.map(t.create_input_sequence_for_classification)
        train_dataset = train_dataset.rename_column("class", "labels")
    elif task == "topic-extraction":
        train_dataset = train_dataset.rename_column("topic", "labels")
        # drop labels column
        train_dataset = train_dataset.remove_columns("class")
        train_dataset = train_dataset.map(t.create_input_sequence_for_causal_lm)
    else:
        raise ValueError("Task not found")
    train_dataset.set_format('torch')
    train_dataset_tokenized = train_dataset.shuffle(seed=1234)  # Shuffle dataset here


    validation_dataset = datasets.Dataset.from_pandas(val)    
    if add_system_message:
        if task == "stance-classification":
            validation_dataset = validation_dataset.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Target: '" + topic +"' Text: '" +sentence  + "' [/INST] " for topic,sentence in zip(examples['topic'],examples['text'])]}, batched=True)
        elif task == "argument-classification" and add_topic ==False:
            validation_dataset = validation_dataset.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Text: '" + prompt +"' [/INST] " for prompt in examples['text']]}, batched=True)
        elif task == "argument-classification" and add_topic == True:
            validation_dataset = validation_dataset.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Target: '" + topic +"' Text: '" +sentence  + "' [/INST] " for topic,sentence in zip(examples['topic'],examples['text'])]}, batched=True)
        elif task == "topic-extraction":
            validation_dataset = validation_dataset.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Text: '" + prompt +"' [/INST] " for prompt in examples['text']]}, batched=True)
        else:
            raise ValueError("Task not found")
    else:
        if task == "stance-classification":
            validation_dataset = validation_dataset.map(lambda examples: {"text": [f"Target: '" + topic +" Text:'" + sentence+"'" for topic,sentence in zip(examples['topic'],examples['text'])]}, batched=True)
        else:
            validation_dataset = validation_dataset.map(lambda examples: {"text": [f"Text:'" + sentence+"'" for sentence in examples['text']]}, batched=True) 

    
    if task == "stance-classification" or task == "argument-classification":
        validation_dataset = validation_dataset.map(t.create_input_sequence_for_classification)
        validation_dataset = validation_dataset.rename_column("class", "labels")
    elif task == "topic-extraction":
        validation_dataset = validation_dataset.rename_column("topic", "labels")
        # drop labels column
        validation_dataset = validation_dataset.remove_columns("class")
        validation_dataset = validation_dataset.map(t.create_input_sequence_for_causal_lm)
    else:
        raise ValueError("Task not found")
    validation_dataset.set_format('torch')
    #shuffle
    validation_dataset = validation_dataset.shuffle(seed=1234)  # Shuffle dataset here

    print(validation_dataset['labels'])
    print(train_dataset_tokenized['labels']) 
    if task == "stance-classification" or task == "argument-classification":   
        llm_data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    elif task == "topic-extraction":
        llm_data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        raise ValueError("Task not found")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # model = peft.prepare_model_for_kbit_training(model)
    modules = find_all_linear_names(model)

    ### Can experiment here, these values worked the best ### 
    lora_kwargs = {}
    if task == "stance-classification" or task == "argument-classification":
        lora_kwargs['task_type'] = "SEQ_CLS"
        lora_kwargs['modules_to_save'] = ['score']
    elif task == "topic-extraction":
        lora_kwargs['task_type'] = "CAUSAL_LM"
    else:
        raise ValueError("Task not found")

    lora_config = peft.LoraConfig(
        r=8,   #8 or 32 or 64                    
        lora_alpha= 32, # 32 or 64 or 16 
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        **lora_kwargs
    )

    if task == "stance-classification" or task == "argument-classification":
        model = peft.prepare_model_for_kbit_training(model)
    model = peft.get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
        if task == "stance-classification" or task == "argument-classification":
            precision_metric = evaluate.load("precision")
            recall_metric = evaluate.load("recall")
            f1_metric= evaluate.load("f1")
            accuracy_metric = evaluate.load("accuracy")
            # if "bart" in model_name:
            #     logits = logits[0]
            # predictions = np.argmax(logits, axis=-1)
            
            print(f"predictions: {predictions}; labels: {labels}")
            
            precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')["precision"]
            recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')["recall"]
            f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')["f1"]
            accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
            
            # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores. 
            return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}
        elif task == "topic-extraction":
            bertscore = evaluate.load("bertscore")
            
            # Compute BERTScore
            bertscore_results = bertscore.compute(predictions=predictions, references=labels, lang="en")
            bertscore_precision = sum(bertscore_results["precision"]) / len(bertscore_results["precision"])
            bertscore_recall = sum(bertscore_results["recall"]) / len(bertscore_results["recall"])
            bertscore_f1 = sum(bertscore_results["f1"]) / len(bertscore_results["f1"])

            # Print evaluation statistics
            print(f"BERTScore: {bertscore_f1:.2f}")
            print(f"BERTScore Precision: {bertscore_precision:.2f}")
            print(f"BERTScore Recall: {bertscore_recall:.2f}")
            print(f"BERTScore F1: {bertscore_f1:.2f}")
            return {"bertscore-precision": bertscore_precision, "bertscore-recall": bertscore_recall, "bertscore-f1": bertscore_f1}
        
    # model = model.cuda()

    lr = 3e-05

    if False:
        training_kwargs = {}
        if task == "stance-classification" or task == "argument-classification":
            training_kwargs['label_smoothing_factor'] = 0.1

        training_output_dir = "./outputs/wiba_training/"
        logging_dir = "./outputs/logs/wiba_training/"
        training_args = transformers.TrainingArguments(
            output_dir=training_output_dir, 
            num_train_epochs=num_epochs,  # demo
            do_train=True,
            do_eval=True,
            auto_find_batch_size=True,
            lr_scheduler_type="constant", #cosine
            # lr_scheduler_kwargs=lr_scheduler_config,
            learning_rate=lr, #0.00001,#3e-05,
            warmup_steps=500,
            # warmup_ratio = 0.03,
            max_grad_norm= 0.3,
            weight_decay=0.1,
            eval_strategy=IntervalStrategy.STEPS,
            optim="paged_adamw_32bit", #"adafactor", #paged_adamw_32bit
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            fp16=True,
            logging_dir=logging_dir,
            logging_steps=100,
            save_total_limit=50,
            eval_steps=100,
            load_best_model_at_end=True,
            neftune_noise_alpha=5,#0.1
            # torch_compile=True,
            **training_kwargs
        )

        llm_trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_tokenized,
            eval_dataset=validation_dataset,
            data_collator=llm_data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        llm_trainer.train()
        llm_trainer.model.save_pretrained(model_path_name)
        tokenizer.save_pretrained(model_path_name)

    if True:
        model.config.use_cache = False
        train_model(
            model=model,
            train_dataset=train_dataset_tokenized,
            validation_dataset=validation_dataset,
            collator=llm_data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            task=task,
            model_path_name=model_path_name,
            num_epochs=num_epochs,
            learning_rate=lr,
            weight_decay=0.1,
            max_grad_norm=0.3,
            grad_accum_steps=8,
            batch_size=1,
            eval_steps=10#0 TODO change to 100
        )

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

def train_model(
    model,
    train_dataset,
    validation_dataset,
    collator,
    tokenizer,
    compute_metrics,
    task,
    model_path_name,
    num_epochs=3,
    learning_rate=1e-5,
    weight_decay=0.1,
    max_grad_norm=0.3,
    batch_size=8,
    grad_accum_steps=4,
    warmup_steps=500,
    eval_steps=100,
    neftune_noise_alpha=5
):
    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset.select_columns(['input_ids', 'attention_mask', 'labels']), batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(validation_dataset.select_columns(['input_ids', 'attention_mask', 'labels']), batch_size=batch_size)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Setup learning rate scheduler
    scheduler = transformers.get_constant_schedule(optimizer)
    
    accelerator = accelerate.Accelerator(mixed_precision='fp16', gradient_accumulation_steps=grad_accum_steps)
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, eval_loader, scheduler)
    
    # Training loop
    best_eval_metric = float('inf')
    if task == "stance-classification" or task == "argument-classification":
        chosen_metric = "f1-score"
    elif task == "topic-extraction":
        chosen_metric = "bertscore-f1"
    global_step = 0
    optimizer.zero_grad()
    
    model, neftune_hook_handle = activate_neftune(model, accelerator, neftune_noise_alpha)
    
    loss = float('inf')
    pbar = tqdm.tqdm(total=eval_steps * grad_accum_steps, desc=f"Training round, loss: {loss:.4f}")
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            # Move batch to device
            # batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            
            accelerator.backward(loss)

            pbar.set_description(f"Training round, loss: {loss.item():.4e}")
            pbar.update(1)
            
            # Gradient accumulation
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
            
                # Evaluation
                if global_step % eval_steps == 0:
                    deactivate_neftune(model, accelerator, neftune_hook_handle)
                    metrics = validation_step(model, tokenizer, eval_loader, compute_metrics, task)
                    state_str = f"Step {global_step},"
                    for key, val in metrics.items():
                        state_str += f" {key.title()}: {val:.4f},"
                    print(state_str)
                    eval_metric = metrics[chosen_metric]
                    
                    # Save best model
                    if eval_metric < best_eval_metric:
                        best_eval_metric = eval_metric
                        model.save_pretrained(model_path_name)
                        tokenizer.save_pretrained(model_path_name)

                    model, neftune_hook_handle = activate_neftune(model, accelerator, neftune_noise_alpha)
                    pbar = tqdm.tqdm(total=eval_steps * grad_accum_steps, desc="Training round")

def validation_step(model, tokenizer, eval_loader, compute_metrics, task):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm.tqdm(total=len(eval_loader), desc="Validation round")
    with torch.no_grad():
        for batch in eval_loader:
            pbar.update(1)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            preds = get_prediction(model, tokenizer, task, batch)
            all_preds.extend(preds)
            if task == "stance-classification" or task == "argument-classification":
                all_labels.extend(batch["labels"].cpu().numpy())
            elif task == "topic-extraction":
                # TODO this will break if batch_size > 1
                all_labels.extend(tokenizer.batch_decode(batch["labels"][batch['labels'] != -100].unsqueeze(0), skip_special_tokens=True))
    
    metrics = compute_metrics((all_preds, all_labels))
    
    model.train()
    return metrics

def get_prediction(model, tokenizer, task, inputs):
    if task == "stance-classification" or task == "argument-classification":
        output = model(**inputs)
        logits = output.logits
        predicted_class = torch.argmax(logits, dim=1)
        # Convert logits to a list of predicted labels
        return predicted_class.cpu().tolist()
    elif task == "topic-extraction":
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
        output = model.generate(**prompt, max_new_tokens=20)
        completion = tokenizer.decode(output[0][prompt['input_ids'].shape[1]:], skip_special_tokens=True)
        # Convert logits to a list of predicted labels
        return [completion]

def predict(test_data, task, add_system_message, add_topic, model_path_name, num_labels, device_map, id2labels, system_message):
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path_name)
    if task == "stance-classification" or task == "argument-classification":
        model = peft.AutoPeftModelForSequenceClassification.from_pretrained(model_path_name, num_labels=num_labels,torch_dtype=torch.float16, device_map=device_map,)
    elif task == "topic-extraction":
        model = peft.AutoPeftModelForCausalLM.from_pretrained(model_path_name, torch_dtype=torch.float16, device_map=device_map)
    else:
        raise ValueError("Task not found")
    
    model, tokenizer = setup_tokens(model, tokenizer, task)
    
    device = torch.device("cuda")
    
    if test_data == "semeval":
        # df_test = pd.read_csv('./data/semeval_test.csv')
        # TODO to remove
        df_test = pd.read_csv('./data/semeval_train.csv')
        df_test = df_test.iloc[:50]
        val_split = 0.2
        df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
        df_test = df_test.iloc[int(len(df_test) * (1 - val_split)):]

        if task == "stance-classification":
            df_test = df_test.rename(columns={"Stance": "target", "Tweet": "text", "Target": "topic"})
            # target map to labels
            print(df_test)
            df_test['target'] = df_test['target'].map(lambda l: {'AGAINST': 'Argument_against', 'FAVOR': 'Argument_for', 'NONE': 'NoArgument'}[l])
            df_test["target"] = df_test["target"].map(id2labels)
            
            df_test = df_test[['text','target','topic']]

        elif task == "topic-extraction":
            df_test = df_test.rename(columns={"Target":"target", "Tweet":"text"})
            df_test = df_test[['text','target']]
        else:
            raise ValueError("Task not found")

    else:
        raise ValueError("Test data not found")


    test_data_text_only = datasets.Dataset.from_pandas(df_test)
    
    if add_system_message:
        if task == "stance-classification":
            test_data_text_only = test_data_text_only.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Target: '" + topic +"' Text: '" +text  + "' [/INST] " for topic,text in zip(examples['topic'],examples['text'])]}, batched=True)
        elif task == "argument-classification" and add_topic ==False:
            test_data_text_only = test_data_text_only.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Text: '" + prompt +"' [/INST] " for prompt in examples['text']]}, batched=True)
        elif task == "argument-classification" and add_topic == True:
            test_data_text_only = test_data_text_only.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Target: '" + topic +"' Text: '" +text  + "' [/INST] " for topic,text in zip(examples['topic'],examples['text'])]}, batched=True)
        elif task == "topic-extraction":
            test_data_text_only = test_data_text_only.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Text: '" + prompt +"' [/INST] " for prompt in examples['text']]}, batched=True)
        else:
            raise ValueError("Task not found")
    else:
        if task == "stance-classification":
            test_data_text_only = test_data_text_only.map(lambda examples: {"text": [f"Target: '" + topic +" Text:'" + text+"'" for topic,text in zip(examples['topic'],examples['text'])]}, batched=True)
        else:
            test_data_text_only = test_data_text_only.map(lambda examples: {"text": [f"Text: '" + text+"'" for text in examples['text']]}, batched=True)
    test_data_text_only.set_format('torch')


    if not add_system_message:
        args = test_data_text_only['text']
    else:
        args = test_data_text_only['text']

    t = ChatTemplateTokenizer(tokenizer)
    predictions = []
    model.eval()
    # # Loop through the validation dataset in batches
    for batch in tqdm.tqdm(args):
        with torch.no_grad():
            if task == "stance-classification" or task == "argument-classification":
                inputs = t.create_input_sequence_for_classification({"text": batch})
                inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
                inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)
            elif task == "topic-extraction":
                messages = to_message_format(batch, None)
                inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            else:
                raise ValueError("Task not found")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            predictions.extend(get_prediction(model, tokenizer, task, inputs))

        # Get the ground truth labels
    df_test["pred_topic"] = predictions

    predictions_path = "./outputs/wiba_training/results/predictions.csv"
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    df_test.to_csv(predictions_path,index=False)

    
# Evaluate the model
def eval(task, test_data):
    predictions_path = "./outputs/wiba_training/results/predictions.csv"
    val = pd.read_csv(predictions_path)


    if task == "argument-classification" and test_data == "gpt_pro_all":
        val["target"] = val["target"].astype(str)
        val["target"] = val["target"].apply(lambda x: 1 if x != "NoArgument" else 0)

    if task == "stance-classification" or task == "argument-classification":
        val["pred_topic"] = val["pred_topic"].astype(int)


    labels = val['target'].tolist()
    predictions = val["pred_topic"].tolist()


    print("Completed Predictions")
    if task == "stance-classification" or task == "argument-classification":
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        acc_metric_result = accuracy_metric.compute(predictions=predictions, references=labels)
        f1_macro = f1_metric.compute(predictions=predictions,references=labels,average='macro')
        f1_micro = f1_metric.compute(predictions=predictions,references=labels,average='micro')
        f1_weighted = f1_metric.compute(predictions=predictions,references=labels,average='weighted')
        recall_metric = evaluate.load("recall")
        precision_metric = evaluate.load("precision")
        recall = recall_metric.compute(predictions=predictions,references=labels,average='macro')
        precision = precision_metric.compute(predictions=predictions,references=labels,average='macro')

        # Print evaluation statistics
        print(f"Accuracy: {acc_metric_result['accuracy']:.2%}")
        print(f"F1 Macro: {f1_macro['f1']:.2%}")
        print(f"F1_micro: {f1_micro['f1']:.2%}")
        print(f"F1 Weighted: {f1_weighted['f1']:.2%}")
        print(f"Recall: {recall['recall']:.2%}")
        print(f"Precision: {precision['precision']:.2%}")

    elif task == "topic-extraction":

        # Convert string labels to integers using LabelEncoder
        le = LabelEncoder()
        # Fit on unique values from both predictions and references
        all_labels = list(set(predictions + labels))
        le.fit(all_labels)
        
        # Transform string labels to integers
        pred_encoded = le.transform(predictions)
        label_encoded = le.transform(labels)
        
        # Calculate classification metrics
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        recall_metric = evaluate.load("recall")
        precision_metric = evaluate.load("precision")

        accuracy = accuracy_metric.compute(predictions=pred_encoded, references=label_encoded)
        f1 = f1_metric.compute(predictions=pred_encoded, references=label_encoded, average="weighted")
        recall = recall_metric.compute(predictions=pred_encoded, references=label_encoded, average="weighted")
        precision = precision_metric.compute(predictions=pred_encoded, references=label_encoded, average="weighted")

        # load natural language generation metrics
        # Load NLG metrics
        bleu_metric = evaluate.load("bleu")
        bertscore = evaluate.load("bertscore")
        
        # Compute BLEU score
        bleu = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in labels])
        
        # Compute BERTScore
        bertscore_results = bertscore.compute(predictions=predictions, references=labels, lang="en")
        bertscore_precision = sum(bertscore_results["precision"]) / len(bertscore_results["precision"])
        bertscore_recall = sum(bertscore_results["recall"]) / len(bertscore_results["recall"])
        bertscore_f1 = sum(bertscore_results["f1"]) / len(bertscore_results["f1"])

        # Print evaluation statistics
        print(f"Accuracy: {accuracy['accuracy']:.2%}")
        print(f"F1: {f1['f1']:.2f}")
        print(f"Recall: {recall['recall']:.2f}")
        print(f"Precision: {precision['precision']:.2f}")
        print(f"BLEU: {bleu['bleu']:.2f}")
        print(f"BERTScore: {bertscore_f1:.2f}")
        print(f"BERTScore Precision: {bertscore_precision:.2f}")
        print(f"BERTScore Recall: {bertscore_recall:.2f}")
        print(f"BERTScore F1: {bertscore_f1:.2f}")

def main(args):
    device_map = {"": 0}

    training_data = args.training_data   #"lima" # "full" , "lima"
    task = args.task      #"argument-classification" # "stance-classification" # argument-classification
    add_data = args.add_data
    model_name = args.model_name
    add_system_message = args.add_system_message
    add_topic = args.add_topic

    test_data  = args.test_data #"ukp_human"  #arg_spoken_human,ukp_human,gpt_pro_all  argqual_stance_human   gpt_pro_all_stance   # "ibm" # "gpt" # "ukp" # "speech" "gpt_pro", "gpt_pro_all" "ibm_spoken" "ibm_coling"
    num_epochs = args.num_epochs
    if task == "argument-classification": 
        num_labels = 2
    elif task == "stance-classification":
        num_labels = 3
    elif task == "topic-extraction":
        num_labels = None
    else:
        raise ValueError("Task not found")

    if task == "stance-classification":
        model_path_name = args.save_model_path + "/" + model_name.replace('/', '-') + "-stance-classification"
    elif task == "argument-classification":
        model_path_name = args.save_model_path + "/" + model_name.replace('/', '-') + "-argument-detection"
    elif task == "topic-extraction":
        model_path_name = args.save_model_path + "/" + model_name.replace('/', '-') + "-topic-extraction"
    else:
        raise ValueError("Task not found")

    if task == "stance-classification" or task == "argument-classification":
        with open('./models/wiba/system_message_arg.txt', 'r') as file:
            system_message = file.read()
    elif task == "topic-extraction":
        with open('./models/wiba/system_message_cte.txt', 'r') as file:
            system_message = file.read()
    else:
        raise ValueError("Task not found")

    id2labels = {
        "NoArgument":0,
        "Argument_for":1,
        "Argument_against":2
    }

    dotenv.load_dotenv()
    hf_token = os.environ['HF_TOKEN']

    if args.do_train:
        train(model_name, hf_token, training_data, task, add_data, add_system_message, add_topic, num_epochs, model_path_name, num_labels, id2labels, system_message)

    if args.do_pred:
        predict(test_data, task, add_system_message, add_topic, model_path_name, num_labels, device_map, id2labels, system_message)

    if args.do_eval:
        eval(task, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, default="semeval")
    parser.add_argument("--task", type=str, default="stance-classification")
    parser.add_argument("--add_data", type=bool, default=False)
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--add_system_message", type=bool, default=True)
    parser.add_argument("--add_topic", type=bool, default=False)
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--do_pred", type=bool, default=True)
    parser.add_argument("--do_eval", type=bool, default=True)
    parser.add_argument("--test_data", type=str, default="semeval")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--save_model_path", type=str, default="./models/wiba/")

    args = parser.parse_args()

    main(args)
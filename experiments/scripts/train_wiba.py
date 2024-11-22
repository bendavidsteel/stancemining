import argparse
import os

import datasets
import evaluate
import numpy as np
import pandas as pd
import peft
from sklearn.preprocessing import LabelEncoder
import torch
import tqdm as tqdm
import transformers
from transformers.trainer_utils import IntervalStrategy


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




def train(llm, training_data, task, add_data, add_system_message, add_topic, num_epochs, model_path_name, num_labels, id2labels, system_message):
    if llm =="llama":
        model_name = "meta-llama/Llama-2-7b-hf"
    elif llm == "yi":
        model_name = "01-ai/Yi-6B"
    elif llm == "mistral":
        model_name = "mistralai/Mistral-7B-v0.1"
    elif llm == "bart":
        model_name = "facebook/bart-large"
    elif llm == "llama-3":
        model_name = "meta-llama/Meta-Llama-3-8B" 
    elif llm == 'phi-3.5':
        model_name = "microsoft/Phi-3.5-mini-instruct"
    else:
        raise ValueError("Model not found")   


    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if task == "stance-classification" or task == "argument-classification":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, torch_dtype=torch.float16, device_map='auto', attn_implementation='flash_attention_2')
    elif task == "topic-extraction":
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto', attn_implementation='flash_attention_2')
    else:
        raise ValueError("Task not found")

    if task == "stance-classification" or task == "argument-classification":
        tokenizer.add_special_tokens({'unk_token': '[UNK]'})
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = tokenizer.unk_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    elif task == "topic-extraction":
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError("Task not found")


    def create_input_sequence_for_classification(sample):
       return tokenizer(sample["text"], truncation=True,max_length=2048,return_token_type_ids=False)
    
    def create_input_sequence_for_causal_lm(sample):
        inputs = tokenizer(sample["text"], truncation=True, max_length=2048, padding='max_length')
        labels = tokenizer(sample["labels"], truncation=True, max_length=2048, padding='max_length')

        # Concatenate the input and label tokens
        input_ids = inputs["input_ids"] + labels["input_ids"]
        attention_mask = inputs["attention_mask"] + labels["attention_mask"]

        # Create labels with -100 for input tokens and actual labels for label tokens
        labels_ids = [-100] * len(inputs["input_ids"]) + labels["input_ids"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids
        }
      

    ### LIMA TRAIN ### 
    ### Using the Less is More approach ### 
    if training_data == "semeval":
        data = pd.read_csv('./data/semeval_train.csv')
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


    if training_data == "full":
        data = pd.read_csv('/data/finetuning/ukp_train.csv')
        data.columns = ['text', 'class']
        data['topic'] = data['text'].str.split("Target: ").str[1].str.split(" Text: ").str[0]
        data["text"] = data["text"].str.split("Text: ").str[1]
        data["text"] = data["text"].astype(str)

        
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
    
    
    if task == "stance-classification" or task == "argument-classification":
        train_dataset = train_dataset.map(create_input_sequence_for_classification, batched=True)
        train_dataset = train_dataset.rename_column("class", "labels")
    elif task == "topic-extraction":
        train_dataset = train_dataset.rename_column("topic", "labels")
        # drop labels column
        train_dataset = train_dataset.remove_columns("class")
        train_dataset = train_dataset.map(create_input_sequence_for_causal_lm)
    else:
        raise ValueError("Task not found")
    train_dataset.set_format('torch')
    train_dataset_tokenized = train_dataset.shuffle(seed=1234)  # Shuffle dataset here

    if task == "stance-classification":
        if training_data == "semeval":
            val = pd.read_csv('./data/semeval_train.csv')
            val = val.dropna(subset=["Tweet"])
            val = val[['Tweet','Stance','Target']]
            val = val.rename(columns={"Stance": "class","Tweet":"text", "Target":"topic"})
            val["class"] = val["class"].map(lambda l: {'AGAINST': 'Argument_against', 'FAVOR': 'Argument_for', 'NONE': 'NoArgument'}[l])
            val["class"] = val["class"].map(id2labels)
            print(val['class'].value_counts())
        else:
            raise ValueError("Training data not found")
        
    elif task == "argument-classification":
        raise NotImplementedError("Argument Classification not implemented")
        ### Val Lima ###
        if training_data == "lima":
            val = pd.read_csv('/data/finetuning/ukp_lima_val_stance.csv')
            val = val.dropna(subset=["text"])
            # map
            val["class"] = val["class"].apply(lambda x: 1 if x != "NoArgument" else 0)
            print(val['class'].value_counts())
        
        
        if add_data:
            arg_mapping = {
                "con":1,
                "pro":1,
                "none":0

            }
             
            df_add = pd.read_csv('/data/finetuning/add_ibm_arg.csv')
            df_add = df_add[df_add['test']==True]
            
            # argument to text, stance to label
            df_add = df_add.rename(columns={"argument":"text","stance":"class"})
            df_add = df_add[df_add['class'] == 'none']

            df_add["class"] = df_add["class"].map(arg_mapping)
            df_add = df_add[["text","class",'topic']]

            val = pd.concat([val,df_add],ignore_index=True)
            
        if training_data == "full":
            val = pd.read_csv('/data/finetuning/ukp_full_val.csv',names=["text","class"])
            val["text"] = val["text"].astype(str)
            val['topic'] = val['text'].str.split("Target: ").str[1].str.split(" Text: ").str[0]
            val["text"] = val["text"].str.split("Text: ").str[1]
            val["text"] = val["text"].astype(str)
            val = val[["text","class","topic"]]
            #map
            val["class"] = val["class"].apply(lambda x: 1 if x != "NoArgument" else 0)

    elif task == "topic-extraction":
        if training_data == "semeval":
            val = pd.read_csv('./data/semeval_train.csv')
            val = val.dropna(subset=["Tweet"])
            val = val[['Tweet','Stance','Target']]
            val = val.rename(columns={"Stance": "class","Tweet":"text", "Target":"topic"})
            print(val['topic'].value_counts())
        else:
            raise ValueError("Training data not found")

    else:
        raise ValueError("Task not found")

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
        validation_dataset = validation_dataset.map(create_input_sequence_for_classification)
        validation_dataset = validation_dataset.rename_column("class", "labels")
    elif task == "topic-extraction":
        validation_dataset = validation_dataset.rename_column("topic", "labels")
        # drop labels column
        validation_dataset = validation_dataset.remove_columns("class")
        validation_dataset = validation_dataset.map(create_input_sequence_for_causal_lm)
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

    model.gradient_checkpointing_enable()
    model = peft.prepare_model_for_kbit_training(model)
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

    model = peft.get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

    def compute_metrics(eval_pred):

        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric= evaluate.load("f1")
        accuracy_metric = evaluate.load("accuracy")

        logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
        if llm == "bart":
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        
        print(f"predictions: {predictions}; labels: {labels}")
        
        precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')["precision"]
        recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')["recall"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')["f1"]
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        
        # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores. 
        return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}
        
    # model = model.cuda()

    lr = 3e-05

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


def predict(llm, test_data, task, add_system_message, add_topic, model_path_name, num_labels, device_map, id2labels, system_message):
    if llm == "llama_base":
        model_path_name = "meta-llama/Llama-2-7b-hf"
 
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path_name)
    if task == "stance-classification" or task == "argument-classification":
        model = peft.AutoPeftModelForSequenceClassification.from_pretrained(model_path_name, num_labels=num_labels,torch_dtype=torch.float16, device_map=device_map,)
    elif task == "topic-extraction":
        model = peft.AutoPeftModelForCausalLM.from_pretrained(model_path_name, torch_dtype=torch.float16, device_map=device_map)
    else:
        raise ValueError("Task not found")
    
    if task == "stance-classification" or task == "argument-classification":
        tokenizer.add_special_tokens({'unk_token': '[UNK]'})
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = tokenizer.unk_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    elif task == "topic-extraction":
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError("Task not found")
    
    device = torch.device("cuda")

    
    model.eval()

    predictions = []
    confidences = []
    labels = []
    
    if test_data == "semeval":
        df_test = pd.read_csv('./data/semeval_test.csv')

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
        # df_test = df_test.rename(columns={"human_eval":"target"})



    elif test_data == "arg_spoken_human":
        df_test = pd.read_csv('/data/finetuning/debate_test.csv')
        df_test = df_test[['sentence','human_eval','topic']]
        df_test = df_test.rename(columns={"sentence":"text","human_eval":"target"})


    elif test_data =="gpt_pro_all":
        df_test = pd.read_csv('/data/finetuning/gpt_test.csv',names=["text","class","topic","type"])
        df_test = df_test.rename(columns={"class":"target"})
        df_test = df_test[['text','target','topic','type']]

    elif test_data == "argqual_stance_human":
        df_test = pd.read_csv('/data/finetuning/ibm_arg_test.csv')
        df_test = df_test[['argument','topic','human_eval','pred_topic']]
        df_test["pred_topic"] = df_test["pred_topic"].str.strip()
        df_test["topic"] = df_test.apply(lambda x: x['pred_topic'] if x['pred_topic'] != "No Topic" else x['topic'],axis=1)
        df_test = df_test.rename(columns={"argument":"text","human_eval":"target"})
        df_test = df_test[['text','target','topic']]
        # df_test = df_test.rename(columns={"target":"target"})

    elif test_data == "gpt_pro_all_stance":
        df_test = pd.read_csv('/data/finetuning/gpt_stance_test.csv')
        df_test = df_test[['text','target','topic','type','pred_topic']]
        # drop target column
        df_test = df_test.drop(columns=["target"])
        df_test = df_test.rename(columns={"pred_topic":"target"})
        df_test = df_test[['text','target','topic']]

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
    results = []


    # # Loop through the validation dataset in batches
    for batch in tqdm.tqdm(args):
        with torch.no_grad():
            input_text = tokenizer(batch, padding=True, truncation=True,max_length=2048,return_tensors="pt").to(device)
            if task == "stance-classification" or task == "argument-classification":
                output = model(**input_text)
                logits = output.logits
                predicted_class = torch.argmax(logits, dim=1)
                # Convert logits to a list of predicted labels
                predictions.extend(predicted_class.cpu().tolist())
            elif task == "topic-extraction":
                output = model.generate(**input_text, max_new_tokens=20)
                completion = tokenizer.decode(output[0][input_text['input_ids'].shape[1]:], skip_special_tokens=True)
                # Convert logits to a list of predicted labels
                predictions.extend([completion])

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
        # rouge_metric = evaluate.load("rouge")
        # meteor_metric = evaluate.load("meteor")
        bertscore = evaluate.load("bertscore")
        
        # Compute BLEU score
        bleu = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in labels])
        
        # Compute ROUGE scores
        # rouge_result = rouge_metric.compute(predictions=predictions, references=labels)
        # rouge1 = rouge_result["rouge1"]
        # rouge2 = rouge_result["rouge2"]
        # rougeL = rouge_result["rougeL"]
        
        # Compute METEOR score
        # meteor = meteor_metric.compute(predictions=predictions, references=labels)
        
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
    
    max_seq_length = 2048 
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    device_map = {"": 0}

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    training_data = args.training_data   #"lima" # "full" , "lima"
    task = args.task      #"argument-classification" # "stance-classification" # argument-classification
    add_data = args.add_data
    llm = args.llm #"mistral" # "yi" , # mistral, bart # llama 
    add_system_message = args.add_system_message
    add_topic = args.add_topic
    
                                    
    """
    Stance Datasets:
    ukp_human       gpt_pro_all_stance          argqual_stance_human
    Arg Datasets:
    ukp_human       gpt_pro_all          arg_spoken_human

    """

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
        model_path_name = args.save_model_path + "/" + llm + "-stance-classification"
    elif task == "argument-classification":
        model_path_name = args.save_model_path + "/" + llm + "-argument-detection"
    elif task == "topic-extraction":
        model_path_name = args.save_model_path + "/" + llm + "-topic-extraction"
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

    if args.do_train:
        train(llm, training_data, task, add_data, add_system_message, add_topic, num_epochs, model_path_name, num_labels, id2labels, system_message)

    if args.do_pred:
        predict(llm, test_data, task, add_system_message, add_topic, model_path_name, num_labels, device_map, id2labels, system_message)

    if args.do_eval:
        eval(task, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, default="semeval")
    parser.add_argument("--task", type=str, default="stance-classification")
    parser.add_argument("--add_data", type=bool, default=False)
    parser.add_argument("--llm", type=str, default="phi-3.5")
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
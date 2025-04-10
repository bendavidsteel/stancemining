import datetime
import json
import os

import bertopic.representation
import hydra
import numpy as np
import omegaconf
import polars as pl
import wandb

from experiments import metrics, datasets

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    dataset_name = config['data']['dataset']
    model_name = config['model']['llmmodelname']
    method = config['model']['method']
    # vectopic_method = config['model']['vectopicmethod']
    min_topic_size = config['model']['mintopicsize']

    wandb.init(
        # set the wandb project where this run will be logged
        project="stance-target-topics",

        # track hyperparameters and run metadata
        config={
            'dataset_name': dataset_name,
            'model_name': model_name,
            'method': method,
            # 'vectopic_method': vectopic_method,
            'config': omegaconf.OmegaConf.to_object(config)
        }
    )

    dataset_base_name = os.path.splitext(dataset_name)[0]
    docs_df = datasets.load_dataset(dataset_name)
    docs = docs_df['Text'].to_list()

    start_time = datetime.datetime.now()
    if method in ['topicllm', 'llmtopic']:
        import stancemining
        model = stancemining.StanceMining(
            method=method, 
            llm_method=config.model.llm_method,
            model_lib='transformers', 
            model_name=model_name,
            model_kwargs={'device_map': 'auto'},
            finetune_kwargs=config.wiba
        )

        doc_targets, probs, polarity = model.fit_transform(docs, bertopic_kwargs={'min_topic_size': min_topic_size})
        target_info = model.get_target_info()
    elif method == 'polar':
        from experiments.methods import polar
        model = polar.Polar()
        doc_targets, probs, polarity = model.fit_transform(docs, use_cache=False)
        target_info = model.get_target_info()
    elif method == 'wiba':
        from experiments.methods import wiba
        wiba_model = wiba.Wiba()
        topic_extraction_path = f'./models/wiba/meta-llama-Llama-3.2-1B-Instruct-topic-extraction-vast-ezstance'
        stance_classification_path = f'./models/wiba/meta-llama-Llama-3.2-1B-Instruct-stance-classification-vast-ezstance'
        doc_targets, probs, polarity = wiba_model.fit_transform(docs, config.wiba, argument_detection_path=None, topic_extraction_path=topic_extraction_path, stance_classification_path=stance_classification_path)
        target_info = wiba_model.get_target_info()
    elif method == 'pacte':
        from experiments.methods import pacte
        pacte_model = pacte.PaCTE()
        model_path = './data/pacte/fddb4f5701a1701a3fee0e3e7b8cb75a/ckp/model.pt'
        doc_targets, probs, polarity = pacte_model.fit_transform(docs, model_path=model_path, min_docs=1, polarization='emb_pairwise', use_cache=False)
        target_info = pacte_model.get_target_info()
    elif method == 'annotator':
        from experiments.methods import annotator
        annotator_name = config['model']['AnnotatorName']
        annotation_path = f'./data/{dataset_base_name}_{annotator_name}.csv'
        annotator_model = annotator.Annotator(annotation_path)
        doc_targets, probs, polarity = annotator_model.fit_transform(docs)
        target_info = annotator_model.get_target_info()
    elif method == 'bertopic':
        import bertopic
        from bertopic.representation import KeyBERTInspired
        rep_model = KeyBERTInspired()
        topic_model = bertopic.BERTopic(representation_model=rep_model, calculate_probabilities=True)
        targets, probs = topic_model.fit_transform(docs)
        topics = topic_model.get_topic_info()
        topic_names = [','.join(t.split('_')[1:]) for t in topics['Name'].tolist()]
        doc_targets = [[topic_names[t]] for t in targets]
        polarity = np.zeros((len(docs), len(topics)))
        target_info = pl.DataFrame({
            'noun_phrase': topic_names
        })
    else:
        raise ValueError(f'Unknown method: {method}')
    end_time = datetime.datetime.now()
    total_time = (end_time - start_time).total_seconds()

    all_targets = target_info['noun_phrase'].to_list()
    doc_targets = [[t] if not isinstance(t, list) else t for t in doc_targets]
    output_docs_df = pl.DataFrame({
        'Text': docs,
        'Target': doc_targets,
        'Probs': probs,
        'Polarity': polarity
    })

    # create unique directory
    cur_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    working_dir = f'./data/{cur_time}/runs'
    # update wandb config with working directory
    wandb.config.update({'working_dir': working_dir})
    os.makedirs(working_dir, exist_ok=True)
    output_docs_df.write_parquet(os.path.join(working_dir, f'{dataset_name}_{method}_output.parquet.zstd'), compression='zstd')
    with open(os.path.join(working_dir, 'config.json'), 'w') as f:
        json.dump(omegaconf.OmegaConf.to_object(config), f)
    target_info.write_parquet(os.path.join(working_dir, f'{dataset_name}_{method}_targets.parquet.zstd'), compression='zstd')
    with open(os.path.join(working_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'wall_time': total_time,
            'end_time': end_time.isoformat(),
        }, f)

    # evaluate the stance targets
    if 'Target' in docs_df.columns and 'Stance' in docs_df.columns:
        gold_targets = docs_df['Target'].to_list()
        gold_stances = docs_df['Stance'].to_list()
        label_map = {'favor': 1, 'against': -1, 'neutral': 0}
        gold_stances = [[label_map[s] for s in stances] for stances in gold_stances]
        all_gold_targets = docs_df['Target'].explode().unique().to_list()
        
        # dists, matches = metrics.targets_closest_distance(all_targets, all_gold_targets)
        # TODO extract the f1 scores from the probs, not the given targets 
        targets_f1, targets_precision, targets_recall = metrics.bertscore_f1_targets(doc_targets, gold_targets)
        stance_f1, stance_precision, stance_recall  = metrics.f1_stances(all_targets, all_gold_targets, doc_targets, gold_targets, polarity, gold_stances)
    else:
        dists, matches = None, None
        targets_f1, stance_f1 = None, None

    norm_targets_dist = metrics.normalized_targets_distance(all_targets, docs)
    doc_dist = metrics.document_distance(probs)
    inclusion = metrics.hard_inclusion(doc_targets)
    target_dist = metrics.target_distance(doc_targets, docs)
    stance_variance = metrics.stance_variance(polarity)

    wandb.log({
        # 'targets_closest_distance': dists,
        'targets_f1': targets_f1,
        'targets_precision': targets_precision,
        'targets_recall': targets_recall,
        'stance_f1': stance_f1,
        'stance_precision': stance_precision,
        'stance_recall': stance_recall,
        'normalized_targets_distance': norm_targets_dist,
        'document_distance': doc_dist,
        'hard_inclusion': inclusion,
        'target_distance': target_dist,
        'stance_variance': stance_variance,
        'wall_time': total_time
    })
    wandb.finish()


if __name__ == '__main__':
    main()
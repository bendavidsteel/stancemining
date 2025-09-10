import datetime
import json
import os

import hydra
import omegaconf
import polars as pl
import torch

from stancemining import metrics, datasets, StanceMining

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    dataset_names = ['vast', 'ezstance']
    model_name = config['model']['llmmodelname']
    # vectopic_method = config['model']['vectopicmethod']
    method = 'llmtopic'
    min_topic_size = config['model']['mintopicsize']

    all_num_beam_groups = [2, 3, 5]

    stance_detection_finetune_kwargs = dict(config.wiba.copy())
    target_extraction_finetune_kwargs = dict(config.wiba.copy())

    stance_detection_finetune_kwargs['model_path'] = f"{stance_detection_finetune_kwargs['save_model_path']}{stance_detection_finetune_kwargs['model_name'].replace('/', '-')}-stance-classification-vast-ezstance-head"
    target_extraction_finetune_kwargs['model_path'] = f"{target_extraction_finetune_kwargs['save_model_path']}{target_extraction_finetune_kwargs['model_name'].replace('/', '-')}-topic-extraction-vast-ezstance-beam"

    model_kwargs = {
        'device_map': {'': 0},
        'attn_implementation': 'flash_attention_2',
        'trust_remote_code': True,
        'torch_dtype': torch.bfloat16
    }

    all_data = []

    for dataset_name in dataset_names:
        docs_df = datasets.load_dataset(dataset_name)
        docs = docs_df['Text'].to_list()

        for num_beam_groups in all_num_beam_groups:
            start_time = datetime.datetime.now()
            
            model = StanceMining(
                llm_method=config.model.llm_method,
                model_inference='transformers', 
                model_name=model_name,
                target_extraction_model_kwargs=model_kwargs,
                target_extraction_finetune_kwargs=target_extraction_finetune_kwargs,
                target_extraction_generation_kwargs={'num_return_sequences': num_beam_groups},
                stance_detection_model_kwargs=model_kwargs,
                stance_detection_finetune_kwargs=stance_detection_finetune_kwargs,
                embedding_model_inference='vllm',
                dbscan_deduplicate=False,
                topic_model='bertopic'
            )

            output_docs_df = model.fit_transform(docs, topic_model_kwargs={'min_topic_size': min_topic_size})
            target_info = model.get_target_info()
            
            end_time = datetime.datetime.now()
            total_time = (end_time - start_time).total_seconds()

            all_targets = target_info['Target'].to_list()
            doc_targets = output_docs_df['Targets'].to_list()

            # create unique directory
            cur_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            working_dir = f'./data/runs/{cur_time}'
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

                doc_stances = output_docs_df['Stances'].to_list()
                
                # dists, matches = metrics.targets_closest_distance(all_targets, all_gold_targets)
                # TODO extract the f1 scores from the probs, not the given targets 
                targets_f1, targets_precision, targets_recall = metrics.bertscore_f1_targets(doc_targets, gold_targets)
                stance_f1, stance_precision, stance_recall  = metrics.f1_stances(all_targets, all_gold_targets, doc_targets, gold_targets, doc_stances, gold_stances)
            else:
                dists, matches = None, None
                targets_f1, stance_f1 = None, None

            mean_num_targets = metrics.mean_num_targets(doc_targets)
            stance_variance = metrics.stance_variance(output_docs_df)
            balanced_cluster_size = metrics.balanced_cluster_size(output_docs_df)

            all_data.append({
                'dataset_name': dataset_name,
                'num_beam_groups': num_beam_groups,
                'targets_f1': targets_f1,
                'targets_precision': targets_precision,
                'targets_recall': targets_recall,
                'stance_f1': stance_f1,
                'stance_precision': stance_precision,
                'stance_recall': stance_recall,
                'mean_num_targets': mean_num_targets,
                'stance_variance': stance_variance,
                'balanced_cluster_size': balanced_cluster_size,
                'wall_time': total_time,
            })

        df = pl.from_dicts(all_data)
        df.write_parquet(os.path.join('data', 'num_beam_groups_ablation_results.parquet.zstd'), compression='zstd')

if __name__ == '__main__':
    main()
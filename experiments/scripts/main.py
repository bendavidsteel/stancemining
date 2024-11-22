import os

import hydra
import numpy as np
import polars as pl
import wandb

from experiments import metrics

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    dataset_name = config['data']['datasetname']
    model_name = config['model']['llmmodelname']
    method = config['model']['method']
    vectopic_method = config['model']['vectopicmethod']
    min_topic_size = config['model']['mintopicsize']

    wandb.init(
        # set the wandb project where this run will be logged
        project="stance-target-topics",

        # track hyperparameters and run metadata
        config={
            'dataset_name': dataset_name,
            'model_name': model_name,
            'method': method,
            'vectopic_method': vectopic_method,
        }
    )

    dataset_base_name = os.path.splitext(dataset_name)[0]
    docs_df = pl.read_csv(f'./data/{dataset_name}')
    docs = docs_df['Tweet'].to_list()

    if method == 'vectopic':
        import vectopic as vp
        vector = vp.Vector('favor', 'against')
        model = vp.VectorTopic(
            vector, 
            method=method, 
            model_lib='transformers', 
            model_name=model_name,
            model_kwargs={'device_map': 'auto'}
        )

        doc_targets, probs, polarity = model.fit_transform(docs, bertopic_kwargs={'min_topic_size': min_topic_size})
        topic_info = model.get_topic_info()
        target_info = model.get_target_info()
    elif method == 'polar':
        from experiments.methods import polar
        model = polar.Polar()
        doc_targets, probs, polarity = model.fit_transform(docs)
        target_info = model.get_target_info()
    elif method == 'wiba':
        from experiments.methods import wiba
        wiba_model = wiba.Wiba()
        doc_targets, probs, polarity = wiba_model.fit_transform(docs)
        target_info = wiba_model.get_target_info()
    elif method == 'pacte':
        from experiments.methods import pacte
        pacte_model = pacte.PaCTE()
        model_path = "./data/pacte/1f0d90862b696aa2a805ebc5c2e75ba1/ckp/model.pt"
        doc_targets, probs, polarity = pacte_model.fit_transform(docs, model_path=model_path, min_docs=1, polarization='emb_pairwise')
        target_info = pacte_model.get_target_info()
    elif method == 'annotator':
        from experiments.methods import annotator
        annotator_name = config['model']['AnnotatorName']
        annotation_path = f'./data/{dataset_base_name}_{annotator_name}.csv'
        annotator_model = annotator.Annotator(annotation_path)
        doc_targets, probs, polarity = annotator_model.fit_transform(docs)
        target_info = annotator_model.get_target_info()
    else:
        raise ValueError(f'Unknown method: {method}')

    all_targets = target_info['ngram'].tolist()
    output_docs_df = pl.DataFrame({
        'Tweet': docs,
        'Target': doc_targets,
        'Probs': probs,
        'Polarity': polarity
    })

    target_to_idx = {target: idx for idx, target in enumerate(all_targets)}

    if len(target_to_idx) > 0:
        output_docs_df = output_docs_df.with_columns([
            pl.col('Polarity').arr.get(
                pl.col('Target').replace_strict(target_to_idx)
            ).alias('TargetPolarity')
        ])
        output_docs_df = output_docs_df.with_columns(pl.col('TargetPolarity').replace_strict({-1: 'AGAINST', 0: 'NONE', 1: 'FAVOR'}))
    else:
        output_docs_df = output_docs_df.with_columns(pl.lit('NONE').alias('TargetPolarity'))

    output_docs_df.with_columns([pl.col('Probs').map_elements(lambda l: str(str(l)), pl.String), pl.col('Polarity').map_elements(lambda l: str(str(l)), pl.String)]).write_csv(f'./data/{dataset_base_name}_output.csv')

    # evaluate the stance targets
    if 'Target' in docs_df.columns and 'Stance' in docs_df.columns:
        gold_targets = docs_df['Target'].to_list()
        gold_stances = docs_df['Stance'].to_list()
        label_map = {'FAVOR': 1, 'AGAINST': -1, 'NONE': 0}
        gold_stances = [label_map[stance] for stance in gold_stances]
        all_gold_targets = list(set(gold_targets))
        
        dists, matches = metrics.targets_closest_distance(all_targets, all_gold_targets)
        targets_f1 = metrics.f1_targets(all_targets, all_gold_targets, doc_targets, gold_targets)
        polarity_f1 = metrics.f1_stances(all_targets, all_gold_targets, doc_targets, gold_targets, polarity, gold_stances)
    else:
        dists, matches = None, None
        targets_f1, polarity_f1 = None, None

    norm_targets_dist = metrics.normalized_targets_distance(all_targets, docs)
    doc_dist = metrics.document_distance(probs)
    target_polarities = metrics.target_polarity(polarity)
    inclusion = metrics.hard_inclusion(doc_targets)
    target_dist = metrics.target_distance(doc_targets, docs)

    wandb.log({
        'targets_closest_distance': dists,
        'targets_f1': targets_f1,
        'polarity_f1': polarity_f1,
        'normalized_targets_distance': norm_targets_dist,
        'document_distance': doc_dist,
        'target_polarities': target_polarities,
        'hard_inclusion': inclusion,
        'target_distance': target_dist
    })
    wandb.finish()


if __name__ == '__main__':
    main()
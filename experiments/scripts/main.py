import configparser
import os

import polars as pl
import wandb

from experiments import metrics

def main():
    config = configparser.ConfigParser()
    config.read('./config/config.ini')

    dataset_name = config['data']['DatasetName']
    model_name = config['model']['LLMModelName']
    method = config['model']['Method']
    vectopic_method = config['model']['VectopicMethod']
    min_topic_size = int(config['model']['MinTopicSize'])

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
        all_targets = target_info['ngram'].tolist()
    elif method == 'polar':
        from experiments.methods import polar
        results = polar.polar(docs)
    elif method == 'wiba':
        from experiments.methods import wiba
        results = wiba.wiba(docs)
    elif method == 'pacte':
        from experiments.methods import pacte
        results = pacte.pacte(docs)
    else:
        raise ValueError(f'Unknown method: {method}')

    output_docs_df = pl.DataFrame({
        'Tweet': docs,
        'Target': doc_targets,
        'Probs': probs,
        'Polarity': polarity
    })

    target_to_idx = {target: idx for idx, target in enumerate(all_targets)}

    output_docs_df = output_docs_df.with_columns([
        pl.col('Polarity').arr.get(
            pl.col('Target').replace_strict(target_to_idx)
        ).alias('TargetPolarity')
    ])
    output_docs_df = output_docs_df.with_columns(pl.col('TargetPolarity').replace_strict({-1: 'AGAINST', 0: 'NONE', 1: 'FAVOR'}))

    dataset_base_name = os.path.splitext(dataset_name)[0]
    output_docs_df.with_columns([pl.col('Probs').map_elements(lambda l: str(str(l)), pl.String), pl.col('Polarity').map_elements(lambda l: str(str(l)), pl.String)]).write_csv(f'./data/{dataset_base_name}_output.csv')

    # evaluate the stance targets
    gold_targets = docs_df['Target'].to_list()
    gold_stances = docs_df['Stance'].to_list()
    label_map = {'FAVOR': 1, 'AGAINST': -1, 'NONE': 0}
    gold_stances = [label_map[stance] for stance in gold_stances]
    all_gold_targets = list(set(gold_targets))

    target_info = model.get_target_info()
    all_targets = target_info['ngram'].tolist()

    dists, matches = metrics.targets_closest_distance(all_targets, all_gold_targets)
    targets_f1 = metrics.f1_targets(all_targets, all_gold_targets, doc_targets, gold_targets)
    polarity_f1 = metrics.f1_stances(all_targets, all_gold_targets, doc_targets, gold_targets, polarity, gold_stances)
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
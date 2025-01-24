import hashlib
import itertools
import os

import numpy as np
import polars as pl
import pandas as pd
import wandb

def get_latest_runs():
    api = wandb.Api()
    runs = api.runs("benjamin-steel-projects/stance-target-topics")
    
    dataset_name_map = {
        'vast/vast_test.csv': 'vast',
        'semeval/semeval_test.csv': 'semeval',
        'semeval_test.csv': 'semeval',
        'ezstance/ezstance_test.csv': 'ezstance',
    }

    datasets = ['vast', 'ezstance']
    methods = ['polar', 'pacte', 'wiba', 'llmtopic', 'topicllm']
    
    latest_by_dataset = {}
    for run in runs:
        if run.state == "finished":
            method = run.config.get('method')
            dataset = run.config.get('dataset_name')
            if dataset in dataset_name_map:
                dataset = dataset_name_map[dataset]
            if dataset not in datasets:
                continue
            if method not in methods:
                continue
            if dataset not in latest_by_dataset:
                latest_by_dataset[dataset] = {}
            if method not in latest_by_dataset[dataset]:
                latest_by_dataset[dataset][method] = [run]
            else:
                latest_by_dataset[dataset][method].append(run)

    # get last 3 runs for each method
    num_last = 3
    for dataset in latest_by_dataset:
        for method in latest_by_dataset[dataset]:
            latest_by_dataset[dataset][method] = sorted(latest_by_dataset[dataset][method], key=lambda x: x.created_at, reverse=True)[:num_last]

    return latest_by_dataset


def main():
    latest_runs = get_latest_runs()

    dataset_dfs = {}

    for dataset in latest_runs:
        for method in latest_runs[dataset]:
            for idx, run in enumerate(latest_runs[dataset][method]):
                # get save dir
                working_dir = run.config.get('working_dir')
                if working_dir is None:
                    continue
                dataset = run.config.get('dataset_name')
                method = run.config.get('method')
                target_df = pl.read_parquet(os.path.join(working_dir, f"{dataset}_{method}_targets.parquet.zstd"))
                target_df = target_df.with_row_index()
                output_df = pl.read_parquet(os.path.join(working_dir, f"{dataset}_{method}_output.parquet.zstd"))
                output_df = output_df.join(
                    output_df.explode('Probs')\
                        .with_columns([
                            (pl.col('Probs').cum_count().over('Text') - 1).alias('TargetIdx')
                        ])\
                        .filter(pl.col('Probs') > 0)\
                        .join(target_df, left_on='TargetIdx', right_on='index')\
                        .drop(['TargetIdx'])\
                        .group_by('Text')\
                        .agg(pl.col('noun_phrase')),
                    on='Text',
                    how='left'
                ).with_columns(pl.col('noun_phrase').fill_null([]))
                output_df = output_df.rename({'noun_phrase': f"noun_phrase_{method}_{idx}", 'Probs': f"Probs_{method}_{idx}"})
                if dataset not in dataset_dfs:
                    dataset_dfs[dataset] = output_df.select(['Text', f"noun_phrase_{method}_{idx}", f"Probs_{method}_{idx}"])
                else:
                    dataset_dfs[dataset] = dataset_dfs[dataset].join(output_df.select(['Text', f"noun_phrase_{method}_{idx}", f"Probs_{method}_{idx}"]), on='Text', how='left')

    methods = latest_runs['vast'].keys()
    num_runs = 3

    pairings = list(itertools.combinations(methods, 2))

    target_df = pl.DataFrame()
    cluster_df = pl.DataFrame()
    for dataset in dataset_dfs:
        dataset_df = dataset_dfs[dataset]
        for pairing in pairings:
            for idx in range(num_runs):
                method1 = pairing[0]
                method2 = pairing[1]
                col1 = f"noun_phrase_{method1}_{idx}"
                col2 = f"noun_phrase_{method2}_{idx}"
                target_df = pl.concat([
                    target_df, 
                    dataset_df.select(['Text', col1, col2])\
                        .filter(pl.col(col1) != pl.col(col2))\
                        .rename({col1: 'noun_phrase1', col2: 'noun_phrase2'})\
                        .with_columns([
                            pl.lit(method1).alias('method1'),
                            pl.lit(method2).alias('method2'),
                        ])
                ], how='diagonal_relaxed')

        for method in methods:
            for idx in range(num_runs):
                col = f"Probs_{method}_{idx}"
                cluster_probs = dataset_df[col].to_numpy()
                cluster_pairs = np.dot(cluster_probs, cluster_probs.T)
                
                triads = []
                for i in range(len(dataset_df)):
                    if np.sum(cluster_pairs[i]) == 0:
                        continue
                    # find a document that has a high probability of being in the same cluster
                    cluster_idxs = np.where(cluster_pairs[i] > 0)[0]
                    cluster_idxs = cluster_idxs[cluster_idxs != i]
                    not_cluster_idxs = np.where(cluster_pairs[i] == 0)[0]
                    not_cluster_idxs = not_cluster_idxs[not_cluster_idxs != i]
                    if len(cluster_idxs) == 0:
                        different_cluster_docs = not_cluster_idxs[np.random.choice(len(not_cluster_idxs), size=2, replace=False)]
                        triads.append((i, different_cluster_docs[0], different_cluster_docs[1], 'both_different'))
                    else:
                        same_cluster_doc = cluster_idxs[np.random.randint(len(cluster_idxs))]
                        different_cluster_doc = not_cluster_idxs[np.random.randint(len(not_cluster_idxs))]
                        triads.append((i, same_cluster_doc, different_cluster_doc, 'same_different'))

                cluster_df = pl.concat([cluster_df, pl.DataFrame({
                    'BaseText': [dataset_df['Text'][int(triad[0])] for triad in triads],
                    'DocumentA': [dataset_df['Text'][int(triad[1])] for triad in triads],
                    'DocumentB': [dataset_df['Text'][int(triad[2])] for triad in triads],
                    'chosen': [triad[3] for triad in triads],
                    'method': [method for _ in triads],
                })])

    target_df = target_df.sample(n=1000)
    cluster_df = cluster_df.sample(n=1000)

    method_map = {method: hashlib.shake_128(method.encode()).hexdigest(4) for method in methods}
    target_df = target_df.with_columns([
        pl.col('method1').replace_strict(method_map),
        pl.col('method2').replace_strict(method_map),
    ])
    cluster_df = cluster_df.with_columns(pl.col('method').replace_strict(method_map))

    # randomize order of same cluster and different cluster
    cluster_df = cluster_df.with_columns(pl.lit(np.random.randint(1, 10, size=(len(cluster_df),))).alias('order'))
    cluster_df = cluster_df.with_columns(
        pl.when(pl.col('order') % 2 == 0).then(pl.col('DocumentA')).otherwise(pl.col('DocumentB')).alias('DocumentA_new'),
        pl.when(pl.col('order') % 2 == 0).then(pl.col('DocumentB')).otherwise(pl.col('DocumentA')).alias('DocumentB_new'),
    ).drop(['DocumentA', 'DocumentB']).rename({'DocumentA_new': 'DocumentA', 'DocumentB_new': 'DocumentB'})

    # hash which were chosen
    chosen_map = {k: hashlib.shake_128(k.encode()).hexdigest(4) for k in ['same_different', 'both_different']}
    cluster_df = cluster_df.with_columns(pl.col('chosen').replace_strict(chosen_map))

    target_df = target_df.with_columns([
        pl.col('noun_phrase1').map_elements(lambda s: ", ".join(f'"{l}"' for l in s.to_list()), pl.String),
        pl.col('noun_phrase2').map_elements(lambda s: ", ".join(f'"{l}"' for l in s.to_list()), pl.String)
    ])
    target_df = target_df.with_columns([
        pl.when(pl.col('noun_phrase1') == '').then(pl.lit('No stance targets')).otherwise(pl.col('noun_phrase1')).alias('noun_phrase1'),
        pl.when(pl.col('noun_phrase2') == '').then(pl.lit('No stance targets')).otherwise(pl.col('noun_phrase2')).alias('noun_phrase2'),
    ])

    target_df.write_csv(f"./data/targets_compare.csv")
    cluster_df.write_csv(f"./data/clusters_compare.csv")

if __name__ == '__main__':
    main()
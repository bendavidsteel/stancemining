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

    for dataset in dataset_dfs:
        dataset_df = dataset_dfs[dataset]
        target_df = pl.DataFrame()
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

        cluster_df = pl.DataFrame()
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
                    triads.append((i, cluster_idxs[np.random.randint(len(cluster_idxs))], not_cluster_idxs[np.random.randint(len(not_cluster_idxs))]))

                cluster_df = pl.concat([cluster_df, pl.DataFrame({
                    'BaseText': [dataset_df['Text'][int(triad[0])] for triad in triads],
                    'SameClusterText': [dataset_df['Text'][int(triad[1])] for triad in triads],
                    'DifferentClusterText': [dataset_df['Text'][int(triad[2])] for triad in triads],
                    'method': [method for _ in triads],
                })])

        target_df.write_parquet(f"experiments/data/{dataset}_targets.parquet.zstd", compression='zstd')
        cluster_df.write_parquet(f"experiments/data/{dataset}_clusters.parquet.zstd", compression='zstd')

if __name__ == '__main__':
    main()
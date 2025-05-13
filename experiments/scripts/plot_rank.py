import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

def get_rank(metric_df: pl.DataFrame, metric, methods, metric_order, rank_data, dataset):
    if dataset is not None:
        this_metric_df = metric_df.filter((pl.col('dataset') == dataset) & (pl.col('metric') == metric))
    else:
        this_metric_df = metric_df.filter(pl.col('metric') == metric)
    values = np.array(this_metric_df.select(methods).rows()[0])
    try:
        method_rank_idx = np.argsort(values)
        ranked_methods = np.array(methods)[method_rank_idx]
        if metric_order[metric]:
            ranked_methods = ranked_methods[::-1]
        method_ranks = {m: i for i, m in enumerate(ranked_methods)}
        rank_data.append({'dataset': dataset, 'metric': metric} | method_ranks)
    except:
        rank_data.append({'dataset': dataset, 'metric': metric} | {methods[i]: len(methods) for i in range(len(methods))})

def main():
    methods = ['PaCTE', 'POLAR', 'WIBA', 'LLMTopic']
    datasets = ['vast', 'ezstance']

    method_name_map = {
        'PaCTE': 'PaCTE',
        'POLAR': 'POLAR',
        'WIBA': 'WIBA',
        'LLMTopic': 'EC'
    }
    
    # Define metrics for each table
    supervised_metrics = [
        'targets_f1',
        'targets_precision',
        'targets_recall',
        'stance_f1',
        'stance_precision',
        'stance_recall'
    ]
    
    unsupervised_metrics = [
        'document_distance',
        'mean_num_targets',
        'stance_variance',
        'cluster_size',
        'wall_time'
    ]

    metric_order = {
        'targets_f1': True,
        'targets_precision': True,
        'targets_recall': True,
        'stance_f1': True,
        'stance_precision': True,
        'stance_recall': True,
        'document_distance': False,
        'mean_num_targets': True,
        'stance_variance': True,
        'cluster_size': True,
        'wall_time': False,
        'stance_target_sets': True,
        'stance_target_clusters': True
    }

    metrics = supervised_metrics + unsupervised_metrics

    metric_df = pl.read_parquet('./data/metrics.parquet')

    human_eval_metrics = ['stance_target_sets', 'stance_target_clusters']
    human_eval_metric_data = [
        {
            'dataset': None,
            'metric': 'stance_target_sets',
            'PaCTE': -2.23,
            'POLAR': -2.79,
            'WIBA': 1.51,
            'LLMTopic': 2.23
        },
        {
            'dataset': None,
            'metric': 'stance_target_clusters',
            'PaCTE': 0.19,
            'POLAR': 0.00,
            'WIBA': 0.62,
            'LLMTopic': 0.34
        }
    ]
    metric_df = pl.concat([metric_df, pl.from_records(human_eval_metric_data)])

    rank_data = []
    for dataset in datasets:
        for metric in metrics:
            get_rank(metric_df, metric, methods, metric_order, rank_data, dataset)

    for metric in human_eval_metrics:
        get_rank(metric_df, metric, methods, metric_order, rank_data, None)

    rank_df = pl.DataFrame(rank_data)
    method_names = [method_name_map[m] for m in methods]

    fig, ax = plt.subplots(figsize=(3,2.2))
    ax.bar(method_names, list(rank_df.select(methods).sum().rows()[0]))
    ax.set_xlabel('Method')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_ylabel('Summed Rank Order')
    fig.tight_layout()
    fig.savefig('./figs/rank_order.png')


if __name__ == '__main__':
    main()
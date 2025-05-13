import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

<<<<<<< HEAD
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

=======
>>>>>>> 443600cf2896e381ebbcac0ab77d76fd60a60ed9
def main():
    methods = ['PaCTE', 'POLAR', 'WIBA', 'LLMTopic']
    datasets = ['vast', 'ezstance']

<<<<<<< HEAD
    method_name_map = {
        'PaCTE': 'PaCTE',
        'POLAR': 'POLAR',
        'WIBA': 'WIBA',
        'LLMTopic': 'EC'
=======
    method_names = {
        'PaCTE': 'PaCTE',
        'POLAR': 'POLAR',
        'WIBA': 'WIBA',
        'LLMTopic': 'ExtractCluster'
>>>>>>> 443600cf2896e381ebbcac0ab77d76fd60a60ed9
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
<<<<<<< HEAD
        'wall_time': False,
        'stance_target_sets': True,
        'stance_target_clusters': True
=======
        'wall_time': False
>>>>>>> 443600cf2896e381ebbcac0ab77d76fd60a60ed9
    }

    metrics = supervised_metrics + unsupervised_metrics

    metric_df = pl.read_parquet('./data/metrics.parquet')

<<<<<<< HEAD
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
=======
    human_eval_metrics = [
        {
            'metric': 'stance_target_sets',
            'pacte': -2.23,
            'polar': -2.79,
            'wiba': 1.51,
            'extractcluster': 2.23
        },
        {
            'metric': 'stance_target_clusters',
            'pacte': 0.19,
            'polar': 0.00,
            'wiba': 0.62,
            'extractcluster': 0.34
        }
    ]
>>>>>>> 443600cf2896e381ebbcac0ab77d76fd60a60ed9

    rank_data = []
    for dataset in datasets:
        for metric in metrics:
<<<<<<< HEAD
            get_rank(metric_df, metric, methods, metric_order, rank_data, dataset)

    for metric in human_eval_metrics:
        get_rank(metric_df, metric, methods, metric_order, rank_data, None)

    rank_df = pl.DataFrame(rank_data)
    method_names = [method_name_map[m] for m in methods]

    fig, ax = plt.subplots(figsize=(3,2.2))
    ax.bar(method_names, list(rank_df.select(methods).sum().rows()[0]))
=======
            values = np.array([metric_data[dataset][metric][m] for m in methods])
            metric_rows.append({'dataset': dataset, 'metric': metric} | {methods[i]: values[i] for i in range(len(methods))})
            try:
                method_rank_idx = np.argsort(values)
                ranked_methods = np.array(methods)[method_rank_idx]
                if metric_order[metric]:
                    ranked_methods = ranked_methods[::-1]
                method_ranks = {m: i for i, m in enumerate(ranked_methods)}
                rank_data.append({'dataset': dataset, 'metric': metric} | method_ranks)
            except:
                rank_data.append({'dataset': dataset, 'metric': metric} | {methods[i]: len(methods) for i in range(len(methods))})

    # TODO add human eval ranks

    rank_df = pl.DataFrame(rank_data)

    fig, ax = plt.subplots(figsize=(3,2.5))
    ax.bar(methods, list(rank_df.select(methods).sum().rows()[0]))
>>>>>>> 443600cf2896e381ebbcac0ab77d76fd60a60ed9
    ax.set_xlabel('Method')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_ylabel('Summed Rank Order')
    fig.tight_layout()
    fig.savefig('./figs/rank_order.png')


if __name__ == '__main__':
    main()
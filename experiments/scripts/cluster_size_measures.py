import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm
import wandb

from stancemining import datasets, metrics

def get_latest_runs():
    api = wandb.Api()
    project_name = os.environ['PROJECT_NAME']
    runs = api.runs(project_name)
    
    dataset_name_map = {
        'vast/vast_test.csv': 'vast',
        'semeval/semeval_test.csv': 'semeval',
        'semeval_test.csv': 'semeval',
        'ezstance/ezstance_test.csv': 'ezstance',
    }

    datasets = ['vast', 'ezstance']
    
    latest_by_dataset = {}
    for run in runs:
        if run.state == "finished":
            method = run.config.get('method')
            dataset = run.config.get('dataset_name')
            if dataset in dataset_name_map:
                dataset = dataset_name_map[dataset]
            if dataset not in datasets:
                continue
            if dataset not in latest_by_dataset:
                latest_by_dataset[dataset] = {}
            if method not in latest_by_dataset[dataset]:
                latest_by_dataset[dataset][method] = [run]
            else:
                latest_by_dataset[dataset][method].append(run)

    # get last 3 runs for each method
    num_last = 5
    for dataset in latest_by_dataset:
        for method in latest_by_dataset[dataset]:
            latest_by_dataset[dataset][method] = sorted(latest_by_dataset[dataset][method], key=lambda x: x.created_at, reverse=True)[:num_last]

    return latest_by_dataset

def get_metrics(run):

    working_dir = run.config.get('working_dir')
    if working_dir is None:
        raise ValueError("Working directory not found")
    dataset = run.config.get('dataset_name')
    method = run.config.get('method')
    working_dir_name = os.path.basename(working_dir)
    target_df = pl.read_parquet(os.path.join('data', 'runs', working_dir_name, f"{dataset}_{method}_targets.parquet.zstd"))
    target_df = target_df.with_row_index()
    output_df = pl.read_parquet(os.path.join('data', 'runs', working_dir_name, f"{dataset}_{method}_output.parquet.zstd"))
    output_df = output_df.with_row_index()
    output_df = output_df.join(
        output_df.explode('Probs')\
            .with_columns([
                (pl.col('Probs').cum_count().over('Text') - 1).alias('TargetIdx')
            ])\
            .filter(pl.col('Probs') > 0)\
            .join(target_df, left_on='TargetIdx', right_on='index', how='left')\
            .drop(['TargetIdx'])\
            .filter(pl.col('noun_phrase').is_not_null())\
            .group_by('index')\
            .agg(pl.col('noun_phrase')),
        on='index',
        how='left'
    ).with_columns(pl.col('noun_phrase').fill_null([]))
    
    dataset_df = datasets.load_dataset(dataset)

    # apply same order
    output_df = output_df.sort('Text')
    dataset_df = dataset_df.sort('Text')

    return {
        'shannon_entropy': metrics.shannon_entropy(output_df['Probs'].to_numpy()),
        'cluster_size': metrics.mean_cluster_size_ratio(output_df['Probs'].to_numpy()),
        'normalized_cluster_range': metrics.cluster_normalized_range(output_df['Probs'].to_numpy()),
        'balanced_cluster_size': metrics.balanced_cluster_size(output_df['Probs'].to_numpy()),
    }

def get_final_metrics_from_runs(runs_data):
    """
    Generate two LaTeX tables - one for supervised metrics and one for unsupervised metrics.
    
    Args:
        runs_data (dict): Nested dictionary containing performance metrics
    Returns:
        tuple[str, str]: Tuple of formatted LaTeX table strings (supervised, unsupervised)
    """
    methods = ['LLMTopic', 'WIBA']
    datasets = ['vast']

    
    # Define metrics for each table
    metrics = ['cluster_size', 'shannon_entropy', 'normalized_cluster_range', 'balanced_cluster_size']

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
        'shannon_entropy': True,
        'normalized_cluster_range': True,
        'balanced_cluster_size': True,
    }

    # TODO extract the f1 scores from the probs, not the given targets 

    rank_data = []
    metric_rows = []
        
    # Filter and sort datasets
    runs_data = {dataset: runs_data.get(dataset, {}) for dataset in datasets}

    metric_data = {}
    
    pbar = tqdm(total=len(datasets) * len(methods) * len(metrics), desc="Computing metrics")
    for dataset in runs_data:
        metric_data[dataset] = {}
        datasets_data = runs_data.get(dataset, {})
        for method in methods:
            runs = datasets_data.get(method.lower(), {})

            pbar.update(1)
            metric_values = [get_metrics(run) for run in runs]
            for metric in metrics:
                if metric not in metric_data[dataset]:
                    metric_data[dataset][metric] = {}

                values = [m_vals[metric] for m_vals in metric_values if metric in m_vals]
                values = [value for value in values if value is not None and value != 'NaN']
                mean_value = sum(values) / len(values) if values else None
                metric_data[dataset][metric][method] = mean_value
    
    

    for dataset in runs_data:
        for metric in metrics:
            values = np.array([metric_data[dataset][metric][m] for m in methods])
            metric_rows.append({'dataset': dataset, 'metric': metric} | {methods[i]: values[i] for i in range(len(methods))})
            method_rank_idx = np.argsort(values)
            ranked_methods = np.array(methods)[method_rank_idx]
            if metric_order[metric]:
                ranked_methods = ranked_methods[::-1]
            method_ranks = {m: i for i, m in enumerate(ranked_methods)}
            rank_data.append({'dataset': dataset, 'metric': metric} | method_ranks)
    
    pass

def main():
    latest_runs = get_latest_runs()
    get_final_metrics_from_runs(latest_runs)


if __name__ == '__main__':
    main()
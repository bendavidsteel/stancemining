import os

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm
import wandb

from stancemining import metrics, datasets

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

def get_metrics(run, threshold):

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
        output_df.explode(['Probs', 'Polarity'])\
            .with_columns([
                (pl.col('Probs').cum_count().over('Text') - 1).alias('TargetIdx')
            ])\
            .filter(pl.col('Probs') > 0)\
            .join(target_df.select(['index', 'noun_phrase']), left_on='TargetIdx', right_on='index', how='left')\
            .drop(['TargetIdx'])\
            .filter(pl.col('noun_phrase').is_not_null())\
            .group_by('index')\
            .agg([pl.col('noun_phrase'), pl.col('Polarity').alias('Stances')]),
        on='index',
        how='left'
    ).with_columns(pl.col('noun_phrase').fill_null([]))

    high_var_target_df = target_df.filter(pl.col('var') > target_df['var'].quantile(threshold))
    output_df = output_df.drop(['noun_phrase', 'Stances']).join(
        output_df.explode(['noun_phrase', 'Stances'])\
            .filter(pl.col('noun_phrase').is_in(high_var_target_df['noun_phrase']))\
            .group_by('index')\
            .agg([pl.col('noun_phrase'), pl.col('Stances')])\
            .select(['index', 'noun_phrase', 'Stances']),
        on='index',
        how='left'
    ).with_columns([
        pl.col('noun_phrase').fill_null([]),
        pl.col('Stances').fill_null([])
    ])

    dataset_df = datasets.load_dataset(dataset)

    # apply same order
    output_df = output_df.sort('Text')
    dataset_df = dataset_df.sort('Text')
    
    dataset_df = dataset_df.with_row_index()
    dataset_df = dataset_df.join(
        dataset_df.explode('Stance')\
            .with_columns([
                pl.col('Stance').replace_strict({'favor': 1, 'against': -1, 'neutral': 0}).alias('Polarity')
            ])\
            .group_by('index')\
            .agg(pl.col('Polarity')),
        on='index',
        how='left',
        maintain_order='left'
    )

    stance_f1, stance_p, stance_r = metrics.f1_stances(
        target_df['noun_phrase'].to_list(),
        dataset_df['Target'].explode().unique().to_list(),
        output_df['noun_phrase'].to_list(),
        dataset_df['Target'].to_list(),
        output_df['Polarity'].to_numpy(),
        dataset_df['Polarity'].to_list(),
        threshold=threshold
    )
    target_f1, target_p, target_r = metrics.bertscore_f1_targets(output_df['noun_phrase'].to_list(), dataset_df['Target'].to_list())

    stance_var = metrics.stance_variance(output_df.select(['noun_phrase', 'Stances']))

    balanced_cluster_size = metrics.balanced_cluster_size(output_df)
    mean_num_targets = metrics.mean_num_targets(output_df['noun_phrase'].to_list())

    return {
        'stance_f1': stance_f1,
        'stance_precision': stance_p,
        'stance_recall': stance_r,
        'target_f1': target_f1,
        'target_precision': target_p,
        'target_recall': target_r,
        'stance_variance': stance_var,
        'balanced_cluster_size': balanced_cluster_size,
        'mean_num_targets': mean_num_targets
    }

def get_final_metrics_from_runs(runs_data):
    """
    Generate two LaTeX tables - one for supervised metrics and one for unsupervised metrics.
    
    Args:
        runs_data (dict): Nested dictionary containing performance metrics
    Returns:
        tuple[str, str]: Tuple of formatted LaTeX table strings (supervised, unsupervised)
    """
    method = 'LLMTopic'
    # methods = ['LLMTopic']
    datasets = ['vast', 'ezstance']

    
    # Define metrics for each table

    thresholds = [0.0, 0.5, 0.75, 0.9]

    # Filter and sort datasets
    runs_data = {dataset: runs_data.get(dataset, {}) for dataset in datasets}

    metric_rows = []
    
    pbar = tqdm(total=len(datasets) * len(thresholds) * 5, desc="Computing metrics")
    for dataset in runs_data:
        datasets_data = runs_data.get(dataset, {})
        runs = datasets_data.get(method.lower(), {})

        for threshold in thresholds:
            pbar.update(1)
            for run in runs:
                metrics = get_metrics(run, threshold)
                metric_rows.append({
                    'dataset': dataset,
                    'method': method,
                    'threshold': threshold,
                    **metrics
                })

    metric_df = pl.from_dicts(metric_rows)
    metric_df.write_parquet('./data/stance_var_threshold_sensitivity.parquet.zstd')
    
def main():
    dotenv.load_dotenv()
    latest_runs = get_latest_runs()
    get_final_metrics_from_runs(latest_runs)


if __name__ == '__main__':
    main()
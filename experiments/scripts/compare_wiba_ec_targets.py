import os

import bert_score
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm
import wandb

from stancemining import datasets

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

def get_output(run):
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
    
    return output_df, dataset_df

def get_final_metrics_from_runs(runs_data):
    """
    Generate two LaTeX tables - one for supervised metrics and one for unsupervised metrics.
    
    Args:
        runs_data (dict): Nested dictionary containing performance metrics
    Returns:
        tuple[str, str]: Tuple of formatted LaTeX table strings (supervised, unsupervised)
    """
    methods = ['LLMTopic', 'PaCTE', 'POLAR', 'WIBA']
    datasets = ['vast', 'ezstance']

    
    # Define metrics for each table
    supervised_metrics = [
        'targets_f1',
        'targets_precision',
        'targets_recall',
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
        'wall_time': False
    }

    metric_rows = []
        
    # Filter and sort datasets
    runs_data = {dataset: runs_data.get(dataset, {}) for dataset in datasets}

    metric_data = {}
    
    pbar = tqdm(total=len(datasets), desc="Computing metrics")
    for dataset in runs_data:
        metric_data[dataset] = {}
        datasets_data = runs_data.get(dataset, {})
        wiba_run = datasets_data.get('WIBA'.lower())[0]
        ec_run = datasets_data.get('topicllm'.lower())[0]

        wiba_output, wiba_dataset = get_output(wiba_run)
        ec_output, ec_dataset = get_output(ec_run)

        gold_doc_targets = wiba_dataset['Target'].to_list()
        wiba_doc_targets = wiba_output['noun_phrase'].to_list()
        ec_doc_targets = ec_output['noun_phrase'].to_list()

        ec_f1s, ec_ps, ec_rs = [], [], []
        wiba_f1s, wiba_ps, wiba_rs = [], [], []
        scorer = bert_score.BERTScorer(lang='en')
        for gold_doc_targets, wiba_doc_targets, ec_doc_targets in tqdm(zip(gold_doc_targets, wiba_doc_targets, ec_doc_targets), desc='Calculating BERTScore'):

            if not wiba_doc_targets:
                wiba_precision, wiba_recall, wiba_f1 = 0, 0, 0
            else:
                # For each predicted label, find its highest similarity with any true label
                # For each true label, find its highest similarity with any predicted label  
                wiba_pred_true_scores = scorer.score(wiba_doc_targets + gold_doc_targets, [gold_doc_targets] * len(wiba_doc_targets) + [wiba_doc_targets] * len(gold_doc_targets))[2].tolist()
                wiba_pred_scores = wiba_pred_true_scores[:len(wiba_doc_targets)]
                wiba_true_scores = wiba_pred_true_scores[len(wiba_doc_targets):]

                # Calculate overall precision/recall/F1
                wiba_precision = sum(wiba_pred_scores) / len(wiba_pred_scores) if wiba_pred_scores else 0
                wiba_recall = sum(wiba_true_scores) / len(wiba_true_scores) if wiba_true_scores else 0
                wiba_f1 = 2 * wiba_precision * wiba_recall / (wiba_precision + wiba_recall) if (wiba_precision + wiba_recall) > 0 else 0
            
            if not ec_doc_targets:
                ec_precision, ec_recall, ec_f1 = 0, 0, 0
            else:
                # For each predicted label, find its highest similarity with any true label
                # For each true label, find its highest similarity with any predicted label  
                ec_pred_true_scores = scorer.score(ec_doc_targets + gold_doc_targets, [gold_doc_targets] * len(ec_doc_targets) + [ec_doc_targets] * len(gold_doc_targets))[2].tolist()
                ec_pred_scores = ec_pred_true_scores[:len(ec_doc_targets)]
                ec_true_scores = ec_pred_true_scores[len(ec_doc_targets):]

                # Calculate overall precision/recall/F1
                ec_precision = sum(ec_pred_scores) / len(ec_pred_scores) if ec_pred_scores else 0
                ec_recall = sum(ec_true_scores) / len(ec_true_scores) if ec_true_scores else 0
                ec_f1 = 2 * ec_precision * ec_recall / (ec_precision + ec_recall) if (ec_precision + ec_recall) > 0 else 0
            
            ec_f1s.append(ec_f1)
            ec_ps.append(ec_precision)
            ec_rs.append(ec_recall)

            wiba_f1s.append(wiba_f1)
            wiba_ps.append(wiba_precision)
            wiba_rs.append(wiba_recall)

        wiba_overall_f1 = np.mean(wiba_f1s)
        wiba_overall_precision = np.mean(wiba_ps)
        wiba_overall_recall = np.mean(wiba_rs)
        ec_overall_f1 = np.mean(ec_f1s)
        ec_overall_precision = np.mean(ec_ps)
        ec_overall_recall = np.mean(ec_rs)
        
        pass

    metric_df = pl.DataFrame(metric_rows)
    metric_df.write_parquet('./data/metrics.parquet')

def main():
    latest_runs = get_latest_runs()
    get_final_metrics_from_runs(latest_runs)


if __name__ == '__main__':
    main()
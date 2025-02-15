import os
import polars as pl
from tqdm import tqdm
import wandb

from experiments import metrics, datasets

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
    num_last = 3
    for dataset in latest_by_dataset:
        for method in latest_by_dataset[dataset]:
            latest_by_dataset[dataset][method] = sorted(latest_by_dataset[dataset][method], key=lambda x: x.created_at, reverse=True)[:num_last]

    return latest_by_dataset

def get_metric(run, metric):

    if hasattr(run, 'new_metrics') and metric in run.new_metrics:
        return run.new_metrics[metric]

    if not hasattr(run, 'new_metrics'):
        run.new_metrics = {}

    working_dir = run.config.get('working_dir')
    if working_dir is None:
        raise ValueError("Working directory not found")
    dataset = run.config.get('dataset_name')
    method = run.config.get('method')
    target_df = pl.read_parquet(os.path.join(working_dir, f"{dataset}_{method}_targets.parquet.zstd"))
    target_df = target_df.with_row_index()
    output_df = pl.read_parquet(os.path.join(working_dir, f"{dataset}_{method}_output.parquet.zstd"))
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
    
    if metric in ['targets_f1', 'targets_precision', 'targets_recall']:
        f1, p, r = metrics.bertscore_f1_targets(output_df['noun_phrase'].to_list(), dataset_df['Target'].to_list())
        run.new_metrics['targets_f1'] = f1
        run.new_metrics['targets_precision'] = p
        run.new_metrics['targets_recall'] = r
        if metric == 'targets_f1':
            return f1
        elif metric == 'targets_precision':
            return p
        elif metric == 'targets_recall':
            return r
    elif metric in ['stance_f1', 'stance_precision', 'stance_recall']:
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

        f1, p, r = metrics.f1_stances(
            target_df['noun_phrase'].to_list(),
            dataset_df['Target'].explode().unique().to_list(),
            output_df['noun_phrase'].to_list(),
            dataset_df['Target'].to_list(),
            output_df['Polarity'].to_numpy(),
            dataset_df['Polarity'].to_list()
        )
        run.new_metrics['stance_f1'] = f1
        run.new_metrics['stance_precision'] = p
        run.new_metrics['stance_recall'] = r
        if metric == 'stance_f1':
            return f1
        elif metric == 'stance_precision':
            return p
        elif metric == 'stance_recall':
            return r
    elif metric == 'target_distance':
        return metrics.target_distance(output_df['noun_phrase'], output_df['Text'])
    elif metric == 'stance_variance':
        return metrics.stance_variance(output_df['Polarity'])
    elif metric == 'mean_num_targets':
        return metrics.mean_num_targets(output_df['noun_phrase'])
    elif metric == 'cluster_size':
        return metrics.mean_cluster_size_ratio(output_df['Probs'].to_numpy())
    elif metric == 'cluster_size_std':
        return metrics.mean_cluster_size_std_ratio(output_df['Probs'].to_numpy())
    else:
        raise ValueError(f"Unknown metric: {metric}")

def generate_latex_tables(runs_data):
    """
    Generate two LaTeX tables - one for supervised metrics and one for unsupervised metrics.
    
    Args:
        runs_data (dict): Nested dictionary containing performance metrics
    Returns:
        tuple[str, str]: Tuple of formatted LaTeX table strings (supervised, unsupervised)
    """
    methods = ['PaCTE', 'POLAR', 'WIBA', 'LLMTopic', 'TopicLLM']
    datasets = ['vast', 'ezstance']
    
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
        'normalized_targets_distance',
        'document_distance',
        'hard_inclusion',
        'stance_variance',
        'cluster_size',
        'cluster_size_std',
        'wall_time'
    ]

    remeasure_metrics = supervised_metrics + ['stance_variance', 'cluster_size', 'cluster_size_std']

    # TODO extract the f1 scores from the probs, not the given targets 

    num_runs = 3
    
    # Column headers for supervised metrics
    supervised_header = [
        "\\begin{tabular}{l|ccc|ccc}",
        "\\toprule",
        "\\multicolumn{1}{c}{} & \\multicolumn{3}{c}{\\textbf{Target}} & \\multicolumn{3}{c}{\\textbf{Stance}} \\\\",
        "\\textbf{Method} & \\textbf{F1 ↑} & \\textbf{Prec. ↑} & \\textbf{Recall ↑} & "
        "\\textbf{F1 ↑} & \\textbf{Prec. ↑} & \\textbf{Recall ↑} \\\\",
        "\\midrule"
    ]

    # Column headers for unsupervised metrics
    unsupervised_header = [
        "\\begin{tabular}{l|cccccc|c}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Normalized} & \\textbf{Document} & \\textbf{Hard} & "
        "\\textbf{Stance} & \\textbf{Cluster} & \\textbf{Cluster} & \\textbf{Wall}\\\\",
        "& \\textbf{Target Dist. } & \\textbf{Distance } & \\textbf{Inclusion ↑} & "
        "\\textbf{Variance ↑} & \\textbf{Size ↑} & \\textbf{Size Std ↓} & \\textbf{Time ↓}\\\\",
        "\\midrule"
    ]
        
    def generate_table_content(header_lines, metrics, runs_data):
        latex_table = header_lines.copy()
        
        # Filter and sort datasets
        runs_data = {dataset: runs_data.get(dataset, {}) for dataset in datasets}
        
        pbar = tqdm(total=len(datasets) * len(methods) * len(metrics), desc="Computing metrics")
        for dataset in runs_data:
            latex_table.append(f"\\multicolumn{{{len(metrics)+1}}}{{c}}{{\\textbf{{{dataset.upper()}}}}}\\\\")
            latex_table.append("\\midrule")
            
            datasets_data = runs_data.get(dataset, {})
            for method in methods:
                row_parts = [method]
                runs = datasets_data.get(method.lower(), {})

                for metric in metrics:
                    pbar.update(1)
                    if metric in remeasure_metrics:
                        values = [get_metric(run, metric) for run in runs]
                    else:
                        values = [run.summary.get(metric) for run in runs]
                    values = [value for value in values if value is not None and value != 'NaN']
                    if len(values) != num_runs:
                        row_parts.append("& -")
                        continue
                    mean_value = sum(values) / len(values) if values else None
                    if metric == 'wall_time':
                        row_parts.append(f"& {mean_value:.1f}")
                    else:
                        row_parts.append(f"& {mean_value:.3f}")
                    
                row_parts.append("\\\\")
                latex_table.append(" ".join(row_parts))
            
            latex_table.append("\\midrule")
        
        # Table footer
        latex_table.extend([
            "\\bottomrule",
            "\\end{tabular}"
        ])
        
        return latex_table
    
    # Generate unsupervised table
    unsupervised_table = generate_table_content(unsupervised_header, unsupervised_metrics, runs_data)

    unsupervised_table = "\n".join(unsupervised_table)

    # Write unsupervised metrics table
    with open(os.path.join('.', 'data', 'unsupervised_metrics_table.tex'), 'w') as f:
        f.write(unsupervised_table)

    # Generate supervised table
    supervised_table = generate_table_content(supervised_header, supervised_metrics, runs_data)

    supervised_table = "\n".join(supervised_table)

    # Write supervised metrics table
    with open(os.path.join('.', 'data', 'supervised_metrics_table.tex'), 'w') as f:
        f.write(supervised_table)

def main():
    latest_runs = get_latest_runs()
    generate_latex_tables(latest_runs)


if __name__ == '__main__':
    main()
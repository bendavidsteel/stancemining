import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

def get_values(metric_df, dataset, metric, methods):
    if dataset is not None:
        this_metric_df = metric_df.filter((pl.col('dataset') == dataset) & (pl.col('metric') == metric))
    else:
        this_metric_df = metric_df.filter(pl.col('metric') == metric)
    values = np.array(this_metric_df.select(methods).rows()[0])
    return values

def get_rank(metric_df: pl.DataFrame, metric, methods, metric_order, rank_data, dataset):
    values = get_values(metric_df, dataset, metric, methods)
    try:
        if metric_order[metric]:
            values = np.nan_to_num(values, nan=0)
        else:
            values = np.nan_to_num(values, nan=np.inf)
        method_rank_idx = np.argsort(values)
        ranked_methods = np.array(methods)[method_rank_idx]
        if metric_order[metric]:
            ranked_methods = ranked_methods[::-1]
        method_ranks = {m: i for i, m in enumerate(ranked_methods)}
        rank_data.append({'dataset': dataset, 'metric': metric} | method_ranks)
    except:
        rank_data.append({'dataset': dataset, 'metric': metric} | {methods[i]: len(methods) for i in range(len(methods))})


def main():
    """
    Generate two LaTeX tables - one for supervised metrics and one for unsupervised metrics.
    
    Args:
        runs_data (dict): Nested dictionary containing performance metrics
    Returns:
        tuple[str, str]: Tuple of formatted LaTeX table strings (supervised, unsupervised)
    """
    methods = ['PaCTE', 'POLAR', 'WIBA', 'LLMTopic']
    datasets = ['vast', 'ezstance']

    method_name_map = {
        'PaCTE': 'PaCTE',
        'POLAR': 'POLAR',
        'WIBA': 'WIBA',
        'LLMTopic': 'ExtractCluster'
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
        # 'document_distance',
        'mean_num_targets',
        'stance_variance',
        'cluster_size'
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

    # remeasure_metrics = []
    remeasure_metrics = supervised_metrics + ['mean_num_targets', 'document_distance', 'stance_variance', 'cluster_size']

    # TODO extract the f1 scores from the probs, not the given targets 

    num_runs = 5
    
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
        "\\begin{tabular}{l|ccc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Mean Num.} & "
        "\\textbf{Stance} & \\textbf{Cluster} \\\\",
        "& \\textbf{Targets ↑} & "
        "\\textbf{Variance ↑} & \\textbf{Size ↑} \\\\",
        "\\midrule"
    ]

    rank_data = []
    metric_rows = []
        
    def generate_table_content(header_lines, metrics, metric_df: pl.DataFrame):
        latex_table = header_lines.copy()

        for dataset in datasets:
            for metric in metrics:
                get_rank(metric_df, metric, methods, metric_order, rank_data, dataset)

        for dataset in datasets:
            latex_table.append(f"\\multicolumn{{{len(metrics)+1}}}{{c}}{{\\textbf{{{dataset.upper()}}}}}\\\\")
            latex_table.append("\\midrule")
            for method in methods:
                row_parts = [method_name_map[method]]
                for metric in metrics:
                    mean_value = metric_df.filter((pl.col('dataset') == dataset) & (pl.col('metric') == metric))[method][0]
                    rank = [d for d in rank_data if d['dataset'] == dataset and d['metric'] == metric][0][method]
                    cell = '& '
                    if mean_value is None or np.isnan(mean_value) or mean_value == 0:
                        cell += '-'
                    else:
                        if rank == 0:
                            cell += r'\textbf{'
                        elif rank == 1:
                            cell += r'\underline{'

                        if metric == 'wall_time':
                            cell += f"{mean_value:.1f}"
                        else:
                            cell += f"{mean_value:.3f}"
                        
                        if rank in [0, 1]:
                            cell += '}'

                    row_parts.append(cell)
                    
                row_parts.append("\\\\")
                latex_table.append(" ".join(row_parts))
            
            latex_table.append("\\midrule")
        

        # Table footer
        latex_table.extend([
            "\\bottomrule",
            "\\end{tabular}"
        ])
        
        return latex_table
    
    metric_df = pl.read_parquet('./data/metrics.parquet')

    # Generate unsupervised table
    unsupervised_table = generate_table_content(unsupervised_header, unsupervised_metrics, metric_df)

    unsupervised_table = "\n".join(unsupervised_table)

    # Write unsupervised metrics table
    with open(os.path.join('.', 'data', 'unsupervised_metrics_table.tex'), 'w') as f:
        f.write(unsupervised_table)

    # Generate supervised table
    supervised_table = generate_table_content(supervised_header, supervised_metrics, metric_df)

    supervised_table = "\n".join(supervised_table)

    # Write supervised metrics table
    with open(os.path.join('.', 'data', 'supervised_metrics_table.tex'), 'w') as f:
        f.write(supervised_table)

    

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
        for metric in supervised_metrics + unsupervised_metrics:
            get_rank(metric_df, metric, methods, metric_order, rank_data, dataset)

    for metric in human_eval_metrics:
        get_rank(metric_df, metric, methods, metric_order, rank_data, None)

    rank_df = pl.DataFrame(rank_data)

    method_name_map = {
        'PaCTE': 'PaCTE',
        'POLAR': 'POLAR',
        'WIBA': 'WIBA',
        'LLMTopic': 'EC'
    }

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
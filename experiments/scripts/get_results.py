import os

import pandas as pd
import wandb

def get_latest_runs():
    api = wandb.Api()
    runs = api.runs("benjamin-steel-projects/stance-target-topics")  # Replace with your username
    
    dataset_name_map = {
        'vast/vast_test.csv': 'vast',
        'semeval/semeval_test.csv': 'semeval',
        'semeval_test.csv': 'semeval',
    }

    latest_by_dataset = {}
    for run in runs:
        if run.state == "finished":
            method = run.config.get('method')
            dataset = run.config.get('dataset_name')
            if dataset in dataset_name_map:
                dataset = dataset_name_map[dataset]
            if dataset not in latest_by_dataset:
                latest_by_dataset[dataset] = {}
            if method not in latest_by_dataset[dataset] or run.created_at > latest_by_dataset[dataset][method].created_at:
                latest_by_dataset[dataset][method] = run
    
    return latest_by_dataset

def generate_latex_table(runs_data):
    """
    Generate a LaTeX table comparing different methods and metrics across datasets.
    Each dataset name appears on its own line.
    
    Args:
        runs_data (dict): Nested dictionary containing performance metrics for different
                         datasets and methods.
    
    Returns:
        str: Formatted LaTeX table string
    """
    # Table header
    latex_table = [
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccccc}",
        "\\hline",
        "\\textbf{Dataset} & \\textbf{Method} & \\textbf{Target BERTScore F1} & "
        "\\textbf{Polarity F1} & \\textbf{Normalized Target Distance} & "
        "\\textbf{Document Distance} & \\textbf{Hard Inclusion} & "
        "\\textbf{Target Distance} \\\\",
        "\\hline"
    ]

    # Configuration
    methods = ['PaCTE', 'POLAR', 'WIBA']
    datasets = ['semeval', 'vast', 'ezstance']
    metrics = [
        'targets_f1',
        'polarity_f1', 
        'normalized_targets_distance',
        'document_distance', 
        'hard_inclusion',
        'target_distance'
    ]

    # Generate table rows
    for dataset in datasets:
        if dataset not in runs_data:
            continue
            
        # Add dataset name on its own line
        latex_table.append(f"{dataset} & & & & & & & \\\\")
        
        datasets_data = runs_data.get(dataset, {})
        for method in methods:
            # Create row starting with empty dataset column and method name
            row_parts = ["& ", method]
            
            # Add metric values
            run = datasets_data.get(method.lower(), {})
            if run and hasattr(run, 'summary'):
                for metric in metrics:
                    value = run.summary.get(metric)
                    if value is not None:
                        row_parts.append(f"& {value:.3f}")
                    else:
                        row_parts.append("& -")
            else:
                row_parts.extend(["& -"] * len(metrics))
            
            # Add row ending
            row_parts.append("\\\\")
            
            # Join row parts and add to table
            latex_table.append("".join(row_parts))
        
        # Add horizontal line after each dataset section
        latex_table.append("\\hline")

    # Table footer
    latex_table.extend([
        "\\end{tabular}",
        "}",
    ])

    # Join all lines with proper spacing
    return "\n".join(latex_table)

def main():
    latest_runs = get_latest_runs()
    latex_table = generate_latex_table(latest_runs)
    
    with open(os.path.join('.', 'data', 'comparison_table.tex'), 'w') as f:
        f.write(latex_table)

if __name__ == '__main__':
    main()
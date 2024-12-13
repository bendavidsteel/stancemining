import wandb
import pandas as pd

def get_latest_runs():
    api = wandb.Api()
    runs = api.runs("benjamin-steel-projects/stance-target-topics")  # Replace with your username
    
    latest_by_method = {}
    for run in runs:
        if run.state == "finished":
            method = run.config.get('method')
            if method not in latest_by_method or run.created_at > latest_by_method[method].created_at:
                latest_by_method[method] = run
    
    return latest_by_method

def generate_latex_table(runs_data):
    latex_table = """\\begin{table}[h]
\\centering
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lccccccc}
\\hline
\\textbf{Method} & \\textbf{Target BERTScore F1} & \\textbf{Polarity F1} & \\textbf{Normalized Target Distance} & \\textbf{Document Distance} & \\textbf{Hard Inclusion} & \\textbf{Target Distance} \\\\
\\hline"""

    methods = ['PaCTE', 'POLAR', 'WIBA']
    metrics = ['targets_f1', 'polarity_f1', 'normalized_targets_distance', 
              'document_distance', 'hard_inclusion', 
              'target_distance']

    for method in methods:
        run = runs_data.get(method.lower())
        row = f"\n{method} "
        if run:
            for metric in metrics:
                value = run.summary.get(metric, '')
                row += f"& {value:.3f} " if value != '' else "& - "
        else:
            row += "& - " * len(metrics)
        row += "\\\\"
        latex_table += row

    latex_table += """
\\hline
\\end{tabular}
}
\\caption{Performance comparison across different metrics}
\\label{tab:performance-comparison}
\\end{table}"""

    return latex_table

def main():
    latest_runs = get_latest_runs()
    latex_table = generate_latex_table(latest_runs)
    
    with open('comparison_table.tex', 'w') as f:
        f.write(latex_table)

if __name__ == '__main__':
    main()
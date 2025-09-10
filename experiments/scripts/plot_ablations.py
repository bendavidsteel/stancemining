
import matplotlib.pyplot as plt
import polars as pl

def plot(path, col_name):
    df = pl.read_parquet(path)

    metric_cols = [c for c in df.columns if c not in ['dataset_name', col_name, 'working_dir']]

    if col_name == 'cosine_similarity_threshold':
        metric_cols = [c for c in metric_cols if c != 'wall_time']

    nice_names = {
        'targets_f1': 'Targets F1',
        'targets_precision': 'Targets Precision',
        'targets_recall': 'Targets Recall',
        'stance_f1': 'Stance Retrieval F1',
        'stance_precision': 'Stance Retrieval Precision',
        'stance_recall': 'Stance Retrieval Recall',
        'mean_num_targets': 'Mean Num. Targets',
        'stance_variance': 'Stance Variance',
        'balanced_cluster_size': 'Balanced Cluster Size',
        'vast': 'VAST',
        'ezstance': 'EZ-STANCE',
        'num_beam_groups': 'Num. Beam Groups',
        'cosine_similarity_threshold': 'Cosine Sim. Threshold',
        'wall_time': 'Wall Time',
    }

    datasets = df['dataset_name'].unique().to_list()
    
    for i, dataset_name in enumerate(datasets):
        fig, axes = plt.subplots(nrows=1, ncols=len(metric_cols), figsize=(2 * len(metric_cols), 3))
        axes = axes.flatten()
        dataset_df = df.filter(pl.col('dataset_name') == dataset_name)
        for j, metric_col in enumerate(metric_cols):
            axes[j].plot(dataset_df[col_name], dataset_df[metric_col], marker='o')
            # axes[i * len(metric_cols) + j].set_title(nice_names[metric_col])
            axes[j].set_xlabel(nice_names[col_name])
            axes[j].set_ylabel(nice_names[metric_col])

        fig.tight_layout()
        fig.savefig(f'./figs/{dataset_name}_{col_name}_ablation.png')

def main():
    plot('./data/num_beam_groups_ablation_results.parquet.zstd', 'num_beam_groups')
    plot('./data/cosine_sim_ablation_results.parquet.zstd', 'cosine_similarity_threshold')

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import polars as pl

def pct_change(mean_df, threshold):
    change_metric_cols = ['stance_variance', 'stance_f1', 'target_f1', 'balanced_cluster_size', 'mean_num_targets']
    pct_change_df = mean_df.group_by('dataset')\
        .agg([(pl.col(m).filter(pl.col('threshold') == threshold).first() / pl.col(m).filter(pl.col('threshold') == 0.0).first() - 1).alias(f'{m}_pct_change') for m in change_metric_cols])
    # get average across datasets
    avg_pct_change = {m: pct_change_df.select(pl.col(f'{m}_pct_change').mean()).item() for m in change_metric_cols}
    print(f"Average percentage change from threshold 0.0 to {threshold}:")
    for m, v in avg_pct_change.items():
        print(f"{m}: {v:.2%}")

def main():

    metric_order = {
        'target_f1': True,
        'target_precision': True,
        'target_recall': True,
        'stance_f1': True,
        'stance_precision': True,
        'stance_recall': True,
        'document_distance': False,
        'mean_num_targets': True,
        'stance_variance': True,
        'cluster_size': True,
        'mean_cluster_size': True,
        'wall_time': False,
        'balanced_cluster_size': True,
    }

    df = pl.read_parquet('./data/stance_var_threshold_sensitivity.parquet.zstd')

    type_cols = ['dataset', 'method', 'threshold']
    metric_cols = [c for c in df.columns if c not in type_cols]
    mean_df = df.group_by(type_cols).agg([pl.col(c).mean() for c in metric_cols])

    datasets = mean_df['dataset'].unique().to_list()
    fig, axes = plt.subplots(len(datasets), len(metric_cols), figsize=(5 * len(metric_cols), 5 * len(datasets)), sharex=True)
    for i, dataset in enumerate(datasets):
        dataset_df = mean_df.filter(pl.col('dataset') == dataset).sort('threshold')

        

        for j, metric in enumerate(metric_cols):
            axes[i, j].plot(dataset_df['threshold'], dataset_df[metric], label=metric)
            axes[i, j].set_title(f'{dataset} - {metric}')
    
    # determine average increase/decrease for stance variance, the two F1s, balanced cluster size, and mean number of targets
    pct_change(mean_df, threshold=0.75)
    pct_change(mean_df, threshold=0.9)

    dataset_rank_df = dataset_df.with_columns([pl.col(c).rank(method='average', descending=metric_order[c]).over('dataset').alias(c) for c in metric_cols])\
        .with_columns(pl.sum_horizontal([pl.col(c) for c in metric_cols]).alias('rank_sum'))\
        .group_by('threshold')\
        .agg(pl.col('rank_sum').sum().alias('total_rank'))\
        .sort('total_rank', descending=True)

    fig.tight_layout()
    fig.savefig('./figs/stance_var_threshold_sensitivity.png')

if __name__ == "__main__":
    main()
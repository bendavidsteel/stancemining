import matplotlib.pyplot as plt
import polars as pl

def main():
    method_name_map = {
        'PaCTE': 'PaCTE',
        'POLAR': 'POLAR',
        'WIBA': 'WIBA',
        'LLMTopic': 'EC'
    }

    metric_name_map = {
        'stance_f1': 'F1 Score',
        'stance_precision': 'Precision',
        'stance_recall': 'Recall'
    }

    dataset_name_map = {
        'vast': 'VAST',
        'ezstance': 'EZ-STANCE'
    }

    metric_df = pl.read_parquet('./data/stance_retrieval_threshold_sensitivity.parquet.zstd')
    datasets = metric_df['dataset'].unique().to_list()
    metrics = metric_df['metric'].unique().to_list()
    
    # Get all unique methods to create consistent color mapping
    all_methods = metric_df['method'].unique().to_list()
    colors = plt.cm.tab10(range(len(all_methods)))
    method_colors = dict(zip(all_methods, colors))
    
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(10, 6))
    
    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            dataset_df = metric_df.filter((pl.col('dataset') == dataset) & (pl.col('metric') == metric))
            
            # Use consistent ordering for legend
            available_methods = dataset_df['method'].unique().to_list()
            for method in all_methods:
                if method not in available_methods:
                    continue
                method_df = dataset_df.filter(pl.col('method') == method)
                quantile_df = method_df.group_by(['dataset', 'metric', 'method', 'threshold'])\
                    .agg([
                        pl.col('value').quantile(0.25).alias('q1'),
                        pl.col('value').quantile(0.5).alias('q2'),
                        pl.col('value').quantile(0.75).alias('q3')
                    ])\
                    .sort('threshold')
                
                color = method_colors[method]
                ax.plot(quantile_df['threshold'], quantile_df['q2'], label=method_name_map[method], color=color)
                ax.fill_between(quantile_df['threshold'], quantile_df['q1'], quantile_df['q3'], 
                               alpha=0.2, color=color)
            
            ax.set_title(f"{dataset_name_map[dataset]} - {metric_name_map[metric]}")
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Metric Value')
            ax.legend(loc='lower left')
    
    fig.tight_layout()
    fig.savefig('./figs/stance_retrieval_threshold_sensitivity.png', dpi=300)

if __name__ == "__main__":
    main()
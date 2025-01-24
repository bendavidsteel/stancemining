import hashlib

import choix
import polars as pl

def calculate_pairwise_scores(df: pl.DataFrame):
    # Initialize ratings
    methods = set(df['method1'].unique().to_list() + df['method2'].unique().to_list())
    
    # TODO find way to incorporate ties
    df = df.with_columns(pl.when(pl.col('better_method') == pl.col('method1')).then(pl.col('method2')).otherwise(pl.col('method1')).alias('loser'))

    method_map = {method: idx for idx, method in enumerate(methods)}
    df = df.with_columns([
        pl.col('better_method').replace_strict(method_map),
        pl.col('loser').replace_strict(method_map)
    ])
    pairings = df.select(pl.concat_list(['better_method', 'loser'])).to_series().to_list()

    scores = choix.lsr_pairwise(len(methods), pairings)

    return scores

def get_method_stats(df: pl.DataFrame):
    stats = {}
    for method in set(df['method1'].unique().to_list() + df['method2'].unique().to_list()):
        total = df.filter((pl.col('method1') == method) | (pl.col('method2') == method)).height
        failures = df.filter(
            ((pl.col('method1') == method) | (pl.col('method2') == method)) & 
            pl.col('better_method').is_null()
        ).height
        
        stats[method] = {
            'failure_rate': failures / total,
            'total_comparisons': total
        }
    
    return stats

def main():
    cluster_df = pl.read_csv('./data/cluster_annotations.csv')
    target_df = pl.read_csv('./data/target_annotations.csv')

    chosen_map = {hashlib.shake_128(k.encode()).hexdigest(4): k for k in ['same_different', 'both_different']}
    methods = ['pacte', 'polar', 'wiba', 'llmtopic', 'topicllm']
    method_map = {hashlib.shake_128(method.encode()).hexdigest(4): method for method in methods}

    cluster_df = cluster_df.with_columns([
        pl.col('chosen').replace_strict(chosen_map),
        pl.col('method').replace_strict(method_map)
    ])

    target_df = target_df.with_columns([
        pl.col('method1').replace_strict(method_map),
        pl.col('method2').replace_strict(method_map),
    ])
    target_df = target_df.with_columns([
        pl.when(pl.col('better_target') == 'Target 1 is better')\
            .then(pl.col('method1'))\
            .when(pl.col('better_target') == 'Target 2 is better')\
            .then(pl.col('method2'))\
            .otherwise(pl.lit(None)).alias('better_method'),
    ])

    # Calculate Elo scores
    elo_scores = calculate_pairwise_scores(target_df)
    print(elo_scores)
    stats = get_method_stats(target_df)
    print(stats)

    # swap documents back to original order
    cluster_df = cluster_df.with_columns([
        pl.when(pl.col('order') % 2 == 0).then(pl.col('DocumentA')).otherwise(pl.col('DocumentB')).alias('DocumentA_new'),
        pl.when(pl.col('order') % 2 == 0).then(pl.col('DocumentB')).otherwise(pl.col('DocumentA')).alias('DocumentB_new'),
    ]).drop(['DocumentA', 'DocumentB']).rename({'DocumentA_new': 'DocumentA', 'DocumentB_new': 'DocumentB'})

    cluster_df = cluster_df.with_columns(
        pl.when((pl.col('chosen') == 'same_different') & (pl.col('same_cluster') == 'Document A'))\
            .then(pl.lit('agree'))\
            .when((pl.col('chosen') == 'same_different') & (pl.col('same_cluster') != 'Document A'))\
            .then(pl.lit('disagree'))\
            .when((pl.col('chosen') == 'both_different') & (pl.col('same_cluster') == 'Neither'))\
            .then(pl.lit('agree'))\
            .when((pl.col('chosen') == 'both_different') & (pl.col('same_cluster') != 'Neither'))\
            .then(pl.lit('disagree'))\
            .otherwise(pl.lit(None)).alias('agreement')
    )

    cluster_score_df = cluster_df.with_columns(pl.col('agreement').replace_strict({'agree': 1, 'disagree': 0}).alias('score'))\
        .group_by('method')\
        .agg((pl.col('score').sum() / pl.col('score').count()).alias('win_rate'))
    print(cluster_score_df)

if __name__ == '__main__':
    main()
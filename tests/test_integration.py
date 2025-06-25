import polars as pl
import torch

import stancemining
import stancemining.plot

def test_trend_integration():
    doc_df = pl.read_csv('./tests/data/active_bluesky_sample.csv')
    doc_df = doc_df.with_columns(pl.col('created_at').str.to_datetime())

    model = stancemining.StanceMining(model_name='Qwen/Qwen3-0.6B', verbose=True)
    document_df = model.fit_transform(doc_df, text_column='text', parent_text_column='parent_text')
    target_info_df = model.get_target_info()
    trend_df, gp_df = stancemining.get_stance_trends(document_df, target_info_df, time_column='created_at', filter_columns=['author'])
    plot_trends(trend_df)

def test_integration():
    doc_df = pl.read_csv('./tests/data/active_bluesky_sample.csv')
    doc_df = doc_df.with_columns(pl.col('created_at').str.to_datetime())

    model = stancemining.StanceMining(model_name='Qwen/Qwen3-0.6B', verbose=True)
    document_df = model.fit_transform(doc_df, text_column='text', parent_text_column='parent_text')
    fig = stancemining.plot.plot_semantic_map(document_df)
    fig.savefig('./tests/semantic_map.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    test_integration()
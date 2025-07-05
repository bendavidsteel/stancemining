import polars as pl

import stancemining
import stancemining.estimate
import stancemining.plot

def main():
    # contains columns 'text' and 'created_at', where 'created_at' is the timestamp of the document
    doc_df = pl.read_csv('./tests/data/active_bluesky_sample.csv')

    model = stancemining.StanceMining(model_name='Qwen/Qwen3-1.7B', verbose=True)
    document_stance_df = model.fit_transform(doc_df, text_column='text', parent_text_column='parent_text')
    trends_df, _ = stancemining.estimate.infer_stance_trends_for_all_targets(
        document_stance_df,
        time_column='created_at',
    )
    fig = stancemining.plot.plot_trend_map(document_stance_df, trends_df)
    fig.savefig('./trends.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
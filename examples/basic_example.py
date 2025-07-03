import polars as pl

import stancemining

def main():
    doc_df = pl.read_csv('./tests/data/active_bluesky_sample.csv')
    docs = doc_df['text'].to_list()

    model = stancemining.StanceMining(model_name='Qwen/Qwen3-1.7B', verbose=True)
    document_df = model.fit_transform(docs, text_column='text')
    fig = stancemining.plot.plot_semantic_map(document_df)
    fig.savefig('./semantic_map.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
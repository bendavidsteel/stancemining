import polars as pl

import stancemining
import stancemining.plot

def main():
    # contains columns 'text' and 'parent_text', where 'parent_text' is the text of the parent document
    # and 'text' is the text of the child document
    doc_df = pl.read_csv('./tests/data/active_bluesky_sample.csv')

    model = stancemining.StanceMining(model_name='Qwen/Qwen3-1.7B', verbose=True)
    document_df = model.fit_transform(doc_df, text_column='text', parent_text_column='parent_text')
    fig = stancemining.plot.plot_semantic_map(document_df)
    fig.savefig('./semantic_map.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
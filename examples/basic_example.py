import time

import polars as pl

import stancemining
import stancemining.plot

def main():
    doc_df = pl.read_csv('./tests/data/active_bluesky_sample.csv')
    docs = doc_df['text'].to_list()

    start_time = time.time()

    model = stancemining.StanceMining(model_name='Qwen/Qwen3-1.7B', verbose=True)
    document_df = model.fit_transform(docs)
    fig = stancemining.plot.plot_semantic_map(document_df)
    fig.savefig('./semantic_map.png', dpi=300, bbox_inches='tight')

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Number of documents: {len(docs)}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
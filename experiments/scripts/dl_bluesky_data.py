from datasets import load_dataset
import polars as pl
import fast_langdetect
from tqdm import tqdm

def detect(t: str):
    if len(t) >= 100:
        t = t[:100]
    try:
        return fast_langdetect.detect(t)['lang']
    except:
        return None

def main():
    ds = load_dataset("alpindale/two-million-bluesky-posts")
    df = ds['train'].to_polars()

    active_df = df.filter(pl.col('author').is_in(df.group_by('author').len().filter(pl.col('len') > 10).sample(100)['author'].to_list()))

    active_df = active_df.filter(pl.col('text').str.contains('[A-Za-z]+')).with_columns(pl.col('text').str.replace('\n', ' '))
    texts = active_df['text'].to_list()
    langs = [detect(t) for t in tqdm(texts)]
    active_df = active_df.with_columns(pl.Series('lang', langs))
    active_df = active_df.filter(pl.col('lang') == 'en')

    # get parent texts
    active_df = active_df.join(df.select(['uri', pl.col('text').alias('parent_text')]), left_on='reply_to', right_on='uri', how='left')

    active_df.select(['text', 'created_at', 'author', 'parent_text']).write_csv('./data/active_bluesky_sample.csv')

if __name__ == '__main__':
    main()
import pandas as pd
import tqdm

from manual_test import add_newlines

def main():
    articles_df = pd.read_csv('./data/fakenewskdd2020/test.csv', sep='\t')
    articles_df['text'] = articles_df['text'].map(add_newlines)
    articles_df.to_csv('./data/fakenewskdd2020/formatted_fakenews.csv', index=False)

if __name__ == '__main__':
    main()
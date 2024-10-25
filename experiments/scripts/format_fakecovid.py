import pandas as pd
import tqdm

from manual_test import add_newlines

def main():
    articles_df = pd.read_csv('./data/FakeCovid_July2020.csv')
    articles_df = articles_df[articles_df['lang'] == 'en']
    articles_df = articles_df[['source_title', 'content_text']]
    articles_df['content_text'] = articles_df['content_text'].map(add_newlines)
    articles_df.to_csv('./data/formatted_fakecovid.csv', index=False)

if __name__ == '__main__':
    main()
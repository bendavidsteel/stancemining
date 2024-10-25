import newspaper
import pandas as pd
import tqdm

from manual_test import add_newlines

def main():
    similarity_df = pd.read_csv('./data/zenodo_release_data.csv')

    urls_1 = similarity_df['content.url1']
    urls_2 = similarity_df['content.url2']

    urls = pd.concat([urls_1, urls_2], axis=0).drop_duplicates()

    urls = urls.dropna().tolist()
    urls = urls[:10]

    articles = []
    for url in tqdm.tqdm(urls):
        article = newspaper.article(url)
        articles.append({'url': url, 'title': article.title, 'text': article.text})

    articles_df = pd.DataFrame(articles)
    articles_df['text'] = articles_df['text'].map(add_newlines)
    articles_df['title'] = articles_df['title'].map(add_newlines)
    articles_df.to_csv('./data/frame_articles.csv', index=False)


if __name__ == '__main__':
    main()
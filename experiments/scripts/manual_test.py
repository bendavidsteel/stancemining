import polars as pl

from mining import reddit

def add_newlines(text):
    last_newline = 0
    i = 0
    while i < len(text) - 1:
        if text[i] == '\n':
            last_newline = i
        elif i - last_newline > 50 and text[i] == ' ':
            text = text[:i] + '\n' + text[i+1:]
            last_newline = i
        elif i - last_newline > 60 and text[i] != ' ':
            text = text[:i] + '-\n' + text[i:]
            last_newline = i
        i += 1
    return text


def main():
    comment_df = pl.read_parquet('./data/canada_comments_filtered_2022-07.parquet.gzip')
    submission_df = pl.read_parquet('./data/canada_submissions_filtered_2022-07.parquet.gzip')

    comment_df, submission_df = reddit.get_parents(comment_df, submission_df)

    # sample 100 comments
    comment_df = comment_df.sample(30)

    # write out to CSV
    comment_df = comment_df.to_pandas()
    comment_df['text'] = comment_df[['title', 'body_parent', 'body']].apply(lambda x: f'Post: "{x[0]}",\nParent Comment: "{x[1]}",\nComment: "{x[2]}"', axis=1)
    comment_df['text'] = comment_df['text'].map(add_newlines)
    comment_df[['text']].to_csv('./data/sample_comments.csv', index=False)

if __name__ == '__main__':
    main()
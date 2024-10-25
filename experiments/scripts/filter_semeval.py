import polars as pl

def main():
    train_path = './data/stancedataset/StanceDataset/train.csv'

    df = pl.read_csv(train_path)

    unique_targets = df.select('Target').unique()['Target'].to_list()

    # sample 5 tweets for each target
    samples = []
    for target in unique_targets:
        samples.append(df.filter(df['Target'] == target).sample(6))
    sample_df = pl.concat(samples)
    # shuffle order
    sample_df = sample_df.sample(fraction=1.0, shuffle=True)

    sample_df.select(['Tweet']).write_csv('./data/semeval_sample.csv')
    sample_df.write_csv('./data/semeval_sample_labelled.csv')

if __name__ == '__main__':
    main()
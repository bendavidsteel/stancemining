import os

import pandas as pd

import vectopic as vp

def main():

    docs_df = pd.read_parquet('./data/canada_comments_filtered_2022-07.parquet.gzip')
    docs = docs_df['body'].tolist()

    vector = vp.Vector('favor', 'against')
    model = vp.VectorTopic(vector)

    topics, proj = model.fit_transform(docs)

    topic_info_df = model.topic_info()

if __name__ == '__main__':
    main()
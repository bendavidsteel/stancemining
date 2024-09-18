import os

import pandas as pd

import vectopic as vp

def main():

    docs_df = pd.read_csv('./data/sample_comments.csv')
    docs = docs_df['text'].tolist()

    llm_config = {
        'num_gpu_layers': 10
    }

    vector = vp.Vector('favor', 'against')
    model = vp.VectorTopic(vector, method='llm', llm_lib='ctransformers', llm_config=llm_config)

    topics, proj = model.fit_transform(docs)

    topic_info_df = model.topic_info()

if __name__ == '__main__':
    main()
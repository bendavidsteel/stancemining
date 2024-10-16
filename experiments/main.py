import os

import pandas as pd

import vectopic as vp

def main():

    docs_df = pd.read_csv('./data/sample_comments.csv')
    docs = docs_df['text'].tolist()

    vector = vp.Vector('favor', 'against')
    model = vp.VectorTopic(
        vector, 
        method='llm', 
        model_lib='transformers', 
        model_name='microsoft/Phi-3.5-mini-instruct',
        model_kwargs={'device_map': 'auto'}
    )

    topic_info = model.fit_transform(docs, bertopic_kwargs={'min_topic_size': 4})

    print(topic_info)

if __name__ == '__main__':
    main()
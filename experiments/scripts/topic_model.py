import bertopic
from bertopic.representation import KeyBERTInspired
import polars as pl

from experiments import datasets

def main():
    dataset_name = 'vast'
    docs_df = datasets.load_dataset(dataset_name)
    docs = docs_df['Text'].to_list()

    rep_model = KeyBERTInspired()
    topic_model = bertopic.BERTopic(representation_model=rep_model, calculate_probabilities=True)
    targets, probs = topic_model.fit_transform(docs)
    topics = topic_model.get_topic_info()

    docs_df = docs_df.with_columns(pl.Series(name='Topic', values=targets))

    hierarchical_topics = topic_model.hierarchical_topics(docs)
    tree = topic_model.get_topic_tree(hierarchical_topics)

    with open('./data/topics/hierarchical_tree.txt', 'w') as f:
        f.write(tree)

    pass

if __name__ == '__main__':
    main()
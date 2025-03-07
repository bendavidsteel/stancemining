import itertools
import random

import numpy as np
import polars as pl
import pytest
from scipy.stats import dirichlet as scipy_dirichlet

from stancemining import StanceMining
from experiments import metrics

class MockTopicModel:
    def __init__(self, num_topics, **kwargs):
        self.num_topics = num_topics

    def fit_transform(self, docs, **kwargs):
        self.c_tf_idf_ = None
        self.topic_representations_ = None
        probs = scipy_dirichlet([1] * (self.num_topics + 1)).rvs(len(docs))
        topics = []
        for prob in probs:
            topics.append(np.argmax(prob) - 1)

        return topics, probs[:,1:] / probs[:,1:].sum(axis=1)[:,None]
    
    def get_topic_info(self):
        return pl.DataFrame({
            "Topic": list(range(self.num_topics)),
            "Representative_Docs": [["doc"] * self.nr_repr_docs] * self.num_topics,
            "Representation": [['keyword'] * 5] * self.num_topics
        })
    
    def _extract_representative_docs(self, tf_idf, documents, topic_reps, nr_samples=5, nr_repr_docs=5):
        self.nr_repr_docs = nr_repr_docs
        return pl.DataFrame({
            "Document": ["doc1", "doc2", "doc3", "doc4", "doc5"]
        }), None, None, None

class MockGenerator:
    def __init__(self, **kwargs):
        pass
    
    def generate(self, prompt, num_samples=1, **kwargs):
        return ["output"] * num_samples


class MockStanceMining(StanceMining):
    def __init__(self, *args, num_targets=3, targets=[], num_topics=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_targets = num_targets
        self.targets = targets
        self.num_topics = num_topics

    def _get_base_topic_model(self, bertopic_kwargs):
        return MockTopicModel(self.num_topics)
    
    def _get_generator(self):
        return MockGenerator()
    
    def _ask_llm_differences(self, docs):
        if not self.targets:
            return [f"diff_{i}" for i in range(self.num_targets)]
        else:
            return self.targets
    
    def _ask_llm_class(self, ngram, docs):
        classes = [self.vector.sent_a, self.vector.sent_b, self.vector.neutral]
        res = []
        for _ in docs:
            res.append(random.choice(classes))
        return res

def test_filter_targets():
    num_docs = 10
    targets = [[f'target_{j}' for j in range(3)] for i in range(num_docs)]
    df = pl.DataFrame({'Targets': targets})
    miner = MockStanceMining()
    df = df.with_columns(miner._filter_similar_phrases_fast(df['Targets']))
    assert len(df) == len(targets)


if __name__ == '__main__':
    test_filter_targets()
    print("filter_targets passed")
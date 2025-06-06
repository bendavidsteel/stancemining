import itertools
import random

import numpy as np
import polars as pl
import pytest
from scipy.stats import dirichlet as scipy_dirichlet

from stancemining.main import StanceMining
from stancemining import metrics

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

def test_topic_llm_fit_transform():
    num_docs = 10
    docs = [f"doc_{i}" for i in range(num_docs)]
    num_targets = 3
    vectopic = MockStanceMining()
    documents = vectopic._topic_llm_fit_transform(docs)
    assert hasattr(vectopic, "topic_info")
    assert hasattr(vectopic, "target_info")
    assert len(vectopic.target_info) == num_targets
    assert len(documents) == len(docs)

def test_llm_topic_fit_transform():
    num_docs = 10
    docs = [f"doc_{i}" for i in range(num_docs)]
    num_targets = 3
    vectopic = MockStanceMining()
    documents = vectopic._llm_topic_fit_transform(docs)
    assert hasattr(vectopic, "topic_info")
    assert hasattr(vectopic, "target_info")
    assert len(vectopic.target_info) == num_targets
    assert len(documents) == len(docs)

def test_filter_targets():
    num_docs = 10
    targets = [[f'target_{j}' for j in range(3)] for i in range(num_docs)]
    df = pl.DataFrame({'Targets': targets})
    miner = MockStanceMining()
    df = df.with_columns(miner._filter_similar_phrases_fast(df['Targets']))
    assert len(df) == len(targets)


def test_get_targets_probs_polarity():
    num_topics = 3
    num_docs = 10
    docs = [f"doc_{i}" for i in range(num_docs)]
    vectopic = MockStanceMining(vector, num_targets=num_targets, num_topics=num_topics)
    documents = vectopic._topic_llm_fit_transform(docs)
    targets, probs, polarity = vectopic._get_targets_probs_polarity(documents)
    assert len(targets) == len(documents)
    assert isinstance(targets, list)
    assert isinstance(probs, np.ndarray)
    assert isinstance(polarity, np.ndarray)
    assert probs.shape[0] == len(documents)
    assert probs.shape[1] == num_targets
    assert polarity.shape[0] == len(documents)
    assert polarity.shape[1] == num_targets


@pytest.mark.parametrize("gold_targets, extracted_targets", [
    (['labradors', 'pet cats', 'farm animals'], ['dogs', 'cats', 'cows']),
    (['labradors', 'pet cats', 'farm animals', 'spaceships'], ['dogs', 'cats', 'cows'])
])
def test_target_match_and_supervised_metrics(gold_targets, extracted_targets):
    num_docs = 20
    gold_doc_targets = []
    doc_targets = []
    polarity = np.zeros((num_docs, len(extracted_targets)))
    gold_polarity = []
    cycle_len = len(extracted_targets) + 2
    for i in range(num_docs):
        if i % cycle_len == 0:
            doc_target = random.sample(extracted_targets, 2)
            doc_targets.append(doc_target)
            for t in doc_target:
                polarity[i, extracted_targets.index(t)] = np.random.choice([-1, 0, 1])

            gold_doc_targets.append(random.sample(gold_targets, 2))
            gold_p = []
            for t in gold_doc_targets:
                gold_p.append(np.random.choice([-1, 0, 1]))
            gold_polarity.append(gold_p)

        elif i % cycle_len == 1:
            doc_targets.append([])
            gold_doc_targets.append([])
            gold_polarity.append([])
        else:
            doc_target = extracted_targets[i % len(extracted_targets)]
            doc_targets.append([doc_target])
            polarity[i, extracted_targets.index(doc_target)] = np.random.choice([-1, 0, 1])

            gold_target = gold_targets[i % len(extracted_targets)]
            gold_doc_targets.append([gold_target])
            gold_polarity.append([np.random.choice([-1, 0, 1])])


    gold_stances = pd.DataFrame({
        "Document": [f"doc_{i}" for i in range(num_docs)],
        "Target": gold_doc_targets,
        "Stance": gold_polarity
    })

    assert all(isinstance(t, list) for t in doc_targets)
    assert all(isinstance(t, list) for t in gold_doc_targets)

    dists, matches = metrics.targets_closest_distance(extracted_targets, gold_targets)
    assert len(dists) == len(extracted_targets)
    assert len(matches) == len(extracted_targets)

    targets_f1 = metrics.bertscore_f1_targets(extracted_targets, gold_targets, doc_targets, gold_stances['Target'].tolist())
    assert 0 <= targets_f1 <= 1

    polarity_f1 = metrics.f1_stances(extracted_targets, gold_targets, doc_targets, gold_stances['Target'].tolist(), polarity, gold_stances['Stance'].tolist())
    assert 0 <= polarity_f1 <= 1

def test_unsupervised_metrics():
    num_docs = 100
    docs = [f"doc_{i}" for i in range(num_docs)]
    sent_a = "sent_a"
    sent_b = "sent_b"
    vector = Vector(sent_a, sent_b)
    vectopic = MockStanceMining(vector, targets=['dogs', 'cats', 'cows'])
    doc_targets, probs, polarity = vectopic.fit_transform(docs)

    target_info = vectopic.get_target_info()
    all_targets = target_info['noun_phrase'].tolist()
    norm_targets_dist = metrics.normalized_targets_distance(all_targets, docs)
    assert 0 <= norm_targets_dist <= 1

    doc_dist = metrics.document_distance(probs)
    assert 0 <= doc_dist <= 1

    target_polarities = metrics.target_polarity(polarity)
    assert len(target_polarities) == len(docs)

    inclusion = metrics.hard_inclusion(doc_targets)
    assert 0 <= inclusion <= 1

    target_dist = metrics.target_distance(doc_targets, docs)
    assert 0 <= target_dist <= 1

def test_filter_similar_phrases():
    num_docs = 100
    docs = [f"doc_{i}" for i in range(num_docs)]
    vectopic = MockStanceMining(targets=['dogs', 'cats', 'cows'])
    stance_targets = [['dogs', 'cats', 'cat']] * num_docs
    filtered_stance_targets = vectopic._filter_similar_phrases(stance_targets)
    assert len(filtered_stance_targets) == len(stance_targets)

def test_bleu_targets():
    num_docs = 100
    gold_docs = [['dogs', 'cats', 'cow']] * num_docs
    doc_targets = [['dogs', 'cats', 'cat']] * num_docs
    bleu_score = metrics.bleu_targets(doc_targets, gold_docs)
    assert 0 <= bleu_score <= 1


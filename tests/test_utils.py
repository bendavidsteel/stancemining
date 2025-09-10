import time

from sentence_transformers import SentenceTransformer
import numpy as np

from stancemining import utils

def test_deduplicate_target_embeddings():
    test_pairs = [
        ('trudeau', "trudeau's", True),
        ('liberal-ndp government', 'ndp-liberal government', True),
        ('cbc', 'cpc', False)
    ]
    all_targets = list(set([pair[0] for pair in test_pairs] + [pair[1] for pair in test_pairs]))
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(all_targets, show_progress_bar=True)
    embed_clusters = utils.deduplicate_target_embeddings(embeddings)
    for i, (target1, target2, expected) in enumerate(test_pairs):
        cluster1 = embed_clusters[all_targets.index(target1)]
        cluster2 = embed_clusters[all_targets.index(target2)]
        score += int((cluster1 == cluster2) == expected)
    total = len(test_pairs)
    assert score > total * 0.5, f"Expected more than half of pairs to match, but got {score} out of {total} matching pairs."

def test_propagate_clusters():
    import cupy as cp
    cluster_labels = np.array([0, 1, 0, 3, 2])
    start_time = time.time()
    cluster_labels = utils.propagate_clusters_cupy(cp.asarray(cluster_labels), len(cluster_labels))
    end_time = time.time()
    print(f"Propagation took {end_time - start_time:.4f} seconds")
    assert np.asarray(cluster_labels) == np.array([0, 1, 0, 3, 0])

if __name__ == '__main__':
    test_propagate_clusters()

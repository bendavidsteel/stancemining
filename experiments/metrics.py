import numpy as np
import sentence_transformers
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

def sentence_embedding_similarity(targets, gold_targets):
    all_targets = targets + gold_targets
    embeddings = sentence_transformers.SentenceTransformer('paraphrase-MiniLM-L6-v2').encode(all_targets)
    target_embeddings = embeddings[:len(targets)]
    gold_target_embeddings = embeddings[len(targets):]
    similarity_matrix = cosine_similarity(target_embeddings, gold_target_embeddings)
    return similarity_matrix

def targets_closest_distance(targets, gold_targets):
    similarity_matrix = sentence_embedding_similarity(targets, gold_targets)
    closest_distances = np.max(similarity_matrix, axis=1)
    matches = [(targets[i], gold_targets[j]) for i, j in enumerate(np.argmax(similarity_matrix, axis=1))]
    return closest_distances, matches

def f1_targets(all_targets, all_gold_targets, doc_targets, gold_doc_targets):
    similarity_matrix = sentence_embedding_similarity(all_targets, all_gold_targets)
    gold_to_extracted = {all_gold_targets[i]: all_targets[j] for i, j in enumerate(np.argmax(similarity_matrix, axis=0))}
    gold_extracted_doc_targets = [gold_to_extracted[target] for target in gold_doc_targets]
    doc_targets = [t if t else 'none' for t in doc_targets]
    f1 = f1_score(gold_extracted_doc_targets, doc_targets, average='macro')
    return f1

def f1_stances(all_targets, all_gold_targets, doc_targets, gold_doc_targets, polarity, gold_polarity):
    similarity_matrix = sentence_embedding_similarity(all_targets, all_gold_targets)
    gold_to_extracted = {all_gold_targets[i]: all_targets[j] for i, j in enumerate(np.argmax(similarity_matrix, axis=0))}
    gold_extracted_doc_targets = [gold_to_extracted[target] for target in gold_doc_targets]
    polarity_to_score = [polarity[i, all_targets.index(gold_extracted_doc_targets[i])] for i in range(len(doc_targets))]
    # TODO split f1 scores by target?
    f1 = f1_score(gold_polarity, polarity_to_score, average='macro')
    return f1

def normalized_targets_distance(all_targets, documents):
    all_docs = all_targets + documents
    embeddings = sentence_transformers.SentenceTransformer('paraphrase-MiniLM-L6-v2').encode(all_docs)
    target_embeddings = embeddings[:len(all_targets)]
    doc_embeddings = embeddings[len(all_targets):]
    target_similarities = cosine_similarity(target_embeddings)
    doc_similarities = cosine_similarity(doc_embeddings)
    target_similarity = np.mean(np.triu(target_similarities, k=1))
    doc_similarity = np.mean(np.triu(doc_similarities, k=1))
    return target_similarity / doc_similarity

def hard_inclusion(doc_targets):
    return len([t for t in doc_targets if t]) / len(doc_targets)

def soft_inclusion(target_probs):
    return NotImplementedError()

def document_distance(target_probs):
    return np.mean(np.triu(cosine_similarity(target_probs), k=1))

def target_polarity(polarity):
    return np.var(polarity, axis=1)

def target_distance(doc_targets, docs):
    embeddings = sentence_transformers.SentenceTransformer('paraphrase-MiniLM-L6-v2').encode(doc_targets + docs)
    target_embeddings = embeddings[:len(doc_targets)]
    doc_embeddings = embeddings[len(doc_targets):]
    similarities = cosine_similarity(target_embeddings, doc_embeddings)
    return np.mean(similarities)

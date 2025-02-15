import bert_score
import numpy as np
import polars as pl
from sacrebleu.metrics import BLEU
import sentence_transformers
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import evaluate

def sentence_embedding_similarity(targets, gold_targets):
    all_targets = targets + gold_targets
    embeddings = sentence_transformers.SentenceTransformer('paraphrase-MiniLM-L6-v2').encode(all_targets)
    target_embeddings = embeddings[:len(targets)]
    gold_target_embeddings = embeddings[len(targets):]
    similarity_matrix = cosine_similarity(target_embeddings, gold_target_embeddings)
    return similarity_matrix

def targets_closest_distance(targets, gold_targets):
    if len(targets) == 0:
        return np.full(len(gold_targets), np.nan), []

    similarity_matrix = sentence_embedding_similarity(targets, gold_targets)
    closest_distances = np.max(similarity_matrix, axis=1)
    matches = [(targets[i], gold_targets[j]) for i, j in enumerate(np.argmax(similarity_matrix, axis=1))]
    return closest_distances, matches

def multi_label_bertscore(scorer, pred_labels, true_labels):
    if not pred_labels:
        return 0, 0, 0

    # For each predicted label, find its highest similarity with any true label
    # For each true label, find its highest similarity with any predicted label  
    pred_true_scores = scorer.score(pred_labels + true_labels, [true_labels] * len(pred_labels) + [pred_labels] * len(true_labels))[2].tolist()
    pred_scores = pred_true_scores[:len(pred_labels)]
    true_scores = pred_true_scores[len(pred_labels):]

    # Calculate overall precision/recall/F1
    precision = sum(pred_scores) / len(pred_scores) if pred_scores else 0
    recall = sum(true_scores) / len(true_scores) if true_scores else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def multi_label_f1(pred_labels, true_labels):
    pred_scores = []
    for p in pred_labels:
        t_scores = [t for t in true_labels if t[0] == p[0]]
        pred_scores.append([t[1] == p[1] for t in t_scores] if t_scores else [False])

    true_scores = []
    for t in true_labels:
        p_scores = [p for p in pred_labels if p[0] == t[0]]
        true_scores.append([p[1] == t[1] for p in p_scores] if p_scores else [False])

    precision = sum([sum(p) / len(p) for p in pred_scores]) / len(pred_scores) if pred_scores else 0
    recall = sum([sum(t) / len(t) for t in true_scores]) / len(true_scores) if true_scores else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def bertscore_f1_targets(doc_targets, gold_doc_targets):
    df = pl.DataFrame({
        'doc_id': list(range(len(doc_targets))),
        'pred': doc_targets,
        'gold': gold_doc_targets
    })
    p, r, f1 = [], [], []
    scorer = bert_score.BERTScorer(lang='en')
    for ex in tqdm(df.to_dicts(), desc='Calculating BERTScore'):
        pred_labels = ex['pred']
        true_labels = ex['gold']
        if not pred_labels:
            precision, recall, ex_f1 = 0, 0, 0
        else:
            # For each predicted label, find its highest similarity with any true label
            # For each true label, find its highest similarity with any predicted label  
            pred_true_scores = scorer.score(pred_labels + true_labels, [true_labels] * len(pred_labels) + [pred_labels] * len(true_labels))[2].tolist()
            pred_scores = pred_true_scores[:len(pred_labels)]
            true_scores = pred_true_scores[len(pred_labels):]
            
            # Calculate overall precision/recall/F1
            precision = sum(pred_scores) / len(pred_scores) if pred_scores else 0
            recall = sum(true_scores) / len(true_scores) if true_scores else 0
            ex_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        p.append(precision)
        r.append(recall)
        f1.append(ex_f1)
    
    df = df.with_columns([
        pl.Series(name='P', values=p, dtype=pl.Float32),
        pl.Series(name='R', values=r, dtype=pl.Float32),
        pl.Series(name='F1', values=f1, dtype=pl.Float32)
    ])
    bertscore_f1 = df['F1'].mean()
    bertscore_precision = df['P'].mean()
    bertscore_recall = df['R'].mean()

    return bertscore_f1, bertscore_precision, bertscore_recall

def f1_stances(all_targets, all_gold_targets, doc_targets, gold_doc_targets, polarity, gold_polarity, threshold=0.9):
    similarity_matrix = sentence_embedding_similarity(all_targets, all_gold_targets)
    extracted_to_gold = {all_targets[i]: all_gold_targets[j] for i, j in enumerate(np.argmax(similarity_matrix, axis=1)) if similarity_matrix[i, j] >= threshold}
    pred_stance = [[(extracted_to_gold[t], polarity[i, all_targets.index(t)]) for t in doc_targets[i] if t in extracted_to_gold] for i in range(len(doc_targets))]
    gold_stance = [list(set([(t, p) for t, p in zip(gold_doc_targets[i], gold_polarity[i])])) for i in range(len(doc_targets))]
    # TODO split f1 scores by target?
    p, r, f1 = [], [], []
    for gold_ex, pred_ex in zip(gold_stance, pred_stance):
        ex_p, ex_r, ex_f1 = multi_label_f1(pred_ex, gold_ex)
        p.append(ex_p)
        r.append(ex_r)
        f1.append(ex_f1)

    f1 = np.mean(f1)
    p = np.mean(p)
    r = np.mean(r)

    # f1 = f1_score(gold_polarity, polarity_to_score, average='macro')
    return f1, p, r

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

def stance_variance(polarity):
    return np.mean(np.nanvar(polarity, axis=0))


def target_distance(doc_targets, docs):
    df = pl.DataFrame({
        'doc_id': list(range(len(doc_targets))),
        'doc': docs,
        'target': doc_targets
    })
    target_df = df.explode('target').drop_nulls()
    embedding_model = sentence_transformers.SentenceTransformer('paraphrase-MiniLM-L6-v2')
    target_embeddings = embedding_model.encode(target_df['target'].to_list())
    target_df = target_df.with_columns([
        pl.Series(name='target_embedding', values=target_embeddings)
    ])
    target_df = target_df.group_by('doc_id').agg(pl.col('target_embedding').alias('target_embedding'))
    target_df = target_df.with_columns(pl.col('target_embedding').map_elements(lambda l: np.mean(np.array(l), axis=0), pl.List(pl.Float32)))
    df = df.join(target_df, on='doc_id', how='left')
    docs_with_target_df = df.filter(pl.col('target').list.len() > 0)
    doc_embeddings = embedding_model.encode(docs_with_target_df['doc'].to_list())
    target_embeddings = np.stack(docs_with_target_df['target_embedding'].to_numpy())
    similarities = np.sum(target_embeddings * doc_embeddings, axis=1) / (np.linalg.norm(target_embeddings, axis=1) * np.linalg.norm(doc_embeddings, axis=1))
    return np.mean(similarities)

def bleu_targets(doc_targets, gold_doc_targets):
    df = pl.DataFrame({
        'doc_id': list(range(len(doc_targets))),
        'pred': doc_targets,
        'gold': gold_doc_targets
    })
    bleu_scores = []
    bleu = BLEU()
    
    for ex in tqdm(df.to_dicts(), desc='Calculating BLEU'):
        pred_labels = ex['pred']
        true_labels = ex['gold']
        
        if not pred_labels:
            bleu_score = 0
        else:
            # Calculate BLEU score for this example
            # We treat each target as a separate "reference" translation
            scores = [bleu.sentence_score(pred_label, true_labels) for pred_label in pred_labels]
            bleu_score = np.mean([score.score for score in scores])
        
        bleu_scores.append(bleu_score)
    
    df = df.with_columns([
        pl.Series(name='BLEU', values=bleu_scores, dtype=pl.Float32)
    ])
    avg_bleu = df['BLEU'].mean()

    return avg_bleu

def mean_num_targets(doc_targets):
    return np.mean([len(t) for t in doc_targets])

def mean_cluster_size_ratio(stance_target_probs):
    return np.mean((stance_target_probs > 0).sum(0)) / stance_target_probs.shape[0]

def mean_cluster_size_std_ratio(stance_target_probs):
    return np.std((stance_target_probs > 0).sum(0)) / stance_target_probs.shape[0]
import logging

from nltk.corpus import stopwords
import numpy as np
import polars as pl

def remove_bad_targets(target_df: pl.DataFrame):
    phrases = [
        'the primary stance target of the piece of text is',
        'the primary stance target of this text is',
        'the primary stance target in the given text is',
        'the primary stance target of the text is',
        'the primary stance target is the noun phrase', 
        'the primary stance target of the given text is',
        'the primary stance target is',
        'stance target: 1.',
        'stance target:',
        'stance target'
    ]
    for phrase in phrases:
        target_df = target_df.with_columns(pl.col('Target').str.replace(phrase, ''))
    exclude_phrases = ['', 'url', 'rt', 'rt @', '@rt']
    target_df = target_df.with_columns(pl.col('Target').str.strip_chars('"').str.strip_chars(':').str.strip_chars())
    target_df = target_df.filter(~(pl.col('Target').str.contains('rt @\w+'))\
                              .or_(pl.col('Target').str.contains('rt \w+'))\
                              .or_(pl.col('Target').str.contains(r'^[\U0001F000-\U0001FFFF\u2600-\u26FF\u2700-\u27BF]+$'))\
                              .or_(pl.col('Target').is_in(stopwords.words('english') + stopwords.words('french')))\
                              .or_(pl.col('Target').str.to_lowercase().is_in(exclude_phrases)))
    return target_df

def get_var_and_max_var_target(documents_df: pl.DataFrame, target_info_df: pl.DataFrame) -> pl.DataFrame:
    if 'topic_id' in target_info_df.columns:
        target_info_df = target_info_df.group_by('noun_phrase')\
            .agg(pl.col('topic_id'), pl.col('polarity'))
    else:
        target_info_df = target_info_df.group_by('noun_phrase')\
            .agg(pl.col('polarity'))
    target_info_df = target_info_df.with_columns([
        pl.col('polarity').list.mean().alias('mean'),
        pl.when(pl.col('polarity').list.len() > 1)\
            .then(pl.col('polarity').list.var())\
            .otherwise(0)
            .alias('var')
    ])

    documents_df = documents_df.join(
        documents_df.explode('Targets')\
            .join(target_info_df, left_on='Targets', right_on='noun_phrase', how='left')\
            .group_by('ID')\
            .agg(pl.all().sort_by('var').last())\
            .with_columns(pl.col('Targets').alias('Target'))\
            .select(['ID', 'Target']),
        on='ID',
        how='left',
        maintain_order='left'
    )
    return documents_df, target_info_df


def filter_stance_targets(all_targets: pl.Series) -> pl.Series:
    # lower case all results
    all_targets = all_targets.list.eval(
        pl.element().str.to_lowercase().str.strip_chars().str.replace('stance target: ', '').str.replace('1. ', '').str.strip_chars().str.strip_chars('"').str.strip_chars("'")
    )
    # remove exact duplicates
    all_targets = all_targets.list.unique()
    return all_targets

def filter_phrases(target_embeds, similarity_threshold=0.9):
    # Compute cosine similarity matrix for current sublist
    embeddings = target_embeds.struct.field('embeddings').to_numpy()
    phrases_list = target_embeds.struct.field('Targets').to_list()
    norms = np.linalg.norm(embeddings, axis=1)
    similarity = np.dot(embeddings, embeddings.T) / np.outer(norms, norms)
    
    # Get upper triangular part to avoid duplicate comparisons
    similarity = np.triu(similarity, k=1)
    
    # Find indices of similar phrases within this sublist
    similar_indices = set(int(i) for i in np.where(similarity > similarity_threshold)[0])
    
    if not similar_indices:
        return phrases_list

    # Filter current sublist
    filtered_sublist = [
        phrase for j, phrase in enumerate(phrases_list)
        if j not in similar_indices
    ]
    return filtered_sublist

class MyLogger:
    def __init__(self, level):
        self.logger = logging.getLogger('StanceMining')
        self.set_level(level)
        self._add_handler()
        self.logger.propagate = False

    def info(self, message):
        self.logger.info(f"{message}")

    def warning(self, message):
        self.logger.warning(f"WARNING: {message}")

    def set_level(self, level):
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level in levels:
            self.logger.setLevel(level)

    def _add_handler(self):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(sh)

        # Remove duplicate handlers
        if len(self.logger.handlers) > 1:
            self.logger.handlers = [self.logger.handlers[0]]

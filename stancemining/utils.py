import logging

import polars as pl


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

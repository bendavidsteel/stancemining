from collections.abc import Iterable
import os

import polars as pl

def load_dataset(name, split='test', group=True, remove_synthetic_neutral=True):
    if isinstance(name, str):
        return _load_one_dataset(name, split, group, remove_synthetic_neutral)
    elif isinstance(name, Iterable):
        return pl.concat([_load_one_dataset(n, split, group, remove_synthetic_neutral) for n in name])
    else:
        raise ValueError(f'Unknown dataset: {name}')
    
def _load_one_dataset(name, split='test', group=True, remove_synthetic_neutral=True):
    if name == 'semeval':
        if split == 'val':
            path = 'semeval/semeval_train.csv'
            df = pl.read_csv(os.path.join('.', 'data', path))
            val_split = 0.2
            df = df.tail(int(len(df) * val_split))
        else:
            path = f'semeval/semeval_{split}.csv'
            df = pl.read_csv(os.path.join('.', 'data', path))
        df = df.rename({'Tweet': 'Text'})
        mapping = {
            'FAVOR': 'favor',
            'AGAINST': 'against',
            'NONE': 'neutral'
        }
    elif 'vast' in name:
        if split == 'val':
            split = 'dev'
        path = f'{name}/{name}_{split}.csv'
        df = pl.read_csv(os.path.join('.', 'data', path))
        if remove_synthetic_neutral:
            # remove synthetic neutrals
            df = df.filter(pl.col('type_idx') != 4)
        df = df.rename({'post': 'Text', 'topic_str': 'Target', 'label': 'Stance'}).select(['Text', 'Target', 'Stance'])
        mapping = {
            0: 'against',
            1: 'favor',
            2: 'neutral'
        }
    elif name == 'ezstance':
        path = f'ezstance/subtaskA/noun_phrase/raw_{split}_all_onecol.csv'
        df = pl.read_csv(os.path.join('.', 'data', path))
        df = df.rename({'Target 1': 'Target', 'Stance 1': 'Stance'})
        mapping = {
            'FAVOR': 'favor',
            'AGAINST': 'against',
            'NONE': 'neutral'
        }
    else:
        raise ValueError(f'Unknown dataset: {name}')
    
    df = df.with_columns(pl.col('Stance').replace_strict(mapping))
    if group:
        df = df.group_by('Text').agg([pl.col('Target'), pl.col('Stance')])

    df = df.select(['Text', 'Target', 'Stance'])

    return df
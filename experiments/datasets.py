import os

import polars as pl

def load_dataset(name, split='test'):
    
    if name == 'semeval':
        path = f'semeval/semeval_{split}.csv'
        df = pl.read_csv(os.path.join('.', 'data', path))
        df = df.rename({'Tweet': 'Text'})
        mapping = {
            'FAVOR': 'favor',
            'AGAINST': 'against',
            'NONE': 'neutral'
        }
    elif name == 'vast':
        path = f'vast/vast_{split}.csv'
        df = pl.read_csv(os.path.join('.', 'data', path))
        df = df.rename({'post': 'Text', 'topic_str': 'Target', 'label': 'Stance'})
        mapping = {
            0: 'against',
            1: 'favor',
            2: 'neutral'
        }
        # remove synthetic neutrals
        df = df.filter(pl.col('type_idx') != 4)
    elif name == 'ezstance':
        path = f'ezstance/subtaskA/mixed/raw_{split}_all_onecol.csv'
        df = pl.read_csv(os.path.join('.', 'data', path))
        pass
    
    df = df.with_columns(pl.col('Stance').replace_strict(mapping))
    df = df.group_by('Text').agg([pl.col('Target'), pl.col('Stance')])

    return df
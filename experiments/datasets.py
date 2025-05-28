from collections.abc import Iterable
import os

import polars as pl

def load_dataset(name, split='test', group=True, remove_synthetic_neutral=True):
    if isinstance(name, str):
        return _load_one_dataset(name, split, group, remove_synthetic_neutral)
    elif isinstance(name, Iterable):
        return pl.concat([_load_one_dataset(n, split, group, remove_synthetic_neutral) for n in name], how='diagonal_relaxed')
    else:
        raise ValueError(f'Unknown dataset: {name}')
    
def _load_one_dataset(name, split='test', group=True, remove_synthetic_neutral=True):
    datasets_path = os.path.join('.', 'data', 'datasets')
    if name == 'semeval':
        if split == 'val':
            path = 'semeval/semeval_train.csv'
            df = pl.read_csv(os.path.join(datasets_path, path))
            val_split = 0.2
            df = df.tail(int(len(df) * val_split))
        elif split == 'train':
            path = 'semeval/semeval_train.csv'
            df = pl.read_csv(os.path.join(datasets_path, path))
            train_split = 0.8
            df = df.head(int(len(df) * train_split))
        elif split == 'test':
            path = f'semeval/semeval_{split}.csv'
            df = pl.read_csv(os.path.join(datasets_path, path))
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
        df = pl.read_csv(os.path.join(datasets_path, path))
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
        df = pl.read_csv(os.path.join(datasets_path, path))
        df = df.rename({'Target 1': 'Target', 'Stance 1': 'Stance'})
        mapping = {
            'FAVOR': 'favor',
            'AGAINST': 'against',
            'NONE': 'neutral'
        }
    elif name == 'ezstance_claim':
        path = f'ezstance/subtaskA/claim/raw_{split}_all_onecol.csv'
        df = pl.read_csv(os.path.join(datasets_path, path))
        df = df.rename({'Target 1': 'Target', 'Stance 1': 'Stance'})
        mapping = {
            'FAVOR': 'favor',
            'AGAINST': 'against',
            'NONE': 'neutral'
        }
    elif name == 'pstance':
        names = ['bernie', 'biden', 'trump']
        pstance_path = os.path.join(datasets_path, 'PStance')
        df = pl.concat([pl.read_csv(os.path.join(pstance_path, f'raw_{split}_{name}.csv')) for name in names])
        df = df.rename({'Tweet': 'Text'})
        mapping = {
            'FAVOR': 'favor',
            'AGAINST': 'against'
        }
    elif name == 'mtcsd':
        if split == 'val':
            split = 'valid'
        mtcsd_path = os.path.join(datasets_path, 'MT-CSD-main', 'data')
        df = pl.DataFrame()
        for target in os.listdir(mtcsd_path):
            target_text_df = pl.read_csv(os.path.join(mtcsd_path, target, 'text.csv'))
            target_split_df = pl.read_json(os.path.join(mtcsd_path, target, f'{split}.json'))
            indexed_text_df = target_split_df.with_row_index('id')\
                .explode('index')\
                .with_row_index('idx')\
                .with_columns((1-pl.col('idx').rank('dense', descending=True).over('id').cast(pl.Int32)).alias('idx'))\
                .join(target_text_df, left_on='index', right_on='id')
            idxes = indexed_text_df['idx'].unique()
            target_df = indexed_text_df.filter(pl.col('idx') == 0).rename({'text': 'text_0'}).select(['id', 'text_0', 'stance'])
            for idx in idxes:
                if idx == 0:
                    continue
                target_df = target_df.join(indexed_text_df.filter(pl.col('idx') == idx).select(['id', 'text']).rename({'text': f'text_{idx}'}), on='id', how='left')
            target_df = target_df.with_columns(pl.lit(target).alias('Target'))
            df = pl.concat([df, target_df])
        mapping = {
            'favor': 'favor',
            'against': 'against',
            'none': 'neutral'
        }
        df = df.with_columns(
            pl.concat_list(
                sorted([col for col in df.columns if 'text_' in col and col != 'text_0'], reverse=True)
            ).list.drop_nulls().alias('ParentTexts')
        )
        df = df.rename({'text_0': 'Text', 'stance': 'Stance'})

    else:
        raise ValueError(f'Unknown dataset: {name}')
    
    df = df.with_columns(pl.col('Stance').replace_strict(mapping))
    if group:
        df = df.group_by('Text').agg([pl.col('Target'), pl.col('Stance')])

    df = df.with_columns(pl.lit(name).alias('Dataset'))

    cols = ['Text', 'Target', 'Stance', 'Dataset']
    if 'ParentTexts' in df.columns:
        cols.append('ParentTexts')
    df = df.select(cols)

    return df
from collections.abc import Iterable
import json
import os
import re

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
    elif name == 'romain_claims':
        if split == 'val':
            split = 'valid'
        elif split == 'test':
            split = 'valid'
        data_path = os.path.join(datasets_path, 'romain_claims', f'{split}.jsonl')
        with open(data_path) as f:
            ds = [json.loads(line) for line in f]

        new_data = []
        for item in ds:
            user_content = [msg for msg in item['messages'] if msg['role'] == 'user'][0]['content']
            assistant_content = [msg for msg in item['messages'] if msg['role'] == 'assistant'][0]['content']
            
            # Find the input text section
            original_text = re.search(r'Input text:\s*\n"((?:.|\n)+?)"\s*\n\n', user_content).group(1)
            
            # Extract claims from JSON response
            claims = re.findall(r'"text":\s*"([^"]+)"', assistant_content)
            for claim in claims:
                new_data.append({'Text': original_text, 'Target': claim, 'Stance': None})

        df = pl.from_dicts(new_data)

        if split == 'valid':
            df = df.head(int(0.5 * len(df)))
        elif split == 'test':
            df = df.tail(int(0.5 * len(df)))

        mapping = {}
    elif name == 'romain_tiktok_claims':
        dataset_path = os.path.join(datasets_path, 'romain_tiktok_claims')
        file_name = '1-claim-extractions-validated.json'
        with open(os.path.join(dataset_path, file_name)) as f:
            items = json.load(f)

        items = [v | {'id': k} for k, v in items.items()]
        df = pl.from_dicts(items)
        if split == 'train':
            df = df.head(int(0.8 * len(df)))
        elif split == 'val':
            df = df.slice(int(0.8 * len(df)), int(0.1 * len(df)))
        elif split == 'test':
            df = df.tail(int(0.1 * len(df)))
        else:
            raise Exception()
        df = df.explode('claims').drop_nulls('claims')
        df = df.select([pl.col('input_text').alias('Text'), pl.col('claims').alias('Target'), pl.lit(None).alias('Stance')])
        mapping = {}

    elif name == 'ctsdt':
        ctsdt_path = os.path.join(datasets_path, 'CTSDT', 'labeled_data.csv')
        df = pl.read_csv(ctsdt_path)

        df = df.sample(fraction=1, shuffle=True)
        train_split = 0.8
        val_split = 0.1
        if split == 'train':
            df = df.slice(0, int(len(df) * train_split))
        elif split == 'val':
            df = df.slice(int(len(df) * train_split), int(len(df) * (train_split + val_split)))
        elif split == 'test':
            df = df.slice(int(len(df) * (train_split + val_split)), len(df))
        else:
            raise ValueError(f'Unknown split: {split}')

        df = df.with_columns(pl.col('sub_branch').str.strip_chars('[]').str.split(', ').list.eval(pl.element().str.strip_chars("'")))
        
        chain_df = df.explode('sub_branch')\
            .with_columns(pl.col('sub_branch').fill_null(pl.col('id')).cast(pl.Int64))\
            .with_columns((1-pl.col('sub_branch').rank('dense', descending=True).over('id').cast(pl.Int32)).alias('idx'))\
            .with_columns(pl.when(pl.col('idx').is_null()).then(pl.lit(0)).otherwise(pl.col('idx')).alias('idx'))\
            .filter(pl.col('idx') >= -5) # drop threads that are too deep
        df = chain_df.drop('text')\
            .join(df.select(['id', 'text']), left_on='sub_branch', right_on='id', how='left')\
            .pivot(index=['id', 'label'], columns='idx', values='text')
        df = df.with_columns(
            pl.concat_list(
                sorted([col for col in df.columns if col not in ['id', 'label', '0']], reverse=True)
            ).list.drop_nulls().alias('ParentTexts')
        )
        df = df.rename({'0': 'Text', 'label': 'Stance'})
        df = df.with_columns(pl.lit('COVID-19 vaccination').alias('Target'))
        mapping = {
            'FAVOR': 'favor',
            'AGAINST': 'against',
            'NEITHER': 'neutral'
        }
    elif name == 'romain_claims':
        path = 'romain_claims/train-v1-valid.jsonl'
        with open(os.path.join(datasets_path, path)) as f:
            items = [json.loads(line) for line in f]

        data = []
        for item in items:
            user_message = [msg for msg in item['messages'] if msg['role'] == 'user'][0]
            assistant_message = [msg for msg in item['messages'] if msg['role'] == 'assistant'][0]
            
            # Extract original text from user message
            user_content = user_message['content']
            # Find the input text section
            if '## Input Text\n' in user_content:
                original_text = user_content.split('## Input Text\n')[1].strip()
            else:
                continue
            
            # Extract claims from JSON response
            try:
                response_json = json.loads(assistant_message['content'].split('```json\n')[1].split('\n```')[0])
                claims = response_json['claims']
                for claim in claims:
                    data.append({'Text': original_text, 'Target': claim, 'Stance': None})
            except:
                continue
        df = pl.DataFrame(data)
        mapping = {}
    else:
        raise ValueError(f'Unknown dataset: {name}')
    
    df = df.with_columns(pl.col('Stance').replace_strict(mapping))
    if group:
        if 'ParentTexts' in df.columns:
            df = df.unique(['ParentTexts', 'Text', 'Target'])
            df = df.group_by(['ParentTexts', 'Text']).agg([pl.col('Target'), pl.col('Stance')])
        else:
            df = df.unique(['Text', 'Target'])
            df = df.group_by('Text').agg([pl.col('Target'), pl.col('Stance')])

    df = df.with_columns(pl.lit(name).alias('Dataset'))

    cols = ['Text', 'Target', 'Stance', 'Dataset']
    if 'ParentTexts' in df.columns:
        cols.append('ParentTexts')
    df = df.select(cols)

    return df
from collections.abc import Iterable
import json
import os
import re

import polars as pl

def load_dataset(name, split='test', group=True, remove_synthetic_neutral=True, task=None):
    if isinstance(name, str):
        return _load_one_dataset(name, split, group, remove_synthetic_neutral, task)
    elif isinstance(name, Iterable):
        return pl.concat([_load_one_dataset(n, split, group, remove_synthetic_neutral, task) for n in name], how='diagonal_relaxed')
    else:
        raise ValueError(f'Unknown dataset: {name}')

def _load_one_dataset(name, split='test', group=True, remove_synthetic_neutral=True, task=None):
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
    elif name == 'catalonia':
        target = 'Catalonia independence'
        sp_df = pl.read_csv(os.path.join(datasets_path, 'catalonia', f'spanish_{split}.csv'), separator='\t', schema_overrides={'id_str': pl.Utf8})
        ct_df = pl.read_csv(os.path.join(datasets_path, 'catalonia', f'catalan_{split}.csv'), separator='\t', schema_overrides={'id_str': pl.Utf8})
        df = pl.concat([sp_df, ct_df], how='diagonal_relaxed').drop('id_str')
        df = df.with_columns(pl.lit(target).alias('Target'))
        df = df.rename({'TWEET': 'Text', 'LABEL': 'Stance'})
        mapping = {
            'FAVOR': 'favor',
            'AGAINST': 'against',
            'NEUTRAL': 'neutral'
        }
    elif name == 'french-election':
        lepen_df = pl.read_csv(os.path.join(datasets_path, 'french-election', 'lepen_fr.csv'))
        macron_df = pl.read_csv(os.path.join(datasets_path, 'french-election', 'macron_fr.csv'))
        referendum_df = pl.read_csv(os.path.join(datasets_path, 'french-election', 'referendum_it.csv'))
        lepen_target = 'Marine Le Pen'
        macron_target = 'Emmanuel Macron'
        referendum_target = 'Constitutional Referendum'
        lepen_df = lepen_df.with_columns(pl.lit(lepen_target).alias('Target'))
        macron_df = macron_df.with_columns(pl.lit(macron_target).alias('Target'))
        referendum_df = referendum_df.with_columns(pl.lit(referendum_target).alias('Target'))
        df = pl.concat([lepen_df, macron_df, referendum_df], how='diagonal_relaxed')
        if split == 'train':
            df = df.filter(pl.col('Set') == 'Training')
        elif split == 'val':
            df = df.filter(pl.col('Set') == 'Test').sample(fraction=0.5, seed=42)
        elif split == 'test':
            df = df.filter(pl.col('Set') == 'Test').sample(fraction=0.5, seed=42)
        df = df.select(['Tweet', 'Target', 'Stance'])
        df = df.rename({'Tweet': 'Text'})
        mapping = {
            'FAVOUR': 'favor',
            'AGAINST': 'against',
            'NONE': 'neutral',
            'none': 'neutral',
            'favor': 'favor',
            'agains': 'against'
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
            except Exception:
                continue
        df = pl.DataFrame(data)
        mapping = {}
    elif name == 'stanceosaurus':
        path = os.path.join(datasets_path, 'stanceosaurus')

        split_mapper = {
            'train': 'train',
            'dev': 'val',
            'test': 'test'
        }

        def recurse_children(item, claim, posts):
            childless_item = item.copy()
            del childless_item['children']
            for child in item['children']:
                child = child.copy()
                child_thread = []
                for ancestor in childless_item.get('thread', []):
                    ancestor = ancestor.copy()
                    if 'thread' in ancestor:
                        del ancestor['thread']
                    child_thread.append(ancestor)
                child_thread.append(childless_item)
                child['thread'] = child_thread
                recurse_children(child, claim, posts)
                del child['children']
                child['claim'] = claim
                posts.append(child)

        languages = os.listdir(path)
        df = pl.DataFrame()

        for language in languages:
            language_df = pl.DataFrame()

            for language_root, dirs, filenames in os.walk(os.path.join(path, language)):
                for filename in filenames:
                    if not (filename.endswith('.json') or filename.endswith('.jsonl')):
                        continue

                    if filename == 'masked.json':
                        continue

                    if language_root in ['dev', 'test', 'train']:
                        if split != split_mapper[language_root]:
                            continue

                    file_path = os.path.join(language_root, filename)
                    try:
                        with open(file_path) as f:
                            if filename.endswith('.json'):
                                items = json.load(f)
                            else:
                                items = [json.loads(line) for line in f]
                    except json.JSONDecodeError:
                        with open(file_path, 'r') as f:
                            items = [json.loads(line) for line in f]
                    
                    if not items:
                        continue

                    posts = []
                    for item in items:
                        post = item['root_tweet']
                        recurse_children(item['root_tweet'], item['claim'], posts)
                        post['claim'] = item['claim']
                        del post['children']
                        post['thread'] = []
                        posts.append(post)

                    language_df = pl.concat([language_df, pl.from_dicts(posts)], how='diagonal_relaxed')

            if language == 'hindi':
                # hindi has messed up json formatting where stance attribute is under leaning and stance has nonsensical values
                language_df = language_df.drop('stance')\
                    .rename({'leaning': 'stance'})\
                    .filter(pl.col('stance') != 'Discussing') # remove discussing because we don't have leaning data

            if language != 'english':
                # need to filter to split
                if split == 'train':
                    language_df = language_df.head(int(0.8 * len(language_df)))
                elif split == 'val':
                    language_df = language_df.slice(int(0.8 * len(language_df)), int(0.1 * len(language_df)))
                elif split == 'test':
                    language_df = language_df.tail(int(0.1 * len(language_df)))

            language_df = language_df.with_columns(pl.lit(language).alias('language'))
            df = pl.concat([df, language_df], how='diagonal_relaxed')

        # clean up trailing spaces
        spelling_mistakes = {
            'Dicussing': 'Discussing',
        }
        df = df.with_columns(pl.col('stance').str.strip_chars().replace(spelling_mistakes))

        df = df.with_columns(pl.col('thread').list.eval(pl.col('').struct.field('text')).alias('ParentTexts'))\
            .rename({'text': 'Text', 'claim': 'Target', 'stance': 'Stance'})
        
        if task == 'claim-entailment-3way':
            df = df.with_columns(pl.when(pl.col('leaning').is_in(['Refuting', 'Supporting'])).then(pl.col('leaning')).otherwise(pl.col('Stance')).alias('Stance'))\
                .with_columns(pl.when(pl.col('Stance').is_in(['Supporting', 'Refuting'])).then(pl.col('Stance')).otherwise(pl.lit('Other')).alias('Stance'))
            
            mapping = {
                'Refuting': 'refuting',
                'Supporting': 'supporting',
                'Other': 'neutral'
            }
        elif task == 'claim-entailment-7way':
            df = df.with_columns(pl.when(pl.col('leaning').is_in(['Refuting', 'Supporting'])).then(pl.format("Leaning {}", pl.col('leaning'))).otherwise(pl.col('Stance')).alias('Stance'))
            mapping = {l: l.lower() for l in df['Stance'].unique()}
        else:
            raise ValueError(f'Unknown task: {task}')
        
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
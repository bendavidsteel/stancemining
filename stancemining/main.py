import logging
from typing import Any, List, Union

from bertopic import BERTopic
import numpy as np
import pandas as pd
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

from stancemining import llms, finetune, prompting, utils
from stancemining.ngram_gen import NGramGeneration

logger = utils.MyLogger('WARNING')


class StanceMining:
    def __init__(
            self, 
            llm_method='finetuned',
            num_representative_docs=5,
            model_lib='transformers', 
            model_name='microsoft/Phi-3.5-mini-instruct', 
            model_kwargs={}, 
            tokenizer_kwargs={},
            stance_detection_finetune_kwargs={},
            target_extraction_finetune_kwargs={},
            embedding_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            load_generator=True,
            get_stance=True,
            verbose=False,
            lazy=True
        ):
        assert llm_method in ['zero-shot', 'finetuned'], f"LLM method must be either 'zero-shot' or 'finetuned', not '{llm_method}'"
        self.llm_method = llm_method

        self.num_representative_docs = num_representative_docs

        self.model_lib = model_lib
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        if 'device_map' not in self.model_kwargs:
            self.model_kwargs['device_map'] = 'auto'
        if 'torch_dtype' not in self.model_kwargs:
            self.model_kwargs['torch_dtype'] = 'auto'

        self.tokenizer_kwargs = tokenizer_kwargs

        if 'model_name' not in stance_detection_finetune_kwargs and 'hf_model' not in stance_detection_finetune_kwargs:
            stance_detection_finetune_kwargs['hf_model'] = 'bendavidsteel/SmolLM2-135M-Instruct-stance-detection'
        self.stance_detection_finetune_kwargs = stance_detection_finetune_kwargs

        if 'model_name' not in target_extraction_finetune_kwargs and 'hf_model' not in target_extraction_finetune_kwargs:
            target_extraction_finetune_kwargs['hf_model'] = 'bendavidsteel/SmolLM2-360M-Instruct-stance-target-extraction'
        self.target_extraction_finetune_kwargs = target_extraction_finetune_kwargs

        self._get_stance = get_stance
        self.verbose = verbose
        if self.verbose:
            logger.set_level("DEBUG")
        else:
            logger.set_level("WARNING")

        if load_generator:
            self.generator = self._get_generator(lazy=lazy)
        else:
            self.generator = None

        self.embedding_model = embedding_model
        self.embedding_cache_df = pl.DataFrame({'text': [], 'embedding': []}, schema={'text': pl.String, 'embedding': pl.Array(pl.Float32, 384)})

    def _get_embedding_model(self):
        model = SentenceTransformer(self.embedding_model)
        return model

    def _get_embeddings(self, docs: Union[List[str], pl.Series], model=None) -> np.ndarray:
        if model is None:
            model = self._get_embedding_model()
        # check for cached embeddings
        if isinstance(docs, pl.Series):
            document_df = docs.rename('text').to_frame()
        else:
            document_df = pl.DataFrame({'text': docs})
        document_df = document_df.join(self.embedding_cache_df, on='text', how='left', maintain_order='left')
        missing_docs = document_df.unique('text').filter(pl.col('embedding').is_null())
        if len(missing_docs) > 0:
            new_embeddings = model.encode(missing_docs['text'].to_list(), show_progress_bar=self.verbose)
            missing_docs = missing_docs.with_columns(pl.Series(name='embedding', values=new_embeddings))
            # cache embeddings
            self.embedding_cache_df = pl.concat([self.embedding_cache_df, missing_docs], how='diagonal_relaxed')
            # add new embeddings to document_df
            document_df = document_df.join(missing_docs, on='text', how='left', maintain_order='left')\
                .with_columns(pl.coalesce(['embedding', 'embedding_right']))\
                .drop('embedding_right')
        embeddings = document_df['embedding'].to_numpy()

        return embeddings

    
    def _ask_llm_stance_target(self, docs: List[str]):
        num_samples = 3
        if self.llm_method == 'zero-shot':
            targets = prompting.ask_llm_zero_shot_stance_target(self.generator, docs, {'num_samples': num_samples})
        elif self.llm_method == 'finetuned':
            df = pl.DataFrame({'Text': docs})
            
            results = finetune.get_predictions("topic-extraction", df, self.target_extraction_finetune_kwargs, model_kwargs=self.target_extraction_model_kwargs)
            if isinstance(results[0], str):
                targets = [[r] for r in results]
            else:
                targets = results
        else:
            raise ValueError(f"Unrecognised self.llm_method value: {self.llm_method}")
        target_df = pl.DataFrame({'Targets': targets})
        target_df = target_df.with_columns(utils.filter_stance_targets(target_df['Targets']))
        return target_df['Targets'].to_list()

    def _ask_llm_stance(self, docs, stance_targets, parent_docs=None):
        if self.llm_method == 'zero-shot':
            return prompting.ask_llm_zero_shot_stance(self.generator, docs, stance_targets)
        elif self.llm_method == 'finetuned':
            data = pl.DataFrame({'Text': docs, 'Target': stance_targets, 'ParentText': parent_docs})
            results = finetune.get_predictions("stance-classification", data, self.stance_detection_finetune_kwargs, model_kwargs=self.stance_detection_model_kwargs)
            results = [r.upper() for r in results]
            return results

    def _filter_document_similar_targets(self, phrases_list: pl.Series, embedding_model=None, similarity_threshold: float = 0.8) -> pl.Series:
        """
        Filter out similar phrases from a list of lists based on embedding similarity,
        only comparing phrases within the same sublist.
        
        Args:
            phrases_list: List of lists containing phrases to filter
            embedding_fn: Function that takes a list of strings and returns numpy array of embeddings
            similarity_threshold: Threshold above which phrases are considered similar (default: 0.8)
            
        Returns:
            List of lists with similar phrases removed
        """
        col_name = phrases_list.name
        df = phrases_list.rename('Targets').to_frame()

        df = df.with_columns(pl.col('Targets').list.len().alias('target_len'))

        # If input has less than 2 items, return as is
        if df['target_len'].min() < 2:
            return phrases_list
        
        df = df.with_row_index()
            
        # Flatten list to compute embeddings efficiently
        target_df = df.explode('Targets')
        
        # Get embeddings for all phrases at once
        all_embeddings = self._get_embeddings(target_df['Targets'], model=embedding_model)
        
        target_df = target_df.with_columns(pl.Series(name='embeddings', values=all_embeddings))
        target_df = target_df.select(['index', pl.struct(['Targets', 'embeddings']).alias('target_embeds')])
        df = target_df.group_by('index').agg(pl.col('target_embeds')).with_columns(pl.col('target_embeds').list.len().alias('target_len'))

        df = df.with_columns(
            pl.when(pl.col('target_len') > 1)
                .then(pl.col('target_embeds').map_elements(utils.filter_phrases, return_dtype=pl.List(pl.String)))\
                .otherwise(pl.col('target_embeds'))\
            .alias('targets')
        )

        return df['targets'].rename(col_name) 
    
    def _filter_all_similar_targets(self, documents_df: pl.DataFrame, embedding_model=None) -> pl.DataFrame:
        """
        Filter out similar targets from a DataFrame of documents based on embedding similarity,
        only comparing targets within the same document.
        
        Args:
            documents_df: DataFrame containing 'Targets' column with lists of targets
            embedding_model: Embedding model to use for computing embeddings

        Returns:
            DataFrame with 'Targets' column filtered for similar targets
        """
        target_df = documents_df.select('Targets')\
            .explode('Targets')\
            .drop_nulls()\
            .rename({'Targets': 'Target'})\
            .group_by('Target')\
            .agg(pl.count().alias('count'))
        embeddings = self._get_embeddings(target_df['Target'], model=embedding_model)
        target_mapper = utils.get_similar_target_mapper(embeddings, target_df)
        documents_df = documents_df.with_columns(
            pl.col('Targets').list.eval(pl.element().replace(target_mapper)).list.unique()
        )
        return documents_df

    def _get_base_topic_model(self, bertopic_kwargs):
        if 'verbose' not in bertopic_kwargs:
            bertopic_kwargs['verbose'] = self.verbose
        if 'umap_model' not in bertopic_kwargs:
            if torch.cuda.is_available():
                from cuml.manifold import UMAP
                bertopic_kwargs['umap_model'] = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, verbose=self.verbose, random_state=42)
        if 'hdbscan_model' not in bertopic_kwargs:
            if torch.cuda.is_available():
                from cuml.cluster import HDBSCAN
                bertopic_kwargs['hdbscan_model'] = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True, verbose=self.verbose, random_state=42)
        return BERTopic(**bertopic_kwargs)

    def get_base_targets(self, docs, embedding_model=None):
        if embedding_model is None:
            embedding_model = self._get_embedding_model()

        if isinstance(docs, list):
            documents_df = pl.DataFrame({"Document": docs}).with_row_index(name='ID')
        elif isinstance(docs, pl.DataFrame):
            documents_df = docs
            assert 'Document' in documents_df.columns, "docs must have a 'Document' column"
            if 'ID' not in documents_df.columns:
                documents_df = documents_df.with_row_index(name='ID')

        stance_targets = self._ask_llm_stance_target(documents_df['Document'])
        documents_df = documents_df.with_columns(pl.Series(name='Targets', values=stance_targets, dtype=pl.List(pl.String)))
        documents_df = documents_df.with_columns(self._filter_document_similar_targets(documents_df['Targets'], embedding_model=embedding_model))
        
        return documents_df

    def _fit_transform(self, docs, bertopic_kwargs={}):
        
        if isinstance(docs, list):
            document_df = pl.DataFrame({"Document": docs}).with_row_index(name='ID')
        elif isinstance(docs, pl.DataFrame):
            document_df = docs
            assert 'Document' in document_df.columns, "docs must have a 'Document' column"
            if 'ID' not in document_df.columns:
                document_df = document_df.with_row_index(name='ID')
        
        embed_model = self._get_embedding_model()
        if 'Targets' not in document_df.columns:
            logger.info("Getting base targets")
            document_df = self.get_base_targets(docs, embedding_model=embed_model)

            # cluster initial stance targets
            base_target_df = document_df.explode('Targets').unique('Targets').drop_nulls()[['Targets']]
        else:
            assert isinstance(document_df.schema['Targets'], pl.List), "Targets column must be a list of strings"
            base_target_df = document_df.explode('Targets').unique('Targets').drop_nulls('Targets')[['Targets']]
        
        logger.info("Fitting BERTopic model")
        self.topic_model = self._get_base_topic_model(bertopic_kwargs)
        targets = base_target_df['Targets'].to_list()
        embeddings = self._get_embeddings(targets, model=embed_model)
        topics, probs = self.topic_model.fit_transform(targets, embeddings=embeddings)
        base_target_df = base_target_df.with_columns([
            pl.Series(name='Topic', values=topics),
            pl.col('Targets').alias('Document')
        ]).with_row_index(name='ID')
        repr_docs, _, _, _ = self.topic_model._extract_representative_docs(
            self.topic_model.c_tf_idf_,
            base_target_df.to_pandas(),
            self.topic_model.topic_representations_,
            nr_samples=500,
            nr_repr_docs=self.num_representative_docs,
        )
        self.topic_model.representative_docs_ = repr_docs
        topic_info = self.topic_model.get_topic_info()

        # get higher level stance targets for each topic
        topic_info = pl.from_pandas(topic_info)
        no_outliers_df = topic_info.filter(pl.col('Topic') != -1)
        logger.info("Getting higher level stance targets")
        stance_targets = [prompting.ask_llm_target_aggregate(self.generator, r['Representative_Docs'], r['Representation']) for r in tqdm(no_outliers_df[['Representative_Docs', 'Representation']].to_dicts(), desc="Getting topic noun phrases")]
        no_outliers_df = no_outliers_df.with_columns(
            pl.Series(name='noun_phrases', values=stance_targets, dtype=pl.List(pl.String))
        )
        no_outliers_df = no_outliers_df.with_columns(self._filter_document_similar_targets(no_outliers_df['noun_phrases'], embedding_model=embed_model))
        
        topic_info = pl.concat([topic_info.filter(pl.col('Topic') == -1), no_outliers_df], how='diagonal_relaxed')\
            .with_columns(pl.col('noun_phrases').fill_null([]))

        logger.info("Mapping targets to topics")
        # map documents to new noun phrases via topics
        target_df = document_df.explode('Targets')
        target_df = target_df.join(base_target_df, on='Targets', how='left')
        target_df = target_df.filter(pl.col('Topic') != -1)
        target_df = target_df.join(topic_info, on='Topic', how='left')

        logger.info("Joining new targets to documents")
        document_df = document_df.join(
                target_df.group_by('ID')\
                    .agg(pl.col('noun_phrases').flatten())\
                    .with_columns(pl.col('noun_phrases').list.drop_nulls().alias('NewTargets')),
            on='ID', 
            how='left',
            maintain_order='left'
        ).drop('noun_phrases')
        
        logger.info("Combining base and topic targets")
        document_df = document_df.with_columns(
            pl.when(pl.col('NewTargets').is_not_null())\
                .then(pl.concat_list(pl.col('Targets'), pl.col('NewTargets')))\
                .otherwise(pl.col('Targets')))
        document_df = document_df.drop(['NewTargets'])

        logger.info("Removing similar stance targets")
        # remove targets that are too similar
        document_df = self._filter_all_similar_targets(document_df, embed_model)

        if self._get_stance:
            logger.info("Getting stance classifications for targets")
            document_df = self.get_stance(document_df)

        logger.info("Getting target info")
        self.target_info = document_df.explode('Targets')\
            .select('Targets')\
            .drop_nulls()\
            .rename({'Targets': 'Target'})\
            .group_by('Target')\
            .len()\
            .rename({'len': 'Count'})
        # join to topic df to get topic info
        self.target_info = self.target_info.join(
            no_outliers_df.drop(['Name', 'Count']).explode('noun_phrases').drop_nulls('noun_phrases').rename({'noun_phrases': 'Target'}),
            on='Target',
            how='left'
        )

        logger.info("Done")
        return document_df
    
    def get_stance(self, document_df: pl.DataFrame) -> pl.DataFrame:
        if 'ID' not in document_df.columns:
            document_df = document_df.with_row_index(name='ID')

        target_df = document_df.explode('Targets').rename({'Targets': 'Target'})
        parent_docs = target_df['ParentDocument'] if 'ParentDocument' in target_df.columns else None
        target_stance = self._ask_llm_stance(target_df['Document'], target_df['Target'], parent_docs=parent_docs)
        target_df = target_df.with_columns(pl.Series(name='stance', values=target_stance))
        target_df = target_df.with_columns(pl.col('stance').replace_strict({'FAVOR': 1, 'AGAINST': -1, 'NEUTRAL': 0}).alias('stance'))

        document_df = document_df.drop('Targets')\
            .join(
                target_df.group_by('ID')\
                    .agg(pl.col('Target').alias('Targets'), pl.col('stance').alias('Stances')),
                on='ID',
                how='left',
                maintain_order='left'
            )
        return document_df
    
    def _get_targets_probs_polarity(self, documents):
        all_targets = self.target_info['noun_phrase'].to_list()
        target_to_idx = {target: idx for idx, target in enumerate(all_targets)}
        doc_targets = documents['Targets'].to_list()
        doc_targets = [[t] if t is not None else [] for t in doc_targets]

        num_targets = len(self.target_info)
        
        polarities = np.full((len(documents), num_targets), np.nan)
        probs = np.zeros((len(documents), num_targets))
        for i, doc_data in enumerate(documents.to_dicts()):
            doc_num_targets = len(doc_data['Targets'])
            for doc_target, doc_polarity in zip(doc_data['Targets'], doc_data['Polarities']):
                idx = target_to_idx[doc_target]
                polarities[i, idx] = doc_polarity
                probs[i, idx] = 1

        return documents, probs, polarities

    def fit_transform(self, docs, embedding_cache=None, bertopic_kwargs={}):

        if embedding_cache is not None:
            assert isinstance(embedding_cache, pl.DataFrame), "embedding_cache must be a polars DataFrame"
            assert 'text' in embedding_cache.columns, "embedding_cache must have a 'text' column"
            assert 'embedding' in embedding_cache.columns, "embedding_cache must have an 'embedding' column"
            assert isinstance(embedding_cache.schema['embedding'], pl.Array), "embedding_cache column 'embedding' must be an array of floats"
            assert isinstance(embedding_cache.schema['text'], pl.String), "embedding_cache column 'text' must be a string"
            self.embedding_cache_df = embedding_cache

        documents = self._fit_transform(docs, bertopic_kwargs)
        
        if self._get_stance:
            return self._get_targets_probs_polarity(documents)
        else:
            return documents

    def get_target_info(self):
        return self.target_info
    
    def get_topic_model(self):
        return self.topic_model

    def _get_generator(self, lazy=False):
        if self.model_lib == 'transformers':
            return llms.Transformers(self.model_name, self.model_kwargs, self.tokenizer_kwargs, lazy=lazy)
        else:
            raise ValueError(f"LLM library '{self.model_lib}' not implemented")
        

                

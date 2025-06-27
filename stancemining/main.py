import logging
from typing import Any, List, Union

import toponymy
import numpy as np
import pandas as pd
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

from stancemining import llms, finetune, prompting, utils
from stancemining.ngram_gen import NGramGeneration

logger = logging.getLogger('StanceMining')
logger.setLevel('WARNING')
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s: %(message)s'))
logger.addHandler(sh)
logger.propagate = False

class StanceMining:
    def __init__(
            self, 
            llm_method='finetuned',
            num_representative_docs=5,
            model_inference='vllm', 
            model_name='microsoft/Phi-4-mini-instruct', 
            model_kwargs={}, 
            tokenizer_kwargs={},
            stance_detection_finetune_kwargs={},
            stance_detection_model_kwargs={},
            target_extraction_finetune_kwargs={},
            target_extraction_model_kwargs={},
            embedding_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            embedding_model_inference='vllm',
            get_stance=True,
            verbose=False,
            lazy=True
        ):
        assert llm_method in ['zero-shot', 'finetuned'], f"LLM method must be either 'zero-shot' or 'finetuned', not '{llm_method}'"
        self.llm_method = llm_method

        self.num_representative_docs = num_representative_docs

        self.model_inference = model_inference
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        if 'device_map' not in self.model_kwargs:
            self.model_kwargs['device_map'] = 'auto'
        if 'torch_dtype' not in self.model_kwargs:
            self.model_kwargs['torch_dtype'] = 'auto'

        self.tokenizer_kwargs = tokenizer_kwargs

        if 'model_name' not in stance_detection_finetune_kwargs and 'hf_model' not in stance_detection_finetune_kwargs:
            stance_detection_finetune_kwargs['hf_model'] = 'bendavidsteel/SmolLM2-135M-Instruct-stance-detection'
        if 'classification_method' not in stance_detection_finetune_kwargs:
            stance_detection_finetune_kwargs['classification_method'] = 'head'
        if 'prompting_method' not in stance_detection_finetune_kwargs:
            stance_detection_finetune_kwargs['prompting_method'] = 'stancemining'
        self.stance_detection_finetune_kwargs = stance_detection_finetune_kwargs

        if 'device_map' not in stance_detection_model_kwargs:
            stance_detection_model_kwargs['device_map'] = 'auto'
        self.stance_detection_model_kwargs = stance_detection_model_kwargs

        if 'model_name' not in target_extraction_finetune_kwargs and 'hf_model' not in target_extraction_finetune_kwargs:
            target_extraction_finetune_kwargs['hf_model'] = 'bendavidsteel/SmolLM2-360M-Instruct-stance-target-extraction'
        if 'generation_method' not in target_extraction_finetune_kwargs:
            target_extraction_finetune_kwargs['generation_method'] = 'list'
        if 'prompting_method' not in target_extraction_finetune_kwargs:
            target_extraction_finetune_kwargs['prompting_method'] = 'stancemining'
        self.target_extraction_finetune_kwargs = target_extraction_finetune_kwargs
        
        if 'device_map' not in target_extraction_model_kwargs:
            target_extraction_model_kwargs['device_map'] = 'auto'
        self.target_extraction_model_kwargs = target_extraction_model_kwargs

        self._get_stance = get_stance
        self.verbose = verbose
        if self.verbose:
            logger.setLevel("DEBUG")
        else:
            logger.setLevel("WARNING")

        self.generator = self._get_generator(lazy=lazy)

        self.embedding_model = embedding_model
        self.embedding_model_inference = embedding_model_inference
        self.embedding_cache_df = pl.DataFrame({'text': [], 'embedding': []}, schema={'text': pl.String, 'embedding': pl.Array(pl.Float32, 384)})

    def _get_embedding_model(self):
        if self.embedding_model_inference == 'vllm':
            import vllm
            model = vllm.LLM(model=self.embedding_model)
        elif self.embedding_model_inference == 'transformers':
            model = SentenceTransformer(self.embedding_model)
        else:
            raise ValueError(f"Embedding model inference method '{self.embedding_model_inference}' not implemented")
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
            if self.embedding_model_inference == 'vllm':
                outputs = model.embed(missing_docs['text'].to_list(), use_tqdm=self.verbose)
                new_embeddings = np.stack([np.array(o.outputs.embedding) for o in outputs])
            elif self.embedding_model_inference == 'transformers':
                new_embeddings = model.encode(missing_docs['text'].to_list(), show_progress_bar=self.verbose)
            else:
                raise ValueError(f"Embedding model inference method '{self.embedding_model_inference}' not implemented")
            
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
            
            if self.model_inference == 'transformers':
                results = finetune.get_predictions("topic-extraction", df, self.target_extraction_finetune_kwargs, model_kwargs=self.target_extraction_model_kwargs)
            elif self.model_inference == 'vllm':
                results = llms.get_vllm_predictions("topic-extraction", df, self.target_extraction_finetune_kwargs, verbose=self.verbose)
            else:
                raise ValueError()

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
            data = pl.DataFrame({'Text': docs, 'Target': stance_targets, 'ParentTexts': parent_docs})
            if isinstance(data.schema['ParentTexts'], pl.String):
                # convert to list
                data = data.with_columns(pl.col('ParentTexts').cast(pl.List(pl.String)))
            if self.model_inference == 'transformers':
                results = finetune.get_predictions("stance-classification", data, self.stance_detection_finetune_kwargs, model_kwargs=self.stance_detection_model_kwargs)
            elif self.model_inference == 'vllm':
                results = llms.get_vllm_predictions("stance-classification", data, self.stance_detection_finetune_kwargs, verbose=self.verbose)
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
            .agg(pl.len().alias('count'))
        embeddings = self._get_embeddings(target_df['Target'], model=embedding_model)
        target_mapper = utils.get_similar_target_mapper(embeddings, target_df)
        documents_df = documents_df.with_columns(
            pl.col('Targets').list.eval(pl.element().replace(target_mapper)).list.unique()
        )
        return documents_df

    def _topic_model(self, targets, embeddings, embedding_model, kwargs):
        if 'show_progress_bars' not in kwargs:
            kwargs['show_progress_bars'] = self.verbose
        clusterer = toponymy.ToponymyClusterer(min_clusters=4)
        topic_model = toponymy.Toponymy(
            llm_wrapper=None,
            text_embedding_model=embedding_model,
            clusterer=clusterer,
            object_description="documents",
            corpus_description="document corpus",
            **kwargs
        )
        try:
            from cuml.manifold.umap import UMAP
            clusterable_vectors = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine').fit_transform(embeddings)
        except ImportError:
            from umap import UMAP
            clusterable_vectors = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine').fit_transform(embeddings)

        # running toponymy fit method except topic naming step which we do ourselves
        exemplar_method = "central"
        keyphrase_method = "information_weighted"
        subtopic_method = "central"
        topic_model.cluster_layers_, topic_model.cluster_tree_ = topic_model.clusterer.fit_predict(
            clusterable_vectors,
            embeddings,
            topic_model.layer_class,
            show_progress_bar=topic_model.show_progress_bars,
            exemplar_delimiters=topic_model.exemplar_delimiters
        )

        # Initialize other data structures
        self.topic_names_ = [[]] * len(topic_model.cluster_layers_)
        topic_model.topic_name_vectors_ = [np.array([])] * len(topic_model.cluster_layers_)
        detail_levels = np.linspace(
            topic_model.lowest_detail_level,
            topic_model.highest_detail_level,
            len(topic_model.cluster_layers_),
        )

        # Get exemplars for layer 0 first and build keyphrase matrix
        if (
            hasattr(topic_model.cluster_layers_[0], "object_to_text_function")
            and topic_model.cluster_layers_[0].object_to_text_function is not None
        ):
            # Non-text objects: use exemplars to build keyphrase matrix
            exemplars, exemplar_indices = topic_model.cluster_layers_[0].make_exemplar_texts(
                targets,
                embeddings,
            )

            # Create aligned text list
            aligned_texts = [""] * len(targets)  # Empty strings for non-exemplars
            for cluster_idx, cluster_exemplars in enumerate(exemplars):
                for exemplar_idx, exemplar_text in zip(
                    exemplar_indices[cluster_idx], cluster_exemplars
                ):
                    aligned_texts[exemplar_idx] = exemplar_text

            # Build keyphrase matrix from aligned texts
            (
                topic_model.object_x_keyphrase_matrix_,
                topic_model.keyphrase_list_,
                topic_model.keyphrase_vectors_,
            ) = topic_model.keyphrase_builder.fit_transform(aligned_texts)
        else:
            # Text objects: build keyphrase matrix directly from objects
            (
                topic_model.object_x_keyphrase_matrix_,
                topic_model.keyphrase_list_,
                topic_model.keyphrase_vectors_,
            ) = topic_model.keyphrase_builder.fit_transform(targets)
            # Still need to generate exemplars for layer 0
            topic_model.cluster_layers_[0].make_exemplar_texts(
                targets,
                embeddings,
            )

        if topic_model.keyphrase_vectors_ is None:
            # If the keyphrase vectors are None, we need to generate them
            topic_model.keyphrase_vectors_ = topic_model.embedding_model.encode(
                topic_model.keyphrase_list_,
                show_progress_bar=topic_model.show_progress_bars,
            )

        # Iterate through the layers and build the topic names
        for i, layer in tqdm(
            enumerate(topic_model.cluster_layers_),
            desc=f"Building topic names by layer",
            disable=not topic_model.show_progress_bars,
            total=len(topic_model.cluster_layers_),
            unit="layer",
        ):
            if i > 0:  # Skip layer 0 exemplars as we already did them
                layer.make_exemplar_texts(
                    targets,
                    embeddings,
                    method=exemplar_method,
                )

            layer.make_keyphrases(
                topic_model.keyphrase_list_,
                topic_model.object_x_keyphrase_matrix_,
                topic_model.keyphrase_vectors_,
                topic_model.embedding_model,
                method=keyphrase_method,
            )

            if i > 0:
                if not hasattr(topic_model.cluster_layers_[0], "topic_name_embeddings"):
                    topic_model.cluster_layers_[0].embed_topic_names(topic_model.embedding_model)

                layer.make_subtopics(
                    topic_model.topic_names_[0],
                    topic_model.cluster_layers_[0].cluster_labels,
                    topic_model.cluster_layers_[0].topic_name_embeddings,
                    topic_model.embedding_model,
                    method=subtopic_method,
                )

            topic_model.topic_name_vectors_[i] = layer.make_topic_name_vector()

        return topic_model.topic_name_vectors_


    def get_base_targets(self, docs, embedding_model=None, text_column='text', parent_text_column='parent_text'):
        if embedding_model is None:
            embedding_model = self._get_embedding_model()

        if isinstance(docs, list):
            documents_df = pl.DataFrame({text_column: docs}).with_row_index(name='ID')
        elif isinstance(docs, pl.DataFrame):
            documents_df = docs
            assert text_column in documents_df.columns, f"docs must have a '{text_column}' column if it is a dataframe"
            if 'ID' not in documents_df.columns:
                documents_df = documents_df.with_row_index(name='ID')

        stance_targets = self._ask_llm_stance_target(documents_df[text_column])
        documents_df = documents_df.with_columns(pl.Series(name='Targets', values=stance_targets, dtype=pl.List(pl.String)))

        # remove bad targets
        target_df = documents_df.explode('Targets').rename({'Targets': 'Target'})
        target_df = utils.remove_bad_targets(target_df)
        documents_df = documents_df.drop('Targets')\
            .join(
                target_df.select(['ID', 'Target']).group_by('ID').agg(pl.col('Target')).rename({'Target': 'Targets'}),
                on='ID',
                how='left'
            )

        documents_df = documents_df.with_columns(self._filter_document_similar_targets(documents_df['Targets'], embedding_model=embedding_model))
        
        return documents_df

    def fit_transform(self, docs: Union[List[str], pl.DataFrame], embedding_cache: pl.DataFrame = None, toponymy_kwargs={}, text_column='text', parent_text_column='parent_text') -> pl.DataFrame:

        if embedding_cache is not None:
            assert isinstance(embedding_cache, pl.DataFrame), "embedding_cache must be a polars DataFrame"
            assert 'text' in embedding_cache.columns, "embedding_cache must have a 'text' column"
            assert 'embedding' in embedding_cache.columns, "embedding_cache must have an 'embedding' column"
            assert isinstance(embedding_cache.schema['embedding'], pl.Array), "embedding_cache column 'embedding' must be an array of floats"
            assert isinstance(embedding_cache.schema['text'], pl.String), "embedding_cache column 'text' must be a string"
            self.embedding_cache_df = embedding_cache
        
        if isinstance(docs, list):
            document_df = pl.DataFrame({text_column: docs}).with_row_index(name='ID')
        elif isinstance(docs, pl.DataFrame):
            document_df = docs
            assert text_column in document_df.columns, f"docs argument must have a '{text_column}' column if it is a dataframe"
            if 'ID' not in document_df.columns:
                document_df = document_df.with_row_index(name='ID')
        
        embed_model = self._get_embedding_model()
        if 'Targets' not in document_df.columns:
            logger.info("Getting base targets")
            document_df = self.get_base_targets(docs, embedding_model=embed_model, text_column=text_column, parent_text_column=parent_text_column)

            # cluster initial stance targets
            base_target_df = document_df.explode('Targets').unique('Targets').drop_nulls()[['Targets']]
        else:
            assert isinstance(document_df.schema['Targets'], pl.List), "Targets column must be a list of strings"
            base_target_df = document_df.explode('Targets').unique('Targets').drop_nulls('Targets')[['Targets']]
        
        logger.info("Fitting topic model")
        targets = base_target_df['Targets'].to_list()
        embeddings = self._get_embeddings(targets, model=embed_model)
        topics = self._topic_model(targets, embeddings, embed_model, toponymy_kwargs)
        no_outliers_df = topic_info.filter(pl.col('Topic') != -1)
        if len(no_outliers_df) > 0:
            logger.info("Getting higher level stance targets")
            stance_targets = prompting.ask_llm_target_aggregate(self.generator, no_outliers_df.select(['Representative_Docs', 'Representation']).to_dicts())
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
            document_df = self.get_stance(document_df, text_column=text_column, parent_text_column=parent_text_column)

        logger.info("Getting target info")
        self.target_info = document_df.explode('Targets')\
            .select('Targets')\
            .drop_nulls()\
            .rename({'Targets': 'Target'})\
            .group_by('Target')\
            .len()\
            .rename({'len': 'Count'})
        # join to topic df to get topic info
        if len(no_outliers_df) > 0:
            self.target_info = self.target_info.join(
                no_outliers_df.drop(['Name', 'Count']).explode('noun_phrases').drop_nulls('noun_phrases').rename({'noun_phrases': 'Target'}),
                on='Target',
                how='left'
            )

        logger.info("Done")
        return document_df
    
    def get_stance(self, document_df: pl.DataFrame, text_column='text', parent_text_column='parent_text') -> pl.DataFrame:
        if 'ID' not in document_df.columns:
            document_df = document_df.with_row_index(name='ID')

        target_df = document_df.explode('Targets').rename({'Targets': 'Target'})
        parent_docs = target_df[parent_text_column] if parent_text_column in target_df.columns else None
        target_stance = self._ask_llm_stance(target_df[text_column], target_df['Target'], parent_docs=parent_docs)
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

    def get_target_info(self):
        return self.target_info
    
    def get_topic_model(self):
        return self.topic_model

    def _get_generator(self, lazy=False):
        if self.model_inference == 'transformers':
            return llms.Transformers(self.model_name, self.model_kwargs, self.tokenizer_kwargs, lazy=lazy)
        elif self.model_inference == 'vllm':
            return llms.VLLM(self.model_name, lazy=lazy, verbose=self.verbose)
        else:
            raise ValueError(f"LLM library '{self.model_inference}' not implemented")
        

                

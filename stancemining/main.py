import functools
import logging
from typing import List, Union

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

from stancemining import llms, finetune, prompting, utils

logger = logging.getLogger('StanceMining')

class StanceMining:
    """Class for performing stance mining on a set of documents.
    
    Args:
        stance_target_type (str): Type of stance target to extract, either 'noun-phrases' or 'claims'.
        llm_method (str): Method to use for LLM inference, either 'prompting' or 'finetuned'.
        model_inference (str): Inference method for the LLM, either 'vllm' or 'transformers'.
        model_name (str): Name of the base LLM model, which will be used for target cluster naming, and, if llm_method is 'prompting', for stance target extraction and detection.
        model_kwargs (dict): Additional keyword arguments for the LLM model.
        tokenizer_kwargs (dict): Additional keyword arguments for the tokenizer.
        stance_detection_model (str): Name of the stance detection model to use. Defaults to 'bendavidsteel/SmolLM2-360M-Instruct-stance-detection' if not provided.
        stance_detection_finetune_kwargs (dict): Keyword arguments for the fine-tuned stance detection model.
        stance_detection_model_kwargs (dict): Keyword arguments for stance detection model inference.
        stance_detection_generation_kwargs (dict): Keyword arguments for generate method during stance detection.
        target_extraction_model (str): Name of the target extraction model to use. Defaults to 'bendavidsteel/SmolLM2-360M-Instruct-target-extraction' if not provided.
        target_extraction_finetune_kwargs (dict): Keyword arguments for the fine-tuned target extraction model.
        target_extraction_generation_kwargs (dict): Keyword arguments for generate method during target extraction.
        target_extraction_model_kwargs (dict): Keyword arguments for target extraction model inference.
        embedding_model (str): Name of the embedding model to use for target extraction.
        embedding_model_inference (str): Inference method for the embedding model, either 'vllm' or 'transformers'.
        topic_model (str): Topic model to use for clustering targets, either 'bertopic' or 'toponymy'.
        cosine_similarity_threshold (float): Cosine similarity threshold for deduplicating targets. Defaults to 0.8.
        verbose (bool): Whether to enable verbose logging. Defaults to False.
    """

    def __init__(
            self, 
            stance_target_type='noun-phrases',
            llm_method='finetuned',
            model_inference='vllm', 
            model_name='microsoft/Phi-4-mini-instruct', 
            model_kwargs={}, 
            tokenizer_kwargs={},
            stance_detection_model=None,
            stance_detection_finetune_kwargs={},
            stance_detection_model_kwargs={},
            stance_detection_generation_kwargs={},
            target_extraction_model=None,
            target_extraction_finetune_kwargs={},
            target_extraction_model_kwargs={},
            target_extraction_generation_kwargs={},
            embedding_model='intfloat/multilingual-e5-small',
            embedding_model_inference='vllm',
            topic_model='bertopic',
            cosine_similarity_threshold=0.8,
            verbose=False,
            use_embedding_cache=True,
        ):
        """Initialize the StanceMining class.
        """
        assert stance_target_type in ['noun-phrases', 'claims'], f"Stance target type must be either 'noun-phrases' or 'claims', not '{stance_target_type}'"
        assert llm_method in ['prompting', 'finetuned'], f"LLM method must be either 'prompting' or 'finetuned', not '{llm_method}'"
        self.stance_target_type = stance_target_type
        self.llm_method = llm_method
        assert model_inference in ['vllm', 'transformers', 'anthropic'], f"Model inference method must be either 'vllm', 'transformers' or 'anthropic', not '{model_inference}'"
        self.model_inference = model_inference
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        if 'device_map' not in self.model_kwargs and self.model_inference == 'transformers':
            self.model_kwargs['device_map'] = 'auto'
        if 'torch_dtype' not in self.model_kwargs and self.model_inference == 'transformers':
            self.model_kwargs['torch_dtype'] = 'auto'

        self.tokenizer_kwargs = tokenizer_kwargs

        if stance_detection_model is None:
            stance_detection_model = 'bendavidsteel/SmolLM2-360M-Instruct-stance-detection'

        if 'model_path' not in stance_detection_finetune_kwargs and 'hf_model' not in stance_detection_finetune_kwargs:
            stance_detection_finetune_kwargs['hf_model'] = stance_detection_model
        if 'classification_method' not in stance_detection_finetune_kwargs:
            stance_detection_finetune_kwargs['classification_method'] = 'head'
        if 'prompting_method' not in stance_detection_finetune_kwargs:
            stance_detection_finetune_kwargs['prompting_method'] = 'stancemining'
        self.stance_detection_finetune_kwargs = stance_detection_finetune_kwargs

        if 'device_map' not in stance_detection_model_kwargs and self.model_inference == 'transformers':
            stance_detection_model_kwargs['device_map'] = 'auto'
        self.stance_detection_model_kwargs = stance_detection_model_kwargs
        self.stance_detection_generation_kwargs = stance_detection_generation_kwargs

        if target_extraction_model is None:
            if self.stance_target_type == 'noun-phrases':
                target_extraction_model = 'bendavidsteel/SmolLM2-360M-Instruct-stance-target-extraction'
            elif self.stance_target_type == 'claims':
                target_extraction_model = 'bendavidsteel/SmolLM2-360M-Instruct-claim-extraction'
            else:
                raise ValueError(f"Unknown stance target type: {self.stance_target_type}. Must be either 'noun-phrases' or 'claims'.")

        if 'model_path' not in target_extraction_finetune_kwargs and 'hf_model' not in target_extraction_finetune_kwargs:
            target_extraction_finetune_kwargs['hf_model'] = target_extraction_model
        if 'generation_method' not in target_extraction_finetune_kwargs:
            target_extraction_finetune_kwargs['generation_method'] = 'list'
        if 'prompting_method' not in target_extraction_finetune_kwargs:
            target_extraction_finetune_kwargs['prompting_method'] = 'stancemining'
        self.target_extraction_finetune_kwargs = target_extraction_finetune_kwargs
        
        if 'device_map' not in target_extraction_model_kwargs and self.model_inference == 'transformers':
            target_extraction_model_kwargs['device_map'] = 'auto'
        self.target_extraction_model_kwargs = target_extraction_model_kwargs
        self.target_extraction_generation_kwargs = target_extraction_generation_kwargs

        self.verbose = verbose
        if self.verbose:
            logger.setLevel("DEBUG")
        else:
            logger.setLevel("WARNING")

        self.embedding_model = embedding_model
        assert embedding_model_inference in ['vllm', 'sentence-transformers'], f"Embedding model inference method must be either 'vllm' or 'sentence-transformers', not '{embedding_model_inference}'"
        self.embedding_model_inference = embedding_model_inference
        self.embedding_cache_df = pl.DataFrame({'text': [], 'embedding': []}, schema={'text': pl.String, 'embedding': pl.Array(pl.Float32, 384)})

        self.cosine_similarity_threshold = cosine_similarity_threshold

        assert topic_model in ['bertopic', 'toponymy'], f"Topic model must be either 'bertopic' or 'toponymy', not '{topic_model}'"
        self.topic_model = topic_model

        self.use_embedding_cache = use_embedding_cache

    def _generate_higher_level_targets(self, document_df: pl.DataFrame, embed_model, topic_model_kwargs, max_layers):
        logger.info("Fitting topic model")
        # get unique targets where most common targets are first
        base_target_df = document_df.select(['Targets'])\
            .explode('Targets')\
            .rename({'Targets': 'Target'})\
            .drop_nulls('Target')\
            .group_by('Target')\
            .len()\
            .sort('len', descending=True)\
            .drop('len')
        doc_targets = base_target_df['Target'].to_list()
        cluster_layers, cluster_df = self._topic_model(doc_targets, embed_model, topic_model_kwargs, max_layers)
        if len(cluster_df) > 0:
            base_target_cluster_df = base_target_df.with_columns(
                pl.Series(name='Clusters', values=np.stack(cluster_layers, axis=-1), dtype=pl.Array(pl.Int64, len(cluster_layers)))\
                    .cast(pl.List(pl.Int64))\
                    .list.filter(pl.element() != -1)  # filter out -1 clusters
            ).explode('Clusters').rename({'Clusters': 'Cluster'})
            logger.info("Getting higher level stance targets")
            stance_targets = self._ask_llm_target_aggregate(cluster_df.select(['Exemplars', 'Keyphrases']).to_dicts())
            cluster_df = cluster_df.with_columns(
                pl.Series(name='ClusterTargets', values=stance_targets, dtype=pl.List(pl.String))
            )
            cluster_df = cluster_df.with_columns(self._filter_document_similar_targets(cluster_df['ClusterTargets'], embedding_model=embed_model))

            cluster_df = cluster_df.with_columns(pl.col('ClusterTargets').fill_null([]))

            logger.info("Mapping targets to topics")
            # map documents to new noun phrases via topics
            target_df = document_df.explode('Targets').rename({'Targets': 'Target'}).select(['ID', 'Target'])
            target_df = target_df.join(base_target_cluster_df, on='Target', how='left')
            target_df = target_df.join(cluster_df.select(['Cluster', 'ClusterTargets']), on='Cluster', how='left')

            logger.info("Grouping new targets to document IDs")
            new_target_df = target_df.explode('ClusterTargets')\
                        .drop_nulls('ClusterTargets')\
                        .group_by('ID')\
                        .agg(pl.col('ClusterTargets').alias('NewTargets'))

            logger.info("Joining new targets to documents")
            document_df = document_df.join(
                new_target_df,
                on='ID', 
                how='left',
                maintain_order='left'
            )
            
            logger.info("Combining base and topic targets")
            document_df = document_df.with_columns(
                pl.when(pl.col('NewTargets').is_not_null())\
                    .then(pl.concat_list(pl.col('Targets'), pl.col('NewTargets')))\
                    .otherwise(pl.col('Targets')))
            document_df = document_df.drop(['NewTargets'])

        return document_df, cluster_df

    def fit_transform(
            self, 
            docs: Union[List[str], pl.DataFrame],
            text_column: str='text', 
            parent_text_column: str='parent_text',
            get_stance: bool=True,
            generate_targets: bool=True,
            generate_higher_level_targets: bool=True,
            deduplicate_all_targets: bool=True,
            targets: List[str]=[],
            topic_model_kwargs: dict={},
            embedding_cache: pl.DataFrame = None,
            max_layers: int=2
        ) -> pl.DataFrame:
        """Find stances from the given documents.

        Args:
            docs (Union[List[str], pl.DataFrame]): List of documents or a DataFrame containing documents.
            text_column (str): Name of the column containing the text in the DataFrame,
                if `docs` is a DataFrame. Defaults to 'text'.
            parent_text_column (str): Name of the column containing the parent text in the DataFrame
                if `docs` is a DataFrame. Defaults to 'parent_text'.
            get_stance (bool): Whether to get stance classifications for the targets. Defaults to True.
            generate_targets (bool): Whether to generate stance targets from the documents. Defaults to True.
            generate_higher_level_targets (bool): Whether to generate higher-level stance targets
                using topic modeling. Defaults to True.
            deduplicate_all_targets (bool): Whether to deduplicate all targets using embedding similarity. Defaults to True.
            targets (List[str]): List of stance targets to use if not generating them.
                If `generate_targets` is True, this should be an empty list.
            topic_model_kwargs (dict): Additional keyword arguments for the topic model.
            embedding_cache (pl.DataFrame): Optional cache of embeddings to use for the documents.
                Should be a polars DataFrame with 'text' and 'embedding' columns.
            max_layers (int): Maximum number of hierarchical topic model layers to use when generating higher-level targets. Defaults to 2.

        Returns:
            pl.DataFrame: DataFrame containing the documents with their stance targets and classifications.
        """
        if not generate_targets:
            assert len(targets) > 0 or (isinstance(docs, pl.DataFrame) and 'Targets' in docs.columns), "If not generating targets, you must provide a list of targets or 'Targets' column in the DataFrame."

        if targets and generate_targets:
            logger.info("Using provided targets in addition to generating targets")

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
            assert text_column in document_df.columns, f"docs argument must have a '{text_column}' column if it is a dataframe, found columns: {document_df.columns}"
            if 'ID' not in document_df.columns:
                document_df = document_df.with_row_index(name='ID')
        
        logger.info("Loading embedding model...")
        embed_model = None
        if 'Targets' not in document_df.columns:
            if generate_targets:
                logger.info("Getting base targets")
                embed_model = self._get_embedding_model()
                document_df = self.get_base_targets(docs, embedding_model=embed_model, text_column=text_column, parent_text_column=parent_text_column)
        
            if targets and not generate_targets:
                logger.info("Using provided targets")
                document_df = document_df.with_columns(pl.lit(targets).alias('Targets'))
            elif targets and generate_targets:
                logger.info("Adding provided targets to generated targets")
                document_df = document_df.with_columns(
                    pl.col('Targets').list.concat(pl.lit(targets))
                )
        else:
            assert isinstance(document_df.schema['Targets'], pl.List), "Targets column must be a list of strings"
            logger.info("Using existing targets in DataFrame")
        
        # cluster initial stance targets
        logger.debug("Exploding targets to get unique targets")
        if generate_higher_level_targets:
            if embed_model is None:
                embed_model = self._get_embedding_model()
            document_df, cluster_df = self._generate_higher_level_targets(document_df, embed_model, topic_model_kwargs, max_layers)

        if generate_targets and deduplicate_all_targets:
            logger.info("Removing similar stance targets")
            # remove targets that are too similar
            if embed_model is None:
                embed_model = self._get_embedding_model()
            document_df = self._filter_all_similar_targets(document_df, embed_model)

        if get_stance:
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
        if generate_higher_level_targets and len(cluster_df) > 0:
            self.target_info = self.target_info.join(
                cluster_df.drop(['Cluster']).explode('ClusterTargets').drop_nulls('ClusterTargets').rename({'ClusterTargets': 'Target'}),
                on='Target',
                how='left'
            )

        logger.info("Done")
        return document_df
    

    def _get_embedding_model(self):
        if self.embedding_model_inference == 'vllm':
            try:
                model = utils.VLLMEmbedder(model=self.embedding_model, kwargs={'gpu_memory_utilization': 0.1})
            except ImportError:
                logger.warning("VLLM is not installed, using SentenceTransformer for embeddings.")
                model = SentenceTransformer(self.embedding_model)
        elif self.embedding_model_inference == 'sentence-transformers':
            model = SentenceTransformer(self.embedding_model)
        else:
            raise ValueError(f"Embedding model inference method '{self.embedding_model_inference}' not implemented")
        return model

    def _get_embeddings(self, docs: Union[List[str], pl.Series], model=None) -> np.ndarray:
        if model is None:
            model = self._get_embedding_model()
        if self.use_embedding_cache:
            # check for cached embeddings
            if isinstance(docs, pl.Series):
                document_df = docs.rename('text').to_frame()
            else:
                document_df = pl.DataFrame({'text': docs})
            document_df = document_df.join(self.embedding_cache_df, on='text', how='left', maintain_order='left')
            missing_docs = document_df.unique('text').filter(pl.col('embedding').is_null())
            if len(missing_docs) > 0:
                logger.debug(f"Computing embeddings for {len(missing_docs)} missing documents")
                new_embeddings = model.encode(missing_docs['text'].to_list(), show_progress_bar=self.verbose).astype(np.float32)
                
                missing_docs = missing_docs.with_columns(pl.Series(name='embedding', values=new_embeddings))
                # cache embeddings
                logger.debug("Updating embedding cache")
                self.embedding_cache_df = pl.concat([self.embedding_cache_df, missing_docs], how='diagonal_relaxed')
                # add new embeddings to document_df
                document_df = document_df.join(missing_docs, on='text', how='left', maintain_order='left')\
                    .with_columns(pl.coalesce(['embedding', 'embedding_right']))\
                    .drop('embedding_right')
            embeddings = document_df['embedding'].to_numpy()
        else:
            embeddings = model.encode(docs, show_progress_bar=self.verbose).astype(np.float32)

        return embeddings

    def _ask_llm_target_aggregate(self, clusters: List[dict]):
        llm = self._get_llm()
        if self.stance_target_type == 'noun-phrases':
            aggregations = prompting.ask_llm_noun_phrase_aggregate(llm, clusters)
        elif self.stance_target_type == 'claims':
            aggregations = prompting.ask_llm_claim_aggregate(llm, clusters)
        else:
            raise ValueError(f"Unrecognised self.stance_target_type value: {self.stance_target_type}")
        llm.unload_model()
        return aggregations

    def _ask_llm_stance_target(self, docs: List[str]):
        num_samples = 3
        if self.llm_method == 'prompting':
            llm = self._get_llm()
            targets = prompting.ask_llm_zero_shot_stance_target(llm, docs, {'num_samples': num_samples})
        elif self.llm_method == 'finetuned':
            df = pl.DataFrame({'Text': docs})

            # TODO add guided decoding to generate list

            task_type = 'topic-extraction' if self.stance_target_type == 'noun-phrases' else 'claim-extraction'

            if self.model_inference == 'transformers':
                results = finetune.get_predictions(task_type, df, self.target_extraction_finetune_kwargs, model_kwargs=self.target_extraction_model_kwargs, generate_kwargs=self.target_extraction_generation_kwargs)
            elif self.model_inference == 'vllm':
                results = llms.get_vllm_predictions(task_type, df, self.target_extraction_finetune_kwargs, verbose=self.verbose, model_kwargs=self.target_extraction_model_kwargs, generate_kwargs=self.target_extraction_generation_kwargs)
            else:
                raise ValueError(f"Cannot run finetuned LLM with model_inference method: {self.model_inference}")

            if isinstance(results[0], str):
                targets = [[r] for r in results]
            else:
                targets = results
        else:
            raise ValueError(f"Unrecognised self.llm_method value: {self.llm_method}")
        target_df = pl.DataFrame({'Targets': targets})
        target_df = target_df.with_columns(utils._filter_stance_targets(target_df['Targets']))
        return target_df['Targets'].to_list()

    def _ask_llm_stance(self, docs, stance_targets, parent_docs=None):
        task = 'stance-classification' if self.stance_target_type == 'noun-phrases' else 'claim-entailment-7way'
        if self.llm_method == 'prompting':
            llm = self._get_llm()
            assert parent_docs is None, "Parent documents not supported for prompting stance detection"
            return prompting.ask_llm_zero_shot_stance(llm, docs, stance_targets, verbose=self.verbose)
        elif self.llm_method == 'finetuned':
            data = pl.DataFrame({'Text': docs, 'Target': stance_targets, 'ParentTexts': parent_docs})
            if isinstance(data.schema['ParentTexts'], pl.String):
                # convert to list
                data = data.with_columns(pl.col('ParentTexts').cast(pl.List(pl.String)))
            if self.model_inference == 'transformers':
                results = finetune.get_predictions(task, data, self.stance_detection_finetune_kwargs, model_kwargs=self.stance_detection_model_kwargs)
            elif self.model_inference == 'vllm':
                results = llms.get_vllm_predictions(task, data, self.stance_detection_finetune_kwargs, verbose=self.verbose, model_kwargs=self.stance_detection_model_kwargs, generate_kwargs=self.stance_detection_generation_kwargs)
            else:
                raise ValueError(f"Cannot run finetuned LLM with model_inference method: {self.model_inference}")
            results = [r.upper() for r in results]
            return results

    def _filter_document_similar_targets(self, phrases_list: pl.Series, embedding_model=None, similarity_threshold: float = 0.8) -> pl.Series:
        """Filter similar phrases.
        
        Filter out similar phrases from a list of lists based on embedding similarity,
        only comparing phrases within the same sublist.
        
        Args:
            phrases_list: List of lists containing phrases to filter
            embedding_model: Embedding model to use for computing embeddings
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

        filter_phrases = functools.partial(utils._filter_phrases, similarity_threshold=self.cosine_similarity_threshold)

        df = df.with_columns(
            pl.when(pl.col('target_len') > 1)
                .then(pl.col('target_embeds').map_elements(filter_phrases, return_dtype=pl.List(pl.String)))\
                .otherwise(pl.col('target_embeds'))\
            .alias('targets')
        )

        return df['targets'].rename(col_name) 
    
    def _filter_all_similar_targets(self, documents_df: pl.DataFrame, embedding_model=None) -> pl.DataFrame:
        """Filter similar targets.
        
        Filter out similar targets from a DataFrame of documents based on embedding similarity, 
        only comparing targets within the same document.
        
        Args:
            documents_df: DataFrame containing 'Targets' column with lists of targets
            embedding_model: Embedding model to use for computing embeddings

        Returns:
            DataFrame with 'Targets' column filtered for similar targets
        """
        logger.debug("Getting target counts for filtering")
        target_df = documents_df.select('Targets')\
            .explode('Targets')\
            .drop_nulls()\
            .rename({'Targets': 'Target'})\
            .group_by('Target')\
            .agg(pl.len().alias('count'))
        logger.debug("Create mapping of small count targets to larger count similar targets")
        if self.stance_target_type == 'noun-phrases':
            max_distance = 0.2
        elif self.stance_target_type == 'claims':
            max_distance = 0.1
        batch_size = 2500000
        if len(target_df) <= batch_size:
            target_mapper = utils._get_similar_target_mapper(target_df, embedding_model=embedding_model, max_distance=max_distance)
        else:
            target_mapper = utils._get_similar_target_mapper_batch(target_df, embedding_model=embedding_model, max_embedding_distance=max_distance, batch_size=batch_size)
        logger.debug("Replacing small count targets with larger count similar targets")
        documents_df = documents_df.with_columns(
            pl.col('Targets').list.eval(pl.element().replace(target_mapper)).list.unique()
        )
        return documents_df

    def _topic_model(self, targets, embedding_model, kwargs, max_layers):
        if self.topic_model == 'toponymy':
            results = self._toponymy_topic_model(targets, embedding_model, kwargs, max_layers)
        elif self.topic_model == 'bertopic':
            results = self._bertopic_topic_model(targets, embedding_model, kwargs)
        else:
            raise ValueError(f"Unknown topic model: {self.topic_model}. Supported models: 'toponymy', 'bertopic'.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory if using CUDA

        return results
        
    def _bertopic_topic_model(self, targets, embedding_model, kwargs):
        try:
            from bertopic import BERTopic
            from bertopic.backend._utils import select_backend
        except ImportError:
            raise ImportError("BERTopic package is not installed. Please install it with 'pip install bertopic'.")

        if 'verbose' not in kwargs:
            kwargs['verbose'] = self.verbose
        if 'umap_model' not in kwargs:
            umap_kwargs = {
                "n_components": 5, # higher dimensionality for more sparse clusters,
                "verbose": self.verbose
            }
            if torch.cuda.is_available():
                try:
                    # use pacmap instead https://github.com/hyhuang00/ParamRepulsor
                    import parampacmap
                    umap_model = parampacmap.ParamPaCMAP(**umap_kwargs)
                except ImportError:
                    try:
                        logger.warning("ParamRepulsor is not installed, using pacmap for dimensionality reduction.")
                        import pacmap
                        umap_model = pacmap.PaCMAP(**umap_kwargs)
                    except ImportError:
                        raise ImportError("Neither ParamRepulsor nor pacmap is installed. Please install one of them with 'pip install parampacmap' or 'pip install pacmap'.")
            else:
                try:
                    import pacmap
                    umap_model = pacmap.PaCMAP(**umap_kwargs)
                except ImportError:
                    raise ImportError("Pacmap is not installed. Please install it with 'pip install pacmap'.")
            kwargs['umap_model'] = umap_model
        if 'hdbscan_model' not in kwargs:
            if torch.cuda.is_available():
                from cuml.cluster import HDBSCAN
                kwargs['hdbscan_model'] = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True, verbose=self.verbose, random_state=42)
        topic_model = BERTopic(**kwargs)
        doc_ids = range(len(targets))

        import pandas as pd
        document_df = pd.DataFrame({"Document": targets, "ID": doc_ids, "Topic": None})

        self.embedding_model = select_backend(
            self.embedding_model, language=topic_model.language, verbose=self.verbose
        )

        batch_size = 2500000
        if len(targets) <= batch_size:
            embeddings = self._get_embeddings(targets, model=embedding_model)

            # Reduce dimensionality and fit UMAP model
            umap_embeddings = topic_model._reduce_dimensionality(embeddings)
        else:
            # batch process to try to avoid OOM
            embeddings = None
            umap_embeddings = None
            for i in tqdm(range(0, len(targets), batch_size), desc="Computing embeddings for targets in batches"):
                batch_texts = targets[i:i+batch_size]
                batch_embeddings = self._get_embeddings(batch_texts, model=embedding_model)

                if i == 0:
                    batch_umap_embeddings = umap_model.fit_transform(batch_embeddings)
                else:
                    batch_umap_embeddings = umap_model.transform(batch_embeddings)
                if umap_embeddings is None:
                    umap_embeddings = batch_umap_embeddings
                else:
                    umap_embeddings = np.vstack([umap_embeddings, batch_umap_embeddings])

        logger.debug("Successfully computed UMAP embeddings, now clustering with HDBSCAN")
        document_df, _ = topic_model._cluster_embeddings(umap_embeddings, document_df)
        logger.debug("Successfully clustered embeddings")

        # Sort and Map Topic IDs by their frequency
        if not topic_model.nr_topics:
            document_df = topic_model._sort_mappings_by_frequency(document_df)

        # Extract topics by calculating c-TF-IDF, reduce topics if needed, and get representations.
        logger.debug("Extracting topics and representations")
        topic_model._extract_topics(
            document_df, embeddings=None, verbose=topic_model.verbose, fine_tune_representation=not topic_model.nr_topics
        )
        if topic_model.nr_topics:
            document_df = topic_model._reduce_topics(document_df)

        # Save the top 3 most representative documents per topic
        topic_model._save_representative_docs(document_df)

        base_target_df = pl.from_pandas(document_df)
        num_representative_docs = 5
        repr_docs, _, _, _ = topic_model._extract_representative_docs(
            topic_model.c_tf_idf_,
            base_target_df.to_pandas(),
            topic_model.topic_representations_,
            nr_samples=500,
            nr_repr_docs=num_representative_docs,
        )
        topic_model.representative_docs_ = repr_docs
        topic_info = topic_model.get_topic_info()
        cluster_df = pl.from_pandas(topic_info)\
            .rename({'Representation': 'Keyphrases', 'Representative_Docs': 'Exemplars', 'Topic': 'Cluster'})\
            .filter(pl.col('Cluster') != -1)

        return [base_target_df['Topic'].to_numpy()], cluster_df

    def _toponymy_topic_model(self, targets, embedding_model, kwargs, max_layers):
        try:
            import toponymy
        except ImportError:
            raise ImportError("Toponymy package is not installed. Please install it with 'pip install toponymy'.")
        if 'show_progress_bars' not in kwargs:
            kwargs['show_progress_bars'] = self.verbose
        clusterer = toponymy.ToponymyClusterer(
            min_clusters=2,
            verbose=self.verbose,
            max_layers=max_layers,
            **kwargs.get('clusterer', {})
        )
        topic_model = toponymy.Toponymy(
            llm_wrapper=None,
            text_embedding_model=embedding_model,
            clusterer=clusterer,
            object_description="documents",
            corpus_description="document corpus",
            **kwargs.get('toponymy', {})
        )
        umap_kwargs = {
            "n_components": 10, # higher dimensionality for more sparse clusters
        }
        if torch.cuda.is_available():
            try:
                # use pacmap instead https://github.com/hyhuang00/ParamRepulsor
                import parampacmap
                umap_model = parampacmap.ParamPaCMAP(**umap_kwargs)
            except ImportError:
                import pacmap
                umap_model = pacmap.PaCMAP(**umap_kwargs)
        else:
            import pacmap
            umap_model = pacmap.PaCMAP(**umap_kwargs)

        embeddings = self._get_embeddings(targets, model=embedding_model)
        clusterable_vectors = umap_model.fit_transform(embeddings)

        # running toponymy fit method except topic naming step which we do ourselves
        exemplar_method = "central"
        keyphrase_method = "information_weighted"
        topic_model.cluster_layers_, topic_model.cluster_tree_ = topic_model.clusterer.fit_predict(
            clusterable_vectors,
            embeddings,
            topic_model.layer_class,
            show_progress_bar=topic_model.show_progress_bars,
            exemplar_delimiters=topic_model.exemplar_delimiters
        )

        # Initialize other data structures
        topic_model.topic_names_ = [[]] * len(topic_model.cluster_layers_)
        topic_model.topic_name_vectors_ = [np.array([])] * len(topic_model.cluster_layers_)

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

        cluster_layer_labels = []
        clusters = []
        num_prev_clusters = 0
        # Iterate through the layers and build the topic names
        for i, layer in tqdm(
            enumerate(topic_model.cluster_layers_),
            desc="Building topic names by layer",
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

            cluster_layer_labels.append(layer.cluster_labels + num_prev_clusters)
            for j, (exemplars, keyphrases) in enumerate(zip(layer.exemplars, layer.keyphrases)):
                clusters.append({
                    'Cluster': j + num_prev_clusters,
                    'Exemplars': exemplars,
                    'Keyphrases': keyphrases,
                })
            num_prev_clusters += len(layer.exemplars)
        cluster_df = pl.DataFrame(clusters)

        return cluster_layer_labels, cluster_df


    def get_base_targets(
            self, 
            docs: Union[List[str], pl.DataFrame], 
            embedding_model=None, 
            text_column='text', 
            parent_text_column='parent_text'
        ) -> pl.DataFrame:
        """Generate stance targets from the given documents.
        
        Args:
            docs (Union[List[str], pl.DataFrame]): List of documents or a DataFrame containing documents.
            embedding_model: Embedding model to use for computing embeddings. If None, uses the default embedding model.
            text_column (str): Name of the column containing the text in the DataFrame,
                if `docs` is a DataFrame. Defaults to 'text'.
            parent_text_column (str): Name of the column containing the parent text in the DataFrame
                if `docs` is a DataFrame. Defaults to 'parent_text'.

        Returns:
            pl.DataFrame: DataFrame containing the documents with their stance targets.
        """
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
            )\
            .with_columns(pl.col('Targets').fill_null([]))  # fill nulls with empty list

        documents_df = documents_df.with_columns(self._filter_document_similar_targets(documents_df['Targets'], embedding_model=embedding_model))
        
        return documents_df


    def get_stance(
            self, 
            document_df: pl.DataFrame, 
            text_column='text', 
            parent_text_column='parent_text'
        ) -> pl.DataFrame:
        """Get stance classifications for the targets in the documents.

        Args:
            document_df (pl.DataFrame): DataFrame containing documents with 'Targets' column.
            text_column (str): Name of the column containing the text in the DataFrame.
                Defaults to 'text'.
            parent_text_column (str): Name of the column containing the parent text in the DataFrame
                Defaults to 'parent_text'.
        
        Returns:
            pl.DataFrame: DataFrame containing the documents with their stance targets and classifications.
        """
        if 'ID' not in document_df.columns:
            document_df = document_df.with_row_index(name='ID')

        target_df = document_df.explode('Targets').drop_nulls('Targets').rename({'Targets': 'Target'})
        parent_docs = target_df[parent_text_column] if parent_text_column in target_df.columns else None
        target_stance = self._ask_llm_stance(target_df[text_column], target_df['Target'], parent_docs=parent_docs)
        target_df = target_df.with_columns(pl.Series(name='stance', values=target_stance))
        
        document_df = document_df.drop('Targets')\
            .join(
                target_df.group_by('ID')\
                    .agg(pl.col('Target').alias('Targets'), pl.col('stance').alias('Stances')),
                on='ID',
                how='left',
                maintain_order='left'
            )\
            .with_columns([
                pl.col('Targets').fill_null([]),
                pl.col('Stances').fill_null([])
            ])
        return document_df

    def get_target_info(self):
        """Get information about the stance targets.
        
        Returns:
            pl.DataFrame: DataFrame containing target information, including counts and topic associations.
        """
        return self.target_info

    def _get_llm(self):
        if self.model_inference == 'transformers':
            return llms.Transformers(self.model_name, self.model_kwargs, self.tokenizer_kwargs)
        elif self.model_inference == 'vllm':
            return llms.VLLM(self.model_name, self.model_kwargs, verbose=self.verbose)
        elif self.model_inference == 'anthropic':
            return llms.Anthropic(self.model_name, self.model_kwargs)
        else:
            raise ValueError(f"LLM library '{self.model_inference}' not implemented")
        

                

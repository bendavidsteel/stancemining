import gc
import logging
import subprocess
from typing import List

from nltk.corpus import stopwords
import numpy as np
import polars as pl
import sklearn.preprocessing
from tqdm import tqdm
import torch
import vllm

logger = logging.getLogger('StanceMining.utils')

class VLLMEmbedder:
        def __init__(self, model: str = "all-MiniLM-L6-v2", kwargs: dict = {}):
            self.llm = vllm.LLM(model=model, task='embed', **kwargs)

        def encode(self, texts: List[str], show_progress_bar: bool = None) -> np.ndarray:
            outputs = self.llm.embed(texts, use_tqdm=show_progress_bar)
            logger.debug(f"Generated {len(outputs)} embeddings with shape {len(outputs[0].outputs.embedding)} using VLLM model {self.llm}")
            embeddings = np.vstack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in outputs])
            logger.debug(f"Combined embeddings into array of shape {embeddings.shape}")
            return embeddings

def cluster_target_embeddings(embeddings, max_distance = 0.2):
    normalized_embeddings = sklearn.preprocessing.normalize(embeddings, axis=1, norm='l2')
    
    try:
        return _cuvl_clustering(normalized_embeddings, max_distance=max_distance)
    except ImportError:
        return _pynndescent_clustering(normalized_embeddings, max_distance=max_distance)

def _propagate_clusters_cupy(cluster_labels, verbose=True):
    import cupy as cp

    labels_range = cp.arange(len(cluster_labels))

    if cp.array_equal(cp.asarray(cluster_labels), labels_range).all():
        return cluster_labels

    for _ in tqdm(range(3), desc="Propagating clusters", disable=not verbose):  # Usually converges in 2-3 iterations
        needs_update = cluster_labels != labels_range
        if not needs_update.any():
            break

        old_labels = cluster_labels[needs_update]
        new_labels = cluster_labels[old_labels]
        cluster_labels[needs_update] = new_labels

        if cp.array_equal(old_labels, new_labels):
            break

        del old_labels
        del new_labels

    return cluster_labels

def _cuvl_clustering(embeddings, max_distance=0.2, batch_size=10000, verbose=True):
    """GPU-accelerated deduplication using cuVS (RAPIDS)."""
    from cuvs.neighbors import cagra
    import cupy as cp

    n_samples = len(embeddings)

    # Check available GPU memory
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    # Convert to GPU array
    embeddings_gpu = cp.asarray(embeddings)
    
    # Build CAGRA index (GPU-accelerated graph-based index)
    index_params = cagra.IndexParams(
        metric="sqeuclidean"
    )
    
    index = cagra.build(index_params, embeddings_gpu)
    
    # Search parameters
    itopk_size = 64  # Number of neighbors to consider
    search_params = cagra.SearchParams(
        itopk_size=itopk_size,
    )
    
    # Initialize cluster labels
    cluster_labels = cp.arange(n_samples)
    
    # Process in batches to find neighbors within threshold
    for i in tqdm(range(0, n_samples, batch_size), desc="Finding neighbors", disable=not verbose):
        batch_end = min(i + batch_size, n_samples)
        batch = embeddings_gpu[i:batch_end]
        
        # Search for k nearest neighbors (adjust k based on expected cluster size)
        k = min(itopk_size, n_samples)  # Adjust based on your needs
        sq_distances, indices = cagra.search(search_params, index, batch, k)
        
        sq_distances = cp.asarray(sq_distances)
        indices = cp.asarray(indices)

        # Filter by distance threshold
        mask = sq_distances < max_distance ** 2
        
        # Merge clusters for points within threshold
        for j in range(batch_end - i):
            valid_neighbors = indices[j][mask[j]]
            if len(valid_neighbors) > 0:
                # Get minimum cluster label among neighbors
                neighbor_labels = cluster_labels[valid_neighbors]
                min_label = cp.min(cp.concatenate([
                    cp.array([cluster_labels[i + j]]), 
                    neighbor_labels
                ]))
                
                # Update all connected points
                cluster_labels[i + j] = min_label
                cluster_labels[valid_neighbors] = min_label
    
    # Propagate cluster assignments (handle transitive closure)
    cluster_labels = _propagate_clusters_cupy(cluster_labels, verbose=verbose)
    
    # Renumber clusters consecutively
    cluster_labels_cpu = cp.asnumpy(cluster_labels)

    # Clean up GPU memory
    del embeddings_gpu, index
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    cp.cuda.Stream.null.synchronize()

    unique_labels = np.unique(cluster_labels_cpu)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    final_labels = np.array([label_map[label] for label in cluster_labels_cpu])
    
    return final_labels

def _pynndescent_clustering(normalized_embeddings, max_distance=0.2, n_neighbors=30, batch_size=100000):
    """Alternative: Process in batches with PyNNDescent for very large datasets.
    This addresses potential memory issues with the full graph approach.
    """
    try:
        import pynndescent
    except ImportError:
        raise ImportError("pynndescent is not installed. Please install it with `pip install pynndescent`.")
    n_samples = len(normalized_embeddings)
    
    # Initialize with each point in its own cluster
    cluster_labels = np.arange(n_samples)
    
    # Build index on full dataset but query in batches
    print("Building PyNNDescent index...")
    index = pynndescent.NNDescent(
        normalized_embeddings,
        n_neighbors=min(n_neighbors, n_samples - 1),
        metric='euclidean',
        low_memory=True,
        compressed=True,
        n_jobs=-1
    )
    
    print("Querying for neighbors in batches...")
    # Process queries in batches to control memory usage
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch = normalized_embeddings[i:batch_end]
        
        # Query the index
        neighbors, distances = index.query(batch, k=n_neighbors)
        
        # Process each point in the batch
        for j, (point_neighbors, point_distances) in enumerate(zip(neighbors, distances)):
            global_idx = i + j
            
            # Find neighbors within threshold
            valid_mask = point_distances < max_distance
            valid_neighbors = point_neighbors[valid_mask]
            
            if len(valid_neighbors) > 0:
                # Find minimum cluster label
                all_labels = np.append(cluster_labels[valid_neighbors], cluster_labels[global_idx])
                min_label = np.min(all_labels)
                
                # Update labels
                cluster_labels[global_idx] = min_label
                cluster_labels[valid_neighbors] = min_label
        
        if (i + batch_size) % 500000 == 0:
            print(f"Processed {min(i + batch_size, n_samples)}/{n_samples} embeddings")
    
    # Consolidate cluster labels (transitive closure)
    print("Consolidating clusters...")
    unique_labels = np.unique(cluster_labels)
    label_mapping = {old: new for new, old in enumerate(unique_labels)}
    cluster_labels = np.array([label_mapping[label] for label in cluster_labels])
    
    print(f"Found {len(unique_labels)} clusters from {n_samples} embeddings")
    return cluster_labels

def _dbscan_clustering(normalized_embeddings, max_distance=0.2):
    fit_kwargs = {}
    try:
        from cuml import DBSCAN
        dbscan_model = DBSCAN(
            eps=max_distance, 
            metric='euclidean', 
            algorithm='rbc',
            max_mbytes_per_batch=2048
        )
        fit_kwargs['out_dtype'] = 'int64'
    except ImportError:
        logging.warning("cuml not available, using sklearn DBSCAN instead.")
        from sklearn.cluster import DBSCAN
        # Use sklearn's DBSCAN if cuml is not available
        dbscan_model = DBSCAN(
            eps=max_distance, 
            metric='euclidean'
        )
    embed_clusters = dbscan_model.fit_predict(normalized_embeddings, **fit_kwargs)
    
    return embed_clusters

def get_similar_target_mapper(embeddings: np.ndarray, target_df: pl.DataFrame, max_distance=0.2):
    assert 'count' in target_df.columns, "target_df must contain 'count' column"
    assert 'Target' in target_df.columns, "target_df must contain 'Target' column"
    assert embeddings.shape[0] == target_df.shape[0], "embeddings must match the number of targets in target_df"

    embed_clusters = cluster_target_embeddings(embeddings, max_distance=max_distance)
    return _clusters_to_mapper(embed_clusters, target_df)

def _clusters_to_mapper(embed_clusters, target_df):
    target_df = target_df.with_columns(pl.Series(name='cluster', values=embed_clusters))
    # primary target should be ideally most common target, and secondly shortest
    primary_target_df = target_df.sort(['count', pl.col('Target').str.len_chars()], descending=[True, False])\
        .unique('cluster', keep='first')\
        .rename({'Target': 'top_target', 'count': 'top_count'})
    target_df = target_df.filter(pl.col('cluster') != -1).join(primary_target_df, on='cluster', how='inner').filter(pl.col('top_target') != pl.col('Target'))
    return {k: v for k, v in target_df.select(['Target', 'top_target']).rows()}

def _propagate_clusters_np(cluster_labels, verbose=True):

    labels_range = np.arange(len(cluster_labels))

    if np.array_equal(np.asarray(cluster_labels), labels_range):
        return cluster_labels

    for _ in tqdm(range(3), desc="Propagating clusters", disable=not verbose):  # Usually converges in 2-3 iterations
        needs_update = cluster_labels != labels_range
        if not needs_update.any():
            break

        old_labels = cluster_labels[needs_update]
        new_labels = cluster_labels[old_labels]
        cluster_labels[needs_update] = new_labels

        if np.array_equal(old_labels, new_labels):
            break
    return cluster_labels

def _minhash_clustering(target_df: pl.DataFrame, threshold=0.7):
    """GPU-accelerated deduplication using cuVS (RAPIDS)"""
    import datasketch

    # Create LSH index
    lsh = datasketch.MinHashLSH(threshold=threshold, num_perm=128)

    byte_sets = target_df.select(pl.col('Target').str.split(' ').list.eval(pl.element().cast(pl.Binary)))['Target'].to_list()

    hashes = datasketch.MinHash.bulk(byte_sets)
    hashes = [datasketch.LeanMinHash(h) for h in hashes]
    with lsh.insertion_session() as session:
        for i, h in tqdm(enumerate(hashes), desc='Inserting hashes into LSH', total=len(hashes)):
            session.insert(i, h)

    n_samples = len(hashes)

    cluster_labels = np.arange(n_samples)

    for i, h in tqdm(enumerate(hashes), desc='Querying hash index', total=n_samples):
        neighbour_indices = lsh.query(h)[1:] # remove first index because it will be the query hash itself

        # Merge clusters for points within threshold
        if len(neighbour_indices) > 0:
            # Get minimum cluster label among neighbors
            neighbor_labels = cluster_labels[neighbour_indices]
            min_label = np.min(np.concatenate([
                np.array([cluster_labels[i]]), 
                neighbor_labels
            ]))
            
            # Update all connected points
            cluster_labels[i] = min_label
            cluster_labels[neighbour_indices] = min_label

    # Propagate cluster assignments (handle transitive closure)
    cluster_labels = _propagate_clusters_np(cluster_labels, verbose=True)
    
    # Renumber clusters consecutively
    unique_labels = np.unique(cluster_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    final_labels = np.array([label_map[label] for label in cluster_labels])

    return final_labels

def _get_similar_target_mapper_batch(target_df: pl.DataFrame, embedding_model_name: str, minhash_threshold=0.7, max_embedding_distance=0.2, batch_size=1000):
    hash_clusters = _minhash_clustering(target_df, threshold=minhash_threshold)
    target_cluster_df = target_df.with_columns(pl.Series(name='cluster', values=hash_clusters))
    embedding_model = VLLMEmbedder(model=embedding_model_name)
    cluster_df = target_cluster_df.with_row_index()\
        .group_by('cluster')\
        .agg([pl.col('Target'), pl.col('index'), pl.col('Target').len().alias('cluster_size')])\
        .filter(pl.col('cluster_size') > 1)
    
    batch_df = cluster_df.sort('cluster_size')\
        .with_columns(pl.col('cluster_size').cum_sum().mod(batch_size).alias('modcumsum'))\
        .with_columns((pl.col('modcumsum').shift(1) > pl.col('modcumsum')).cast(pl.Int32).fill_null(0).cum_sum().alias('batches'))\
        .group_by('batches')\
        .agg([pl.col('Target').flatten(), pl.col('index').flatten()])

    next_cluster_idx = 0
    embed_clusters = np.full(target_df.shape[0], -1)
    for batch_i, batch in enumerate(batch_df.to_dicts()):
        logger.info(f"Finding embedding clusters in minhash clusters batch {batch_i + 1}/{len(batch_df)}")
        batch_targets = batch['Target']
        batch_embeddings = embedding_model.encode(batch_targets, show_progress_bar=True)
        sub_clusters = _cuvl_clustering(batch_embeddings, max_distance=max_embedding_distance, verbose=True)
        del batch_embeddings
        torch.cuda.empty_cache()
        gc.collect()

        # ensure sub clusters have distinct ids between macro clusters
        if len(sub_clusters[sub_clusters != -1]) > 0:
            sub_clusters[sub_clusters != -1] += next_cluster_idx
            next_cluster_idx = sub_clusters.max() + 1

        batch_idx = np.asarray(batch['index'])
        embed_clusters[batch_idx] = sub_clusters

    return _clusters_to_mapper(embed_clusters, target_df)

def deduplicate_all_similar_targets(document_df: pl.DataFrame, embedding_model_name: str, batch_size: int = 1000, minhash_threshold=0.7, max_embedding_distance: float = 0.1) -> pl.DataFrame:
    target_df = document_df.select('Targets')\
        .explode('Targets')\
        .drop_nulls()\
        .rename({'Targets': 'Target'})\
        .group_by('Target')\
        .agg(pl.len().alias('count'))
    
    target_mapper = _get_similar_target_mapper_batch(target_df, embedding_model_name, minhash_threshold=minhash_threshold, max_embedding_distance=max_embedding_distance, batch_size=batch_size)
    document_df = document_df.with_columns(
        pl.col('Targets').list.eval(pl.element().replace(target_mapper)).list.unique()
    )
    return document_df


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
        'stance target',
        'target1',
        'target2'
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

def _get_var_and_max_var_target(documents_df: pl.DataFrame, target_info_df: pl.DataFrame) -> pl.DataFrame:
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


def _filter_stance_targets(all_targets: pl.Series) -> pl.Series:
    # lower case all results
    all_targets = all_targets.list.eval(
        pl.element().str.to_lowercase().str.strip_chars().str.replace('stance target: ', '').str.replace('1. ', '').str.strip_chars().str.strip_chars('"').str.strip_chars("'")
    )
    # remove exact duplicates
    all_targets = all_targets.list.unique()
    return all_targets

def _filter_phrases(target_embeds, similarity_threshold=0.9):
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


def get_transcripts_from_video_files(
        video_paths: List[str], 
        hf_token: str, 
        whisper_model: str = "large-v2", 
        batch_size: int = 16, 
        save_speaker_embeddings: bool = False, 
        verbose: bool = True,
        skip_errors: bool = False
    ) -> pl.DataFrame:
    """Get transcripts from a list of video file paths using whisperx.

    Requires whisperx, moviepy, and pyannote.audio.

    Args:
        video_paths (List[str]): List of paths to video files.
        hf_token (str): Hugging Face token for accessing models.
        whisper_model (str): Whisper model to use (default: "large-v2").
        batch_size (int): Batch size for processing (default: 16).
        save_speaker_embeddings (bool): Whether to save speaker embeddings (default: False).
        verbose (bool): Whether to show progress bar (default: True).

    Returns:
        pl.DataFrame: DataFrame containing the transcripts and diarization results.
    """
    try:
        import whisperx
        from whisperx.audio import SAMPLE_RATE
    except ImportError:
        raise ImportError("whisperx is not installed. Please install it with `pip install whisperx`.")

    def load_audio_from_video_file(video_path):
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", video_path,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le", 
            "-ar", str(SAMPLE_RATE),
            "-"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg command failed with error: {e.stderr.decode()}") from e
        audio = np.frombuffer(result.stdout, np.int16).flatten().astype(np.float32) / 32768.0
        return audio

    return _get_transcripts_from_audio(
        video_paths, 
        load_audio_from_video_file, 
        hf_token=hf_token, 
        whisper_model=whisper_model, 
        batch_size=batch_size, 
        verbose=verbose, 
        save_speaker_embeddings=save_speaker_embeddings,
        skip_errors=skip_errors
    )

def get_transcripts_from_audio_files(
        audio_paths: List[str], 
        hf_token: str, 
        whisper_model: str = "large-v2", 
        batch_size: int = 16, 
        save_speaker_embeddings: bool = False, 
        verbose: bool = True,
        skip_errors: bool = False
    ) -> pl.DataFrame:
    """Get transcripts from a list of audio file paths using whisperx.

    Requires whisperx and pyannote.audio.

    Args:
        audio_paths (List[str]): List of paths to audio files.
        hf_token (str): Hugging Face token for accessing models.
        whisper_model (str): Whisper model to use (default: "large-v2").
        batch_size (int): Batch size for processing (default: 16).
        save_speaker_embeddings (bool): Whether to save speaker embeddings (default: False).
        verbose (bool): Whether to show progress bar (default: True).

    Returns:
        pl.DataFrame: DataFrame containing the transcripts and diarization results.
    """
    try:
        import whisperx
    except ImportError:
        raise ImportError("whisperx is not installed. Please install it with `pip install whisperx`.")
    
    def load_audio_from_file(audio_file):
        audio = whisperx.load_audio(audio_file)
        return audio

    return _get_transcripts_from_audio(
        audio_paths, 
        load_audio_from_file, 
        hf_token=hf_token, 
        whisper_model=whisper_model, 
        batch_size=batch_size, 
        verbose=verbose, 
        save_speaker_embeddings=save_speaker_embeddings,
        skip_errors=skip_errors
    )

def _get_transcripts_from_audio(
        items,
        audio_loader, 
        hf_token, 
        whisper_model="large-v2", 
        batch_size=16, 
        save_speaker_embeddings=False,
        verbose=True,
        skip_errors=False
    ) -> pl.DataFrame:
    import torch

    try:
        import whisperx
        from pyannote.audio import Pipeline
        from whisperx.audio import SAMPLE_RATE
    except ImportError:
        raise ImportError("whisperx and/or pyannote is not installed. Please install them with `pip install whisperx pyannote.audio`.")

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    model = whisperx.load_model(whisper_model, device_name, compute_type=compute_type)
    diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token).to(device)

    results = []
    for item in tqdm(items, disable=not verbose, desc="Transcribing"):
        try:
            audio = audio_loader(item)
            result = model.transcribe(audio, batch_size=batch_size)

            # delete model if low on GPU resources
            # import gc; gc.collect(); torch.cuda.empty_cache(); del model

            # 2. Align whisper output
            language = result['language']
            try:
                model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
            except ValueError:
                d = {
                    'path': item,
                    'result': result,
                }
                results.append(d)
                continue

            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

            # delete model if low on GPU resources
            # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

            # 3. Assign speaker labels
            # add min/max number of speakers if known
            audio_data = {
                'waveform': torch.from_numpy(audio[None, :]),
                'sample_rate': SAMPLE_RATE
            }
            segments, embeddings = diarize_model(audio_data, return_embeddings=True)
            diarize_segments = pl.DataFrame(segments.itertracks(yield_label=True), schema=['segment', 'label', 'speaker'])
            diarize_segments = diarize_segments.with_columns([
                pl.col('segment').map_elements(lambda s: s.start, pl.Float64).alias('start'),
                pl.col('segment').map_elements(lambda s: s.end, pl.Float64).alias('end')
            ]).to_pandas()
            
            result = whisperx.assign_word_speakers(diarize_segments, result)
            result['language'] = language

            d = {
                'path': item,
                'result': result,
                'diarize_segments': diarize_segments[['label', 'start', 'end', 'speaker']].to_dict(orient='records'),
            }

            if save_speaker_embeddings:
                d['embeddings'] = embeddings

            results.append(d)
        except Exception as e:
            if skip_errors:
                logger.error(f"Error processing {item}: {e}")
                continue
            else:
                raise

    return pl.DataFrame(results)


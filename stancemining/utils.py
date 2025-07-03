import logging
import tempfile
from typing import List

from nltk.corpus import stopwords
import numpy as np
import polars as pl
import sklearn.preprocessing
from tqdm import tqdm

def deduplicate_target_embeddings(embeddings):
    normalized_embeddings = sklearn.preprocessing.normalize(embeddings, axis=1, norm='l2')
    max_distance = 0.2
    try:
        from cuml import DBSCAN
        dbscan_model = DBSCAN(eps=max_distance, metric='euclidean', algorithm='rbc', min_samples=2)
    except ImportError:
        logging.warning("cuml not available, using sklearn DBSCAN instead.")
        from sklearn.cluster import DBSCAN
        # Use sklearn's DBSCAN if cuml is not available
        dbscan_model = DBSCAN(eps=max_distance, metric='euclidean', min_samples=2)
    embed_clusters = dbscan_model.fit_predict(normalized_embeddings)
    return embed_clusters

def get_similar_target_mapper(embeddings: np.ndarray, target_df: pl.DataFrame):
    assert 'count' in target_df.columns, "target_df must contain 'count' column"
    assert 'Target' in target_df.columns, "target_df must contain 'Target' column"
    assert embeddings.shape[0] == target_df.shape[0], "embeddings must match the number of targets in target_df"

    embed_clusters = deduplicate_target_embeddings(embeddings)
    target_df = target_df.with_columns(pl.Series(name='cluster', values=embed_clusters))
    primary_target_df = target_df.sort('count', descending=True).unique('cluster', keep='first').rename({'Target': 'top_target', 'count': 'top_count'})
    target_df = target_df.filter(pl.col('cluster') != -1).join(primary_target_df, on='cluster', how='inner').filter(pl.col('top_target') != pl.col('Target'))
    return {k: v for k, v in target_df.select(['Target', 'top_target']).rows()}

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

def filter_phrases(target_embeds, similarity_threshold=0.9):
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

class Transcription:
    def __init__(self, hf_token):
        device = "cuda" 
        compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
        self.model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        self.diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token).to(torch.device(device))

        self.align_models = {}

    def process_video(self, video_bytes, batch_size=16):
        # Create a temporary file to write the mp4 bytes
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_video_file:
            with tempfile.NamedTemporaryFile(suffix='.mp3') as temp_audio_file:
                temp_video_path = temp_video_file.name
                temp_video_file.write(video_bytes)

                temp_audio_path = temp_audio_file.name

                # Use moviepy to extract audio
                with VideoFileClip(temp_video_path) as video:
                    video.audio.write_audiofile(temp_audio_path)
                
                audio = whisperx.load_audio(temp_audio_path)
        
        return self.process_audio(audio, batch_size=batch_size)

    def process_audio(self, audio, batch_size=16):
        device = "cuda" 

        # 1. Transcribe with original whisper (batched)
        # save model to local path (optional)
        # model_dir = "/path/"
        # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

        
        result = self.model.transcribe(audio, batch_size=batch_size)

        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del model

        # 2. Align whisper output
        language = result['language']
        if language not in self.align_models:
            model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
            self.align_models[language] = (model_a, metadata)
        else:
            model_a, metadata = self.align_models[language]
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

        # 3. Assign speaker labels
        # add min/max number of speakers if known
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        segments, embeddings = self.diarize_model(audio_data, return_embeddings=True)
        diarize_segments = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_segments['start'] = diarize_segments['segment'].apply(lambda x: x.start)
        diarize_segments['end'] = diarize_segments['segment'].apply(lambda x: x.end)
        
        result = whisperx.assign_word_speakers(diarize_segments, result)
        result['language'] = language
        return result, diarize_segments, embeddings


def get_transcripts_from_video_files(
        video_paths: List[str], 
        hf_token: str, 
        whisper_model: str = "large-v2", 
        batch_size: int = 16, 
        save_speaker_embeddings: bool = False, 
        verbose: bool = True
    ) -> pl.DataFrame:
    """
    Get transcripts from a list of video file paths using whisperx.

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
        from moviepy.editor import VideoFileClip
    except ImportError:
        raise ImportError("whisperx and/or moviepy is not installed. Please install them with `pip install whisperx moviepy`.")

    def load_audio_from_video_file(video_path):
        with tempfile.NamedTemporaryFile(suffix='.mp3') as temp_audio_file:
                temp_audio_path = temp_audio_file.name

                # Use moviepy to extract audio
                with VideoFileClip(video_path) as video:
                    video.audio.write_audiofile(temp_audio_path)
                
                audio = whisperx.load_audio(temp_audio_path)
                return audio

    return _get_transcripts_from_audio(video_paths, load_audio_from_video_file, hf_token=hf_token, whisper_model=whisper_model, batch_size=batch_size, verbose=verbose, save_speaker_embeddings=save_speaker_embeddings)

def get_transcripts_from_audio_files(
        audio_paths: List[str], 
        hf_token: str, 
        whisper_model: str = "large-v2", 
        batch_size: int = 16, 
        save_speaker_embeddings: bool = False, 
        verbose: bool = True
    ) -> pl.DataFrame:
    """
    Get transcripts from a list of audio file paths using whisperx.

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

    return _get_transcripts_from_audio(audio_paths, load_audio_from_file, hf_token=hf_token, whisper_model=whisper_model, batch_size=batch_size, verbose=verbose, save_speaker_embeddings=save_speaker_embeddings)

def _get_transcripts_from_audio(
        items,
        audio_loader, 
        hf_token, 
        whisper_model="large-v2", 
        batch_size=16, 
        save_speaker_embeddings=False,
        verbose=True
    ) -> pl.DataFrame:
    import torch

    try:
        import whisperx
        from pyannote.audio import Pipeline
        from whisperx.audio import SAMPLE_RATE
    except ImportError:
        raise ImportError("whisperx and/or pyannote is not installed. Please install them with `pip install whisperx pyannote.audio`.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    model = whisperx.load_model(whisper_model, device, compute_type=compute_type)
    diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token).to(device)

    results = []
    for item in tqdm(items, disable=not verbose, desc="Transcribing"):
        audio = audio_loader(item)
        result = model.transcribe(audio, batch_size=batch_size)

        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del model

        # 2. Align whisper output
        language = result['language']
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
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
        diarize_segments = pl.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_segments = diarize_segments.with_columns([
            pl.col('segment').struct.field('start').alias('start'),
            pl.col('segment').struct.field('end').alias('end')
        ])
        
        result = whisperx.assign_word_speakers(diarize_segments, result)
        result['language'] = language

        d = {
            'path': item,
            'result': result,
            'diarize_segments': diarize_segments,
        }

        if save_speaker_embeddings:
            d['embeddings'] = embeddings

        results.append(d)

    return pl.DataFrame(results)


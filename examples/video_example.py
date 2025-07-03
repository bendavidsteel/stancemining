import polars as pl

import stancemining
import stancemining.plot
import stancemining.utils

def main():
    video_paths = [
        './tests/data/video1.mp4',
        './tests/data/video2.mp4',
        './tests/data/video3.mp4'
    ]

    hf_token = 'hf_your_huggingface_token_here'  # Replace with your Hugging Face token

    transcripts_df = stancemining.utils.get_transcripts_from_video_files(video_paths, hf_token)

    docs = transcripts_df['text'].to_list()

    model = stancemining.StanceMining(model_name='Qwen/Qwen3-1.7B', verbose=True)
    document_df = model.fit_transform(docs, text_column='text')
    fig = stancemining.plot.plot_semantic_map(document_df)
    fig.savefig('./semantic_map.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
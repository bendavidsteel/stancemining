import time

import polars as pl

import stancemining
import stancemining.plot

def main():
    document_df = pl.from_dicts([
        {
            'text': "I love the new features of the product, especially the user-friendly interface.",
            'Targets': ["new features", "user-friendly interface"],
        },
        {
            'text': "The recent update has made the app slower and more buggy. Not happy with it.",
            'Targets': ["recent update", "app performance"],
        }
    ])

    model = stancemining.StanceMining(
        stance_detection_finetune_kwargs={'hf_model': 'bendavidsteel/Qwen3-1.7B-claim-entailment-5-labels'},
        stance_target_type='claims',
        verbose=True
    )
    stance_df = model.get_stance(document_df)
    
    for doc_data in stance_df.to_dicts():
        print(f"Document: {doc_data['text']}")
        for i in range(len(doc_data['Targets'])):
            target = doc_data['Targets'][i]
            stance = doc_data['Stances'][i]
            print(f"  Target: {target} | Stance: {stance}")

if __name__ == '__main__':
    main()
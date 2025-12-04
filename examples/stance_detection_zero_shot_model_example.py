import polars as pl

import stancemining

def main():
    document_df = pl.from_dicts([
        {
            'text': "Immigration policies under Trudeau's Liberals has been a disaster for Canada.",
            'Targets': ["Justin Trudeau is a good prime-minister", "The Liberal party immigration policy is bad"],
        },
        {
            'text': "I'm all for climate action, but the carbon tax implemented by the government is just too high and is hurting everyday Canadians.",
            'Targets': ["The carbon tax is too high", "Climate action is necessary"],
        }
    ])

    # Using Qwen3-30B-A3B-Thinking-2507-FP8 - strongest model that fits on 24GB GPU
    # MoE architecture: 30.5B total params, only 3.3B active during inference
    # FP8 quantization allows it to fit in 24GB VRAM with excellent performance
    model = stancemining.StanceMining(
        model_name='nvidia/Qwen3-30B-A3B-NVFP4',
        stance_target_type='claims',
        llm_method='prompting',
        model_kwargs={
            'gpu_memory_utilization': 0.90,
            'max_model_len': 8192,
            'enable_prefix_caching': True,
        },
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
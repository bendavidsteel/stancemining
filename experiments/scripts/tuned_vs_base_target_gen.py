import datetime
import json
import os

import bertopic.representation
import hydra
import numpy as np
import omegaconf
import polars as pl
import wandb

from experiments import metrics, datasets
from stancemining import prompting, finetune

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    dataset_name = 'vast'
    model_name = config['model']['llmmodelname']
    method = config['model']['method']
    docs_df = datasets.load_dataset(dataset_name)
    docs = docs_df['Text'].to_list()

    import stancemining as vp
    vector = vp.Vector('favor', 'against')
    model = vp.StanceMining(
        vector, 
        method=method, 
        llm_method=config.model.llm_method,
        model_lib='transformers', 
        model_name=model_name,
        model_kwargs={'device_map': 'auto'},
        finetune_kwargs=config.wiba
    )

    num_samples = 3
    zero_shot_targets = prompting.ask_llm_zero_shot_stance_target(model.generator, docs, {'num_samples': num_samples})
    
    model_name = model.finetune_kwargs['model_name'].replace('/', '-')
    topic_extraction_path = f'./models/wiba/{model_name}-topic-extraction-vast-ezstance'
    df = pl.DataFrame({'Text': docs})
    results = finetune.get_predictions("topic-extraction", df, topic_extraction_path, model.finetune_kwargs)
    if isinstance(results[0], str):
        results = [[r] for r in results]

    pass

if __name__ == "__main__":
    main()



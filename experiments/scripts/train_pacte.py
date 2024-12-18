import os

import hydra
import polars as pl

from experiments import datasets
from experiments.methods import pacte

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    dataset_name = config['data']['dataset']
    docs_df = datasets.load_dataset(dataset_name, split='train', group=False)
    docs_df = docs_df.filter(docs_df['Stance'] != 'neutral')
    docs = docs_df['Text'].to_list()
    labels = docs_df['Stance'].to_list()
    
    pacte_model = pacte.PaCTE()
    pacte_model.train(docs, labels=labels)

if __name__ == '__main__':
    main()
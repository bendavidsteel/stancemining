import os

import polars as pl

from experiments.methods import pacte

def main():
    docs_df = pl.read_csv(f'./data/semeval_train.csv')
    docs_df = docs_df.filter(docs_df['Stance'] != 'NONE')
    docs = docs_df['Tweet'].to_list()
    labels = docs_df['Stance'].to_list()
    
    pacte_model = pacte.PaCTE()
    pacte_model.train(docs, labels=labels)

if __name__ == '__main__':
    main()
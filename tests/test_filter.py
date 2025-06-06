import itertools
import random

import numpy as np
import polars as pl
import pytest
from scipy.stats import dirichlet as scipy_dirichlet

from stancemining import StanceMining
from stancemining import metrics

from test_all import MockStanceMining

def test_filter_targets():
    num_docs = 10
    targets = [[f'target_{j}' for j in range(3)] for i in range(num_docs)]
    df = pl.DataFrame({'Targets': targets})
    miner = MockStanceMining()
    df = df.with_columns(miner._filter_similar_phrases_fast(df['Targets']))
    assert len(df) == len(targets)


if __name__ == '__main__':
    test_filter_targets()
    print("filter_targets passed")
import argparse
import os

import dotenv
import hydra
import torch
import tqdm

from train_models import _main

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    _main(config, config.wiba)

if __name__ == "__main__":
    main()
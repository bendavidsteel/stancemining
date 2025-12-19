# stancemining

A tool for mining stance targets from a corpus of documents.

## Installation

```bash
pip install stancemining
```

There are some optional additional dependencies for extra functionality.
If you want to use the implemented Gaussian process for time series modelling, use `stancemining[gp]`.
If you want to train new task-specific models, use `stancemining[train]`.
If you want to use the plotting functions, use `stancemining[plot]`.

## Documentation
Documentation is available at [stancemining.readthedocs.io](https://stancemining.readthedocs.io)

## Usage

### To get stance targets and stance from a corpus of documents
```
import stancemining

model = stancemining.StanceMining()

document_df = model.fit_transform(docs)
target_info_df = model.get_target_info()
```

### To get stance target, stance, and stance trends from a corpus of documents
```
import stancemining

model = stancemining.StanceMining()
document_df = model.fit_transform(docs)
trend_df = stancemining.get_trends_for_all_targets(document_df)
```

## StanceMining App

This library comes with a web app to explore the results of the output.

Here is a video demo of the app: [StanceMining App Demo](https://www.youtube.com/watch?v=4tvqq8GTUHU)

### To deploy stancemining app

The <your-data-path> should be either an absolute path, or a path relative to the `app` directory where the compose.yaml file is located.
It should contain a `doc_stance` directory with the output of the fit_transform method saved as `.parquet.zstd` files, and a `target_trends` directory with the outputs from the `infer_stance_trends_for_all_targets` or `infer_stance_trends_for_target` method saved as `.parquet.zstd` files. There can be multiple files in each directory, and the app will automatically load all of them and concatenate them.

If you need authentication for the app, you can set the environment variable `STANCE_AUTH_URL_PATH` to the URL of your authentication service (e.g., `myauth.com/login`). That path must accept a POST request with a JSON body containing `username` and `password` fields, and return a JSON response with a `token` field.
If you do not need authentication, you can leave the environment variable unset.
```
export STANCE_DATA_PATH=<your-data-path>
export STANCE_AUTH_URL_PATH=<your-auth-url/login>
docker compose -f ./app/compose.yaml up
```

## To train new finetuned LLM models

Rename `./config/config_default.yaml` to `./config/config.yaml` and set the parameters in the file.
Run:
```
python ./experiments/scripts/train_models.py
```

## If you use this library, please cite:
```
@inproceedings{steel2025corpus,
  title={Corpus-Oriented Stance Target Extraction},
  author={Steel, Benjamin and Ruths, Derek},
  booktitle={Proceedings of the 6th Workshop on Computational Approaches to Discourse, Context and Document-Level Inferences (CODI 2025)},
  pages={209--227},
  year={2025}
}
```

## To reproduce experimental results from Corpus-Oriented Stance Target Extraction paper
Rename `./config/config_default.yaml` to `./config/config.yaml` and set the parameters in the file.
Run:
```
python ./experiments/scripts/main.py
```

You may need to setup a WanDB project.
Run 
```
python ./experiments/scripts/get_results.py
```
To have the metrics written to a latex table saved as a tex file.





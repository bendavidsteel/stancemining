# Stance Target Mining

A tool for mining stance targets from a corpus of documents.

## Installation

```bash
pip install stancemining
```

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
If you need authentication for the app, you can set the environment variable `STANCE_AUTH_URL_PATH` to the URL of your authentication service (e.g., `myauth.com/login`). That path must accept a POST request with a JSON body containing `username` and `password` fields, and return a JSON response with a `token` field.
If you do not need authentication, you can leave the environment variable unset.
```
export STANCE_DATA_PATH=<your-data-path>
export STANCE_AUTH_URL_PATH=<your-auth-url/login>
docker compose -f ./app/compose.yaml up
```

## To reproduce experimental results
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





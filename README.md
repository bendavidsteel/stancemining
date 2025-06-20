# Stance Target Mining

A tool for mining stance targets from a corpus of documents.

# Installation

```bash
pip install git+https://github.com/bendavidsteel/stancemining.git
```

# To get stance targets and stance from a corpus of documents
```
import stancemining

model = stancemining.StanceMining()

document_df = model.fit_transform(docs)
target_info_df = model.get_target_info()
```

# To get stance target, stance, and stance trends from a corpus of documents
```
import stancemining

model = stancemining.StanceMining()
document_df = model.fit_transform(docs)
target_info_df = model.get_target_info()
trend_df = stancemining.get_stance_trends(document_df, target_info_df)
```

# To deploy stancemining app
```
docker compose up
```

# To reproduce experimental results
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





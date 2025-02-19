# Stance Target Mining

A tool for mining stance targets from a corpus of documents.

# To run
```
import stancemining

model = stancemining.StanceMining()

doc_targets, probs, polarity = model.fit_transform(docs)
target_info = model.get_target_info()
```

# To reproduce results
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



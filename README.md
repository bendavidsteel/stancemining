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
```
python ./experiments/scripts/main.py
```



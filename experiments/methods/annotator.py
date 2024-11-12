import os

import polars as pl

class Annotator:
    def __init__(self, annotator_path):
        if not os.path.exists(annotator_path):
            raise FileNotFoundError(f"Annotator file not found: {annotator_path}")

        self.df = pl.read_csv(annotator_path)

    def fit_transform(self, docs):
        annotated_docs = self.df['Tweet'].to_list()
        assert len(docs) == len(annotated_docs)
        assert all([doc == annotated_doc for doc, annotated_doc in zip(docs, annotated_docs)])

        final_responses = []
        for d in self.df.to_dicts():
            if d['2nd clustering'] != None:
                final_responses.append(d['2nd clustering'])
            elif d['1st clustering'] != None:
                final_responses.append(d['1st clustering'])
            else:
                final_responses.append(d['initial_targets'])

        doc_targets = []
        polarity = []
        for response in final_responses:
            target, stance, specificity = response.split(', ')
            doc_targets.append(target)
            polarity.append(stance)

        doc_targets = self.df['target'].to_list()
        probs = self.df['probs'].to_list()
        polarity = self.df['polarity'].to_list()
        self.all_targets = list(set(doc_targets))
        return doc_targets, probs, polarity

    def get_target_info(self):
        return pl.DataFrame(self.all_targets, columns=['target'])
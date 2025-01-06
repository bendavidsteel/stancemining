from typing import List

from bertopic import BERTopic
import numpy as np
import pandas as pd
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

from vectopic import llms, finetune, prompting
from vectopic.ngram_gen import NGramGeneration

keyword_prompt = """<|system|>You are a helpful, respectful and honest assistant for labeling topics..</s>
<|user|>
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.</s>
<|assistant|>"""

def get_var_and_max_var_target(documents_df, target_info):
    target_info_df = pl.DataFrame(target_info)
    target_info_df = target_info_df.group_by('noun_phrase')\
        .agg(pl.col('topic_id'), pl.col('polarity'))
    target_info_df = target_info_df.with_columns([
        pl.col('polarity').list.mean().alias('mean'),
        pl.when(pl.col('polarity').list.len() > 1)\
            .then(pl.col('polarity').list.var())\
            .otherwise(0)
            .alias('var')
    ])

    documents_df = documents_df.join(
        documents_df.explode('Targets')\
            .join(target_info_df, left_on='Targets', right_on='noun_phrase', how='left')\
            .group_by('ID')\
            .agg(pl.all().sort_by('var').last())\
            .with_columns(pl.col('Targets').alias('Target'))\
            .select(['ID', 'Target']),
        on='ID',
        how='left'
    )
    return documents_df, target_info_df


class Vector:
    def __init__(self, sent_a, sent_b):
        self.sent_a = sent_a
        self.sent_b = sent_b
        self.neutral = 'neutral'

class VectorTopic:
    def __init__(self, 
                 vector, 
                 method='llm', 
                 llm_method='zero-shot',
                 num_representative_docs=5,
                 model_lib='transformers', 
                 model_name='microsoft/Phi-3.5-mini-instruct', 
                 model_kwargs={}, 
                 tokenizer_kwargs={},
                 finetune_kwargs={}
                ):
        self.vector = vector
        self.method = method
        assert llm_method in ['zero-shot', 'finetuned'], f"LLM method must be either 'zero-shot' or 'finetuned', not '{llm_method}'"
        self.llm_method = llm_method

        self.num_representative_docs = num_representative_docs

        self.model_lib = model_lib
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.finetune_kwargs = finetune_kwargs

        # lazily load generator if using in between fine-tuned models
        self.generator = self._get_generator(lazy=llm_method=='finetuned')

        self.embedding_model = 'paraphrase-MiniLM-L6-v2'

    def _get_embedding_model(self):
        model = SentenceTransformer(self.embedding_model)
        return model

    def _get_embeddings(self, docs, model=None):
        if model is None:
            model = self._get_embedding_model()
        embeddings = model.encode(docs)
        return embeddings

    
    def _ask_llm_stance_target(self, docs: List[str]):
        if self.llm_method == 'zero-shot':
            num_samples = 3
            return prompting.ask_llm_zero_shot_stance_target(self.generator, docs, {'num_samples': num_samples})
        elif self.llm_method == 'finetuned':
            model_name = self.finetune_kwargs['model_name'].replace('/', '-')
            topic_extraction_path = f'./models/wiba/{model_name}-topic-extraction-vast-ezstance'
            df = pl.DataFrame({'Text': docs})
            results = finetune.get_predictions("topic-extraction", df, topic_extraction_path, self.finetune_kwargs)
            if isinstance(results[0], str):
                results = [[r] for r in results]
            return results

    def _ask_llm_stance(self, docs, stance_targets):
        if self.llm_method == 'zero-shot':
            return prompting.ask_llm_zero_shot_stance(self.generator, docs, stance_targets)
        elif self.llm_method == 'finetuned':
            model_name = self.finetune_kwargs['model_name'].replace('/', '-')
            stance_classification_path = f'./models/wiba/{model_name}-stance-classification-vast-ezstance'
            data = pl.DataFrame({'Text': docs, 'Target': stance_targets})
            results = finetune.get_predictions("stance-classification", data, stance_classification_path, self.finetune_kwargs)
            results = [r.upper() for r in results]
            return results


    def _ask_llm_score(self, noun_phrase, docs):
        text_name = "comment"
        prompt = f"""Targeted towards {noun_phrase}, is the following {text_name} more likely to be {self.vector.sent_a}, {self.vector.neutral}, or {self.vector.sent_b}? {{doc}}. Output either {self.vector.sent_a}, {self.vector.neutral}, or {self.vector.sent_b}."""
        return self.generator.get_prompt_response_probs(prompt, docs, self.vector.sent_a, self.vector.sent_b, self.vector.neutral)

    def _ask_llm_class(self, noun_phrase, docs):
        text_name = "comment"
        prompt = f"""Targeted towards {noun_phrase}, is the following {text_name} more likely to be {self.vector.sent_a}, {self.vector.neutral}, or {self.vector.sent_b}? {{doc}}. Output either {self.vector.sent_a}, {self.vector.neutral}, or {self.vector.sent_b}."""
        classifications = []
        for doc in docs:
            prompt = prompt.format(doc=doc)
            output = self.generator.generate(prompt, max_new_tokens=10, num_samples=1)[0]
            if self.vector.sent_a in output.lower():
                classification = self.vector.sent_a
            elif self.vector.sent_b in output.lower():
                classification = self.vector.sent_b
            else:
                classification = self.vector.neutral
            classifications.append(classification)
        return classifications

    def _remove_similar_phrases(self, noun_phrases, embedding_model=None):
        if len(noun_phrases) < 2:
            return noun_phrases
        # use embedding model to remove similar phrases
        embeddings = self._get_embeddings(noun_phrases, model=embedding_model)
        # get cosine similarity
        similarity = np.dot(embeddings, embeddings.T) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embeddings, axis=1)[:, None])
        similarity = np.triu(similarity, k=1)
        # get indices of similar phrases
        indices = np.where(similarity > 0.8)
        # remove similar phrases
        noun_phrases = [noun_phrases[i] for i in range(len(noun_phrases)) if i not in indices[0]]
        return noun_phrases

    def _get_base_topic_model(self, bertopic_kwargs):
        return BERTopic(**bertopic_kwargs)

    def _topic_llm_fit_transform(self, docs, embeddings=None, bertopic_kwargs={}):
        embedding_model = self._get_embedding_model()
        if embeddings is None:
            embeddings = self._get_embeddings(docs, model=embedding_model)

        topic_model = self._get_base_topic_model(bertopic_kwargs)
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
        documents = pd.DataFrame({"Document": docs, "ID": range(len(docs)), "Topic": topics})
        repr_docs, _, _, _ = topic_model._extract_representative_docs(
            topic_model.c_tf_idf_,
            documents,
            topic_model.topic_representations_,
            nr_samples=500,
            nr_repr_docs=self.num_representative_docs,
        )
        topic_model.representative_docs_ = repr_docs
        topic_info = topic_model.get_topic_info()
        target_info = []

        document_df = pl.from_pandas(documents)
        topic_info_df = pl.from_pandas(topic_info)

        # get stance targets for outliers
        outliers_df = document_df.filter(pl.col('Topic') == -1)
        stance_targets = self._ask_llm_stance_target(outliers_df['Document'])
        stance_targets = [self._remove_similar_phrases(ts, embedding_model=embedding_model) for ts in stance_targets]
        outliers_df = outliers_df.with_columns(pl.Series(name='Targets', values=stance_targets))

        # get stance targets for non-outliers
        stance_targets = [prompting.ask_llm_multi_doc_targets(self.generator, r['Representative_Docs']) for r in tqdm(topic_info_df[['Representative_Docs', 'Representation']].to_dicts(), desc="Getting topic noun phrases")]
        stance_targets = [self._remove_similar_phrases(ts, embedding_model=embedding_model) for ts in stance_targets]
        topic_info_df = topic_info_df.with_columns(pl.Series(name='Targets', values=stance_targets))

        non_outliers_df = document_df.filter(pl.col('Topic') != -1)
        non_outliers_df = non_outliers_df.join(topic_info_df, on='Topic', how='left')
        document_df = pl.concat([outliers_df, non_outliers_df], how='diagonal_relaxed')

        # get stances for all targets
        targets_df = document_df.explode('Targets')
        targets_df = targets_df.with_columns(
            pl.Series(name='Stance', values=self._ask_llm_stance(targets_df['Document'], targets_df['Targets']))
        )
        targets_df = targets_df.with_columns(pl.col('Stance').replace_strict({'FAVOR': 1, 'AGAINST': -1, 'NEUTRAL': 0}).alias('Polarity'))
        target_info.extend(targets_df[['Targets', 'Polarity']].rename({'Targets': 'noun_phrase', 'Polarity': 'polarity'}).to_dicts())

        # join targets back to documents
        document_df = document_df.join(
            targets_df.group_by('ID')\
                .agg(pl.col('Targets'), pl.col('Polarity'))\
                .rename({'Targets': 'Targets', 'Polarity': 'Polarities'}),
            on='ID',
            how='left'
        )

        documents_df, target_info_df = get_var_and_max_var_target(documents_df, target_info)
        self.target_info = target_info_df

        return documents
    
    def _get_base_targets(self, docs, embedding_model):
        target_info = []
        documents_df = pl.DataFrame({"Document": docs, "ID": range(len(docs))})

        stance_targets = self._ask_llm_stance_target(documents_df['Document'])
        stance_targets = [self._remove_similar_phrases(ts, embedding_model=embedding_model) for ts in stance_targets]
        documents_df = documents_df.with_columns(pl.Series(name='noun_phrases', values=stance_targets, dtype=pl.List(pl.String)))
        
        noun_phrase_df = documents_df.explode('noun_phrases').rename({'noun_phrases': 'noun_phrase'}).drop_nulls()
        noun_phrase_df = noun_phrase_df.with_columns(pl.Series(name='stance', values=self._ask_llm_stance(noun_phrase_df['Document'], noun_phrase_df['noun_phrase'])))
        noun_phrase_df = noun_phrase_df.with_columns(pl.col('stance').replace_strict({'FAVOR': 1, 'AGAINST': -1, 'NEUTRAL': 0}).alias('polarity'))
        target_info.extend(noun_phrase_df[['noun_phrase', 'polarity']].to_dicts())
        documents_df = documents_df.drop('noun_phrases').join(
            noun_phrase_df.group_by('ID')\
                .agg(pl.col('noun_phrase'), pl.col('polarity'))\
                .rename({'noun_phrase': 'Targets', 'polarity': 'Polarities'}),
            on='ID',
            how='left'
        ).with_columns([
            pl.col('Targets').fill_null([]),
            pl.col('Polarities').fill_null([])
        ])
        return documents_df, target_info

    def _llm_topic_fit_transform(self, docs, bertopic_kwargs={}):
        embedding_model = self._get_embedding_model()
        documents_df, target_info = self._get_base_targets(docs, embedding_model)

        # cluster initial stance targets
        topic_model = self._get_base_topic_model(bertopic_kwargs)
        base_targets_df = documents_df.explode('Targets').unique('Targets').drop_nulls()[['Targets']]
        topics, probs = topic_model.fit_transform(base_targets_df['Targets'].to_list())
        base_targets_df = base_targets_df.with_columns([
            pl.Series(name='Topic', values=topics),
            pl.col('Targets').alias('Document')
        ]).with_row_index(name='ID')
        repr_docs, _, _, _ = topic_model._extract_representative_docs(
            topic_model.c_tf_idf_,
            base_targets_df.to_pandas(),
            topic_model.topic_representations_,
            nr_samples=500,
            nr_repr_docs=self.num_representative_docs,
        )
        topic_model.representative_docs_ = repr_docs
        topic_info = topic_model.get_topic_info()

        # get higher level stance targets for each topic
        topic_info = pl.from_pandas(topic_info)
        no_outliers_df = topic_info.filter(pl.col('Topic') != -1)
        stance_targets = [prompting.ask_llm_target_aggregate(self.generator, r['Representative_Docs'], r['Representation']) for r in tqdm(no_outliers_df[['Representative_Docs', 'Representation']].to_dicts(), desc="Getting topic noun phrases")]
        stance_targets = [self._remove_similar_phrases(ts, embedding_model=embedding_model) for ts in stance_targets]
        no_outliers_df = no_outliers_df.with_columns(
            pl.Series(name='noun_phrases', values=stance_targets, dtype=pl.List(pl.String))
        )
        topic_info = pl.concat([topic_info.filter(pl.col('Topic') == -1), no_outliers_df], how='diagonal_relaxed')\
            .with_columns(pl.col('noun_phrases').fill_null([]))

        # map documents to new noun phrases, then get stance on new noun phrases
        targets_df = documents_df.explode('Targets')
        targets_df = targets_df.join(base_targets_df, on='Targets', how='left')
        targets_df = targets_df.filter(pl.col('Topic') != -1)
        targets_df = targets_df.join(topic_info, on='Topic', how='left')
        noun_phrases_df = targets_df.explode('noun_phrases').rename({'noun_phrases': 'noun_phrase'})
        noun_phrases_df = noun_phrases_df.with_columns(pl.Series(name='stance', values=self._ask_llm_stance(noun_phrases_df['Document'], noun_phrases_df['noun_phrase'])))
        noun_phrases_df = noun_phrases_df.with_columns(pl.col('stance').replace_strict({'FAVOR': 1, 'AGAINST': -1, 'NEUTRAL': 0}).alias('polarity'))
        target_info.extend(noun_phrases_df.rename({'Topic': 'topic_id'}).select(['noun_phrase', 'polarity', 'topic_id']).to_dicts())
        documents_df = documents_df.join(
                noun_phrases_df.group_by('ID')\
                    .agg(pl.col('noun_phrase'), pl.col('polarity'))\
                    .rename({'noun_phrase': 'NewTargets', 'polarity': 'NewPolarities'}), 
            on='ID', 
            how='left'
        )
        documents_df = documents_df.with_columns(
            pl.when(pl.col('NewTargets').is_not_null())\
                .then(pl.concat_list(pl.col('Targets'), pl.col('NewTargets')))\
                .otherwise(pl.col('Targets')))
        documents_df = documents_df.with_columns(
            pl.when(pl.col('NewPolarities').is_not_null())\
                .then(pl.concat_list(pl.col('Polarities'), pl.col('NewPolarities')))\
                .otherwise(pl.col('Polarities')))
        documents_df = documents_df.drop(['NewTargets', 'NewPolarities'])

        documents_df, target_info_df = get_var_and_max_var_target(documents_df, target_info)
        self.target_info = target_info_df

        return documents_df
    
    def _embedding_fit_transform():
        raise NotImplementedError("Method 'steeringvectors' not implemented")
        if embeddings is None:
            embeddings = get_embeddings(docs)

        model, tokenizer = get_generator()

        # TODO potentially use other topic modelling allowing doc to have multiple topics
        # or use probs to allow docs to fit into multiple topics
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
        self.topic_info = topic_model.get_topic_info()

        for idx, topic in topic_info.iterrows():
            # get generic vector, check projections of docs along that vector
            # then for multiple points in the space, take the mean difference vector, and use that as a steering vector
            # to produce generations of suggestions for noun_phrases
            noun_phrases = ask_llm(topic['representative_docs'])
            topic_id = topic['id']
            topic_docs = [docs[i] for i in range(len(docs)) if topics[i] == topic_id]
            noun_phrase_probs = []
            means = []
            vars = []
            for noun_phrase in noun_phrases:
                # calculate probability of doc along vector
                # https://arjunbansal.substack.com/p/llms-know-more-than-what-they-say
                # get noun_phrase specific target vector, and check projection of doc activations along that vector
                probs = ask_llm(noun_phrase, topic_docs)
                noun_phrase_probs.append(probs)
                means.append(np.mean(probs))
                vars.append(np.var(probs))

            self.topic_info.at[idx, 'noun_phrase_probs'] = noun_phrase_probs
            self.topic_info.at[idx, 'noun_phrase_means'] = means
            self.topic_info.at[idx, 'noun_phrase_vars'] = vars

        return documents
    
    def _get_targets_probs_polarity(self, documents):
        all_targets = self.target_info['noun_phrase'].to_list()
        target_to_idx = {target: idx for idx, target in enumerate(all_targets)}
        doc_targets = documents['Target'].to_list()
        doc_targets = [[t] if t is not None else [] for t in doc_targets]

        num_targets = len(self.target_info)
        
        polarities = np.zeros((len(documents), num_targets))
        probs = np.zeros((len(documents), num_targets))
        polarities_dict = documents['Polarities'].to_list()
        for i, doc_data in enumerate(documents.to_dicts()):
            doc_num_targets = len(doc_data['Targets'])
            for doc_target, doc_polarity in zip(doc_data['Targets'], doc_data['Polarities']):
                idx = target_to_idx[doc_target]
                polarities[i, idx] = doc_polarity
                probs[i, idx] = 1/doc_num_targets

        return doc_targets, probs, polarities

    def fit_transform(self, docs, embeddings=None, bertopic_kwargs={}):
        # could also compute generalized stance (i.e. sentiment), then find vector
        # then use steering vector to find the difference along the vector
        # https://www.lesswrong.com/posts/ndyngghzFY388Dnew/implementing-activation-steering

        # the same idea more fleshed out for evaluation 
        # https://arjunbansal.substack.com/p/llms-know-more-than-what-they-say
        if self.method == 'topicllm':
            documents = self._topic_llm_fit_transform(docs, embeddings, bertopic_kwargs)
        elif self.method == 'llmtopic':
            documents = self._llm_topic_fit_transform(docs, bertopic_kwargs=bertopic_kwargs)
        elif self.method == 'steeringvectors':
            documents = self._embedding_fit_transform(docs, embeddings)
        else:
            raise ValueError(f"Method '{self.method}' not implemented")
        
        return self._get_targets_probs_polarity(documents)

    def get_topic_info(self):
        return self.topic_info

    def get_target_info(self):
        return self.target_info

    def _get_generator(self, lazy=False):
        if self.model_lib == 'transformers':
            return llms.Transformers(self.model_name, self.model_kwargs, self.tokenizer_kwargs, lazy=lazy)
        else:
            raise ValueError(f"LLM library '{self.model_lib}' not implemented")
        
    def _get_vector_probs(self, noun_phrase, vector, topic_docs, model, tokenizer):
        # for each noun_phrase, get probability of doc along vector
        prompt = "{text}. Targeted towards {noun_phrase}, is the preceeding text more likely to be {sent_a} or {sent_b}?".format(text="{text}", noun_phrase=noun_phrase, sent_a=vector.sent_a, sent_b=vector.sent_b)
        # get probability of doc along vector
        probs = self.generator.get_prompt_response_probs(prompt, topic_docs, model, tokenizer, vector.sent_a, vector.sent_b)
        return probs

                
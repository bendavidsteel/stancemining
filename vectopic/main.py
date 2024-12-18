from bertopic import BERTopic
import numpy as np
import pandas as pd
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

from vectopic import llms
from vectopic.ngram_gen import NGramGeneration

keyword_prompt = """<|system|>You are a helpful, respectful and honest assistant for labeling topics..</s>
<|user|>
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.</s>
<|assistant|>"""


class Vector:
    def __init__(self, sent_a, sent_b):
        self.sent_a = sent_a
        self.sent_b = sent_b
        self.neutral = 'neutral'

class VectorTopic:
    def __init__(self, 
                 vector, 
                 method='llm', 
                 num_representative_docs=5,
                 model_lib='transformers', 
                 model_name='microsoft/Phi-3.5-mini-instruct', 
                 model_kwargs={}, 
                 tokenizer_kwargs={}
                ):
        self.vector = vector
        self.method = method

        self.num_representative_docs = num_representative_docs

        self.model_lib = model_lib
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs

        self.generator = self._get_generator()

        self.embedding_model = 'paraphrase-MiniLM-L6-v2'

    def _get_embedding_model(self):
        model = SentenceTransformer(self.embedding_model)
        return model

    def _get_embeddings(self, docs):
        embedding_model = self._get_embedding_model()
        embeddings = embedding_model.encode(docs)
        return embeddings

    def _ask_llm_differences(self, docs):
        prompt = f"""Describe some specific stance targets that some of these documents would disagree on.
        {docs}. Please output 3 suggestions of stance targets of max 3 words, separated by commas. Do not output 'something vs something', just name one version of the stance target."""
        outputs = self.generator.generate(prompt, max_new_tokens=30, num_samples=3)
        ngrams = [g.split('\n')[0] for g in outputs]
        ngrams = [i for g in ngrams for i in g.split(',')]
        ngrams = [g.lower().replace(f'{self.vector.sent_a}:', '').replace(f'{self.vector.sent_b}:', '') for g in ngrams]
        ngrams = [g for g in ngrams if g != '']
        ngrams = [g.strip() for g in ngrams]
        return list(set(ngrams))
    
    def _ask_llm_score(self, ngram, docs):
        text_name = "comment"
        prompt = f"""Targeted towards {ngram}, is the following {text_name} more likely to be {self.vector.sent_a}, {self.vector.neutral}, or {self.vector.sent_b}? {{doc}}. Output either {self.vector.sent_a}, {self.vector.neutral}, or {self.vector.sent_b}."""
        return self.generator.get_prompt_response_probs(prompt, docs, self.vector.sent_a, self.vector.sent_b, self.vector.neutral)

    def _ask_llm_class(self, ngram, docs):
        text_name = "comment"
        prompt = f"""Targeted towards {ngram}, is the following {text_name} more likely to be {self.vector.sent_a}, {self.vector.neutral}, or {self.vector.sent_b}? {{doc}}. Output either {self.vector.sent_a}, {self.vector.neutral}, or {self.vector.sent_b}."""
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

    def _get_base_topic_model(self, bertopic_kwargs):
        return BERTopic(**bertopic_kwargs)

    def _topic_llm_fit_transform(self, docs, embeddings=None, bertopic_kwargs={}):
        if embeddings is None:
            embeddings = self._get_embeddings(docs)

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
        self.topic_info = topic_model.get_topic_info()
        target_info = []

        self.topic_info = self.topic_info.assign(ngrams=[[]] * len(self.topic_info))
        documents = documents.assign(Targets=[[]] * len(documents), Polarities=[[]] * len(documents))
        for idx, topic in self.topic_info.iterrows():
            # prompt LLM with sample of docs, and ask for suggestions of disagreement
            ngrams = self._ask_llm_differences(topic['Representative_Docs'])
            topic_id = topic['Topic']
            topic_docs = documents[documents['Topic'] == topic_id]
            ngrams_data = []
            for ngram in ngrams:
                # calculate probability of doc along vector
                # prompt LLM with target and docs, use token probs as scalar along vector
                raw_topic_docs = topic_docs['Document']
                classifications = self._ask_llm_class(ngram, raw_topic_docs)
                ngram_polarities = [1 if c == self.vector.sent_a else -1 if c == self.vector.sent_b else 0 for c in classifications]
                topic_docs[f'{ngram}_Polarity'] = ngram_polarities
                # topic_info.at[idx, f"{ngram}_polarities"] = polarities
                ngram_data = {'ngram': ngram, 'polarities': ngram_polarities}
                ngrams_data.append(ngram_data)
                ngram_data['topic_id'] = topic_id
                target_info.append(ngram_data)
            self.topic_info.at[idx, f"ngrams"] = ngrams_data

            for i, doc in topic_docs.iterrows():
                targets = [ngram for ngram in ngrams]
                doc_polarities = [{'ngram': ngram, 'polarity': doc[f"{ngram}_Polarity"]} for ngram in ngrams]
                documents.at[i, 'Targets'] = targets
                documents.at[i, 'Polarities'] = doc_polarities

        target_info_df = pd.DataFrame(target_info)
        target_info_df = target_info_df.groupby('ngram').agg({'topic_id': list, 'polarities': sum}).reset_index()
        target_info_df['mean'] = target_info_df['polarities'].apply(lambda x: np.mean(x))
        target_info_df['var'] = target_info_df['polarities'].apply(lambda x: np.var(x))
        self.target_info = target_info_df

        ngram_to_var = {row['ngram']: row['var'] for idx, row in target_info_df.iterrows()}

        # calculate ngram with largest variance
        documents['Target'] = documents['Targets'].apply(lambda x: max(x, key=lambda ngram: ngram_to_var[ngram]) if len(x) > 0 else None)

        return documents
    
    def _llm_topic_fit_transform(self, docs, bertopic_kwargs={}):
        
        target_info = []
        documents = pd.DataFrame({"Document": docs, "ID": range(len(docs))})
        documents = documents.assign(Targets=[[]] * len(documents), Polarities=[[]] * len(documents))
        for idx, doc in documents.iterrows():
            # prompt LLM with sample of docs, and ask for suggestions of disagreement
            ngrams = self._ask_llm_differences(doc)
            ngrams_data = []
            for ngram in ngrams:
                # calculate probability of doc along vector
                # prompt LLM with target and docs, use token probs as scalar along vector
                classifications = self._ask_llm_class(ngram, doc)
                ngram_polarities = [1 if c == self.vector.sent_a else -1 if c == self.vector.sent_b else 0 for c in classifications]
                # topic_docs[f'{ngram}_Polarity'] = ngram_polarities
                # topic_info.at[idx, f"{ngram}_polarities"] = polarities
                ngram_data = {'ngram': ngram, 'polarities': ngram_polarities}
                ngrams_data.append(ngram_data)
                # ngram_data['topic_id'] = topic_id
                target_info.append(ngram_data)
            self.topic_info.at[idx, f"ngrams"] = ngrams_data

            targets = [ngram for ngram in ngrams]
            doc_polarities = [{'ngram': ngram, 'polarity': doc[f"{ngram}_Polarity"]} for ngram in ngrams]
            documents.at[idx, 'Targets'] = targets
            documents.at[idx, 'Polarities'] = doc_polarities

        topic_model = self._get_base_topic_model(bertopic_kwargs)
        topics, probs = topic_model.fit_transform(documents.explode('ngrams').to_list())
        repr_docs, _, _, _ = topic_model._extract_representative_docs(
            topic_model.c_tf_idf_,
            documents,
            topic_model.topic_representations_,
            nr_samples=500,
            nr_repr_docs=self.num_representative_docs,
        )
        topic_model.representative_docs_ = repr_docs
        topic_info = topic_model.get_topic_info()

        target_info_df = pd.DataFrame(target_info)
        target_info_df = target_info_df.groupby('ngram').agg({'topic_id': list, 'polarities': sum}).reset_index()
        target_info_df['mean'] = target_info_df['polarities'].apply(lambda x: np.mean(x))
        target_info_df['var'] = target_info_df['polarities'].apply(lambda x: np.var(x))
        self.target_info = target_info_df

        ngram_to_var = {row['ngram']: row['var'] for idx, row in target_info_df.iterrows()}

        # calculate ngram with largest variance
        documents['Target'] = documents['Targets'].apply(lambda x: max(x, key=lambda ngram: ngram_to_var[ngram]) if len(x) > 0 else None)

        return documents
    
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
            # to produce generations of suggestions for ngrams
            ngrams = ask_llm(topic['representative_docs'])
            topic_id = topic['id']
            topic_docs = [docs[i] for i in range(len(docs)) if topics[i] == topic_id]
            ngram_probs = []
            means = []
            vars = []
            for ngram in ngrams:
                # calculate probability of doc along vector
                # https://arjunbansal.substack.com/p/llms-know-more-than-what-they-say
                # get ngram specific target vector, and check projection of doc activations along that vector
                probs = ask_llm(ngram, topic_docs)
                ngram_probs.append(probs)
                means.append(np.mean(probs))
                vars.append(np.var(probs))

            self.topic_info.at[idx, 'ngram_probs'] = ngram_probs
            self.topic_info.at[idx, 'ngram_means'] = means
            self.topic_info.at[idx, 'ngram_vars'] = vars

        return documents
    
    def _get_targets_probs_polarity(self, documents):
        all_targets = self.target_info['ngram'].to_list()
        target_to_idx = {target: idx for idx, target in enumerate(all_targets)}
        doc_targets = documents['Target'].to_list()
        doc_targets = [[t] if t is not None else [] for t in doc_targets]

        num_targets = len(self.target_info)
        
        polarities = np.zeros((len(documents), num_targets))
        probs = np.zeros((len(documents), num_targets))
        polarities_dict = documents['Polarities'].to_list()
        for i, doc_polarities in enumerate(polarities_dict):
            doc_num_targets = len(doc_polarities)
            for d in doc_polarities:
                idx = target_to_idx[d['ngram']]
                polarities[i, idx] = d['polarity']
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
            documents = self._llm_topic_fit_transform(docs, embeddings)
        elif self.method == 'steeringvectors':
            documents = self._embedding_fit_transform(docs, embeddings)
        else:
            raise ValueError(f"Method '{self.method}' not implemented")
        
        return self._get_targets_probs_polarity(documents)

    def get_topic_info(self):
        return self.topic_info

    def get_target_info(self):
        return self.target_info

    def _get_generator(self):
        if self.model_lib == 'transformers':
            return llms.Transformers(self.model_name, self.model_kwargs, self.tokenizer_kwargs)
        else:
            raise ValueError(f"LLM library '{self.model_lib}' not implemented")
        
    def _get_vector_probs(self, ngram, vector, topic_docs, model, tokenizer):
        # for each ngram, get probability of doc along vector
        prompt = "{text}. Targeted towards {ngram}, is the preceeding text more likely to be {sent_a} or {sent_b}?".format(text="{text}", ngram=ngram, sent_a=vector.sent_a, sent_b=vector.sent_b)
        # get probability of doc along vector
        probs = self.generator.get_prompt_response_probs(prompt, topic_docs, model, tokenizer, vector.sent_a, vector.sent_b)
        return probs

                
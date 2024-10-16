from bertopic import BERTopic
import numpy as np
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




def get_embeddings(docs):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs)
    return embeddings




class Vector:
    def __init__(self, sent_a, sent_b):
        self.sent_a = sent_a
        self.sent_b = sent_b
        self.neutral = 'neutral'

class VectorTopic:
    def __init__(self, vector, method=None, model_lib='transformers', model_name='microsoft/Phi-3.5-mini-instruct', model_kwargs={}, tokenizer_kwargs={}):
        self.vector = vector
        self.method = method

        self.model_lib = model_lib
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs

        self.generator = self.get_generator()

    def _ask_llm_differences(self, docs):
        prompt = f"""Describe a specific thing that some of these documents would {self.vector.sent_a}, and some would {self.vector.sent_b}.
        {docs}. Output max 3 words describing it."""
        ngrams = self.generator.generate(prompt, max_new_tokens=3, num_samples=3)
        ngrams = [i for g in ngrams for i in g.split(',')]
        ngrams = [i for g in ngrams for i in g.split('\n')]
        ngrams = [g for g in ngrams if g != '']
        ngrams = [g.lower() for g in ngrams]
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

    def fit_transform(self, docs, embeddings=None, bertopic_kwargs={}):
        # could also compute generalized stance (i.e. sentiment), then find vector
        # then use steering vector to find the difference along the vector
        # https://www.lesswrong.com/posts/ndyngghzFY388Dnew/implementing-activation-steering

        # the same idea more fleshed out for evaluation 
        # https://arjunbansal.substack.com/p/llms-know-more-than-what-they-say
        if self.method == 'llm':
            if embeddings is None:
                embeddings = get_embeddings(docs)

            topic_model = BERTopic(**bertopic_kwargs)
            topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
            topic_info = topic_model.get_topic_info()

            topic_info = topic_info.assign(ngrams=[[]] * len(topic_info))
            for idx, topic in topic_info.iterrows():
                # prompt LLM with sample of docs, and ask for suggestions of disagreement
                ngrams = self._ask_llm_differences(topic['Representative_Docs'])
                topic_id = topic['Topic']
                topic_docs = [docs[i] for i in range(len(docs)) if topics[i] == topic_id]
                ngram_data = []
                for ngram in ngrams:
                    # calculate probability of doc along vector
                    # prompt LLM with target and docs, use token probs as scalar along vector
                    classifications = self._ask_llm_class(ngram, topic_docs)
                    polarities = [1 if c == self.vector.sent_a else -1 if c == self.vector.sent_b else 0 for c in classifications]
                    # topic_info.at[idx, f"{ngram}_polarities"] = polarities
                    ngram_data.append({'ngram': ngram, 'mean': np.mean(polarities), 'var': np.var(polarities)})
                topic_info.at[idx, f"ngrams"] = ngram_data

            return topic_info
        
        if self.method == 'steeringvectors':
            if embeddings is None:
                embeddings = get_embeddings(docs)

            model, tokenizer = get_generator()

            # TODO potentially use other topic modelling allowing doc to have multiple topics
            # or use probs to allow docs to fit into multiple topics
            topic_model = BERTopic()
            topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
            topic_info = topic_model.get_topic_info()

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

                topic_info.at[idx, 'ngram_probs'] = ngram_probs
                topic_info.at[idx, 'ngram_means'] = means
                topic_info.at[idx, 'ngram_vars'] = vars

            return topic_info
        else:
            raise ValueError(f"Method '{self.method}' not implemented")
                
    def get_generator(self):
        if self.model_lib == 'transformers':
            return llms.Transformers(self.model_name, self.model_kwargs, self.tokenizer_kwargs)
        else:
            raise ValueError(f"LLM library '{self.model_lib}' not implemented")
        
    def get_vector_probs(self, ngram, vector, topic_docs, model, tokenizer):
        # for each ngram, get probability of doc along vector
        prompt = "{text}. Targeted towards {ngram}, is the preceeding text more likely to be {sent_a} or {sent_b}?".format(text="{text}", ngram=ngram, sent_a=vector.sent_a, sent_b=vector.sent_b)
        # get probability of doc along vector
        probs = self.generator.get_prompt_response_probs(prompt, topic_docs, model, tokenizer, vector.sent_a, vector.sent_b)
        return probs

                
from bertopic import BERTopic
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import torch

from vectopic.ngram_gen import NGramGeneration

keyword_prompt = """<|system|>You are a helpful, respectful and honest assistant for labeling topics..</s>
<|user|>
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.</s>
<|assistant|>"""


def get_generator():
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    # Pipeline
    return model, tokenizer

def get_embeddings(docs):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs)
    return embeddings

def calculate_sequence_prob(model, tokenizer, input_ids, seq_ids):
    # TODO should be a faster way of doing this?
    prob = 1.0
    current_input = input_ids.clone()
    
    for token_id in seq_ids:
        with torch.no_grad():
            outputs = model(current_input)
            next_token_logits = outputs[0][:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            token_prob = next_token_probs[0, token_id].item()
            prob *= token_prob
            
            # Append the token to the input for the next iteration
            current_input = torch.cat([current_input, torch.tensor([[token_id]])], dim=1)
    
    return prob

def get_prompt_response_probs(prompt, docs, model, tokenizer, sent_a, sent_b):
    sent_a_ids = tokenizer.encode(sent_a, add_special_tokens=False)
    sent_b_ids = tokenizer.encode(sent_b, add_special_tokens=False)
    probs = []
    
    for doc in tqdm(docs):
        formatted_prompt = prompt.format(text=doc)
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
        
        # Calculate probability for sent_a
        prob_a = calculate_sequence_prob(model, tokenizer, input_ids, sent_a_ids)
        
        # Calculate probability for sent_b
        prob_b = calculate_sequence_prob(model, tokenizer, input_ids, sent_b_ids)
        
        # Calculate relative probability
        relative_prob = prob_a / (prob_a + prob_b)
        probs.append(relative_prob)
    
    return probs

def get_vector_probs(ngram, vector, topic_docs, model, tokenizer):
    # for each ngram, get probability of doc along vector
    prompt = "{text}. Targeted towards {ngram}, is the preceeding text more likely to be {sent_a} or {sent_b}?".format(text="{text}", ngram=ngram, sent_a=vector.sent_a, sent_b=vector.sent_b)
    # get probability of doc along vector
    probs = get_prompt_response_probs(prompt, topic_docs, model, tokenizer, vector.sent_a, vector.sent_b)
    return probs


class Vector:
    def __init__(self, sent_a, sent_b):
        self.sent_a = sent_a
        self.sent_b = sent_b

class VectorTopic:
    def __init__(self, vector, method=None):
        self.vector = vector
        self.method = method

        self.model, self.tokenizer = get_generator()

    def _ask_llm_differences(self, docs):
        prompt = f"""What is the main difference by which these documents {self.vector.sent_a} or {self.vector.sent_b}?
        {docs}"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        num_samples = 3
        outputs = self.model.generate(input_ids, max_new_tokens=20, do_sample=True, num_return_sequences=num_samples, pad_token_id=self.tokenizer.eos_token_id)
        ngrams = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return ngrams
    
    def _ask_llm_score(self, ngram, docs):
        prompt = f"""Targeted towards {ngram}, is the preceeding text more likely to be {self.vector.sent_a} or {self.vector.sent_b}?"""
        return get_prompt_response_probs(prompt, docs, self.model, self.tokenizer, self.vector.sent_a, self.vector.sent_b)

    def fit_transform(self, docs, embeddings=None):
        # could also compute generalized stance (i.e. sentiment), then find vector
        # then use steering vector to find the difference along the vector
        # https://www.lesswrong.com/posts/ndyngghzFY388Dnew/implementing-activation-steering

        # the same idea more fleshed out for evaluation 
        # https://arjunbansal.substack.com/p/llms-know-more-than-what-they-say
        if self.method == 'llm':
            if embeddings is None:
                embeddings = get_embeddings(docs)

            topic_model = BERTopic()
            topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
            topic_info = topic_model.get_topic_info()

            for idx, topic in topic_info.iterrows():
                # prompt LLM with sample of docs, and ask for suggestions of disagreement
                ngrams = self._ask_llm_differences(topic['Representative_Docs'])
                topic_id = topic['id']
                topic_docs = [docs[i] for i in range(len(docs)) if topics[i] == topic_id]
                ngram_probs = []
                means = []
                vars = []
                for ngram in ngrams:
                    # calculate probability of doc along vector
                    # prompt LLM with target and docs, use token probs as scalar along vector
                    probs = self._ask_llm_score(ngram, topic_docs)
                    ngram_probs.append(probs)
                    means.append(np.mean(probs))
                    vars.append(np.var(probs))

                topic_info.at[idx, 'ngram_probs'] = ngram_probs
                topic_info.at[idx, 'ngram_means'] = means
                topic_info.at[idx, 'ngram_vars'] = vars

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
                


                
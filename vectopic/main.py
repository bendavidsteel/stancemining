from bertopic import BERTopic
from bertopic.representation import TextGeneration
from bertopic.representation._utils import truncate_document
from ctransformers import AutoModelForCausalLM
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
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


def get_generator(num_gpu_layers=10):
    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/zephyr-7B-alpha-GGUF",
        model_file="zephyr-7b-alpha.Q4_K_M.gguf",
        model_type="mistral",
        gpu_layers=num_gpu_layers,
        hf=True
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")

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

    def fit_transform(self, docs, embeddings=None):
        if self.method == 1:
            if embeddings is None:
                embeddings = get_embeddings(docs)

            model, tokenizer = get_generator()

            representation_model = {"ngrams": NGramGeneration(model, tokenizer)}
            topic_model = BERTopic(representation_model=representation_model)
            topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
            topic_info = topic_model.get_topic_info()

            for idx, topic in topic_info.iterrows():
                ngrams = topic['ngrams']
                topic_id = topic['id']
                topic_docs = [docs[i] for i in range(len(docs)) if topics[i] == topic_id]
                ngram_probs = []
                means = []
                vars = []
                for ngram in ngrams:
                    

                    ngram_probs.append(probs)
                    means.append(np.mean(probs))
                    vars.append(np.var(probs))

                topic_info.at[idx, 'ngram_probs'] = ngram_probs
                topic_info.at[idx, 'ngram_means'] = means
                topic_info.at[idx, 'ngram_vars'] = vars

            return topic_info
                


                
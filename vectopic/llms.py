import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

class BaseLLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, prompt, max_new_tokens=100, num_samples=3):
        raise NotImplementedError
    
    
class Transformers(BaseLLM):
    def __init__(self, model_name, model_kwargs={}, tokenizer_kwargs={}):
        super().__init__(model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    
    def generate(self, prompt, max_new_tokens=100, num_samples=3):
        conversation = [
            {'role': 'user', 'content': prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(conversation, return_dict=True, return_tensors='pt', add_generation_prompt=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        generate_kwargs = {}
        if num_samples > 1:
            generate_kwargs['num_return_sequences'] = num_samples
            generate_kwargs['num_beams'] = 10 * num_samples
            generate_kwargs['do_sample'] = True

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **generate_kwargs)
        return [self.tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True) for output in outputs]

    def calculate_sequence_prob(self, inputs, seq_id):
        # TODO should be a faster way of doing this?
        prob = 1.0
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            token_prob = next_token_probs[seq_id].item()
            prob *= token_prob
        
        return prob

    def get_prompt_response_probs(self, prompt, docs, sent_a, sent_b, neutral):
        sent_a_ids = self.tokenizer.encode(sent_a, add_special_tokens=False)
        sent_b_ids = self.tokenizer.encode(sent_b, add_special_tokens=False)
        neutral_ids = self.tokenizer.encode(neutral, add_special_tokens=False)
        probs = []
        
        for doc in tqdm.tqdm(docs):
            formatted_prompt = prompt.format(doc=doc)
            messages = [
                {'role': 'user', 'content': formatted_prompt}
            ]
            inputs = self.tokenizer.apply_chat_template(messages, return_dict=True, return_tensors='pt', add_generation_prompt=True)
            
            # Calculate probabilities
            prob_a = self.calculate_sequence_prob(inputs, sent_a_ids)
            prob_b = self.calculate_sequence_prob(inputs, sent_b_ids)
            prob_neutral = self.calculate_sequence_prob(inputs, neutral_ids)
            
            # Normalize probabilities
            total_prob = prob_a + prob_b + prob_neutral
            prob_a /= total_prob
            prob_b /= total_prob
            prob_neutral /= total_prob
            # get final prob across the spectrum

        
        return probs
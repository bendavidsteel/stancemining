

class BaseLLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, prompt, max_new_tokens=100, num_samples=3):
        raise NotImplementedError
    
class LlamaCppPython(BaseLLM):
    def __init__(self):
        from llama_cpp import Llama
        self.llm = Llama.from_pretrained(
            repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
            filename="*q8_0.gguf",
            verbose=False
        )

    def generate(self, prompt, max_new_tokens=100, num_samples=3):
        output = self.llm(
            prompt, # Prompt
            max_tokens=max_new_tokens, # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
            echo=False # Echo the prompt back in the output
        ) # Generate a completion, can also call create_completion)
        return output
    
class Transformers(BaseLLM):
    def __init__(self, model_name):
        super().__init__(model_name)
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
    
    def generate(self, prompt, max_new_tokens=100, num_samples=3):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens, num_return_sequences=num_samples, do_sample=True)
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
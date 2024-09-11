from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def main():
    # Load model directly
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    input_text = "Hi, what is 7 plus 11?"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    output = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == '__main__':
    main()
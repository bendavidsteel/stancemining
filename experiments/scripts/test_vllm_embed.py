import vllm
import sentence_transformers
import numpy as np

def main():
    documents = [
        "This is a test sentence.",
        "This is another test sentence.",
        "Yet another example of a sentence."]

    model = sentence_transformers.SentenceTransformer('all-MiniLM-L12-v2')
    st_embeddings = model.encode(documents)

    model = vllm.LLM(model='sentence-transformers/all-MiniLM-L12-v2')
    vllm_outputs = model.embed(documents)
    vllm_embeddings = [np.array(o.outputs.embedding) for o in vllm_outputs]

    for i in range(len(documents)):
        print(f"Document: {documents[i]}")
        print(f"ST Embedding: {st_embeddings[i][:10]}...")  # Print first 10 elements for brevity
        print(f"vLLM Embedding: {vllm_embeddings[i][:10]}...")  # Print first 10 elements for brevity
        for atol in [1e-4, 1e-3, 1e-2]:
            print(f"Are embeddings close with atol={atol}? {np.allclose(st_embeddings[i], vllm_embeddings[i], atol=atol)}")
        print("-" * 50)

if __name__ == '__main__':
    main()
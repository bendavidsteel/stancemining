
from vectopic import get_vector_probs, get_generator, Vector

def test_get_prompt_responses():
    docs = ["doc1", "doc2", "doc3"]
    model, tokenizer = get_generator()
    sent_a = "sent_a"
    sent_b = "sent_b"
    vector = Vector(sent_a, sent_b)
    probs = get_vector_probs("ngram", vector, docs, model, tokenizer)
    assert len(probs) == 3
    assert all([0 <= prob <= 1 for prob in probs])

if __name__ == '__main__':
    test_get_prompt_responses()
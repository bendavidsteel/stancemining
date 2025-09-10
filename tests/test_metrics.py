from stancemining import metrics

def test_repetition():
    texts = ['hello this is ben this is ben this is ben']
    rep = metrics.max_ngram_repetition(texts)
    assert rep == 3

if __name__ == '__main__':
    test_repetition()
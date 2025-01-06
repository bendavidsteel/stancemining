import collections
import hashlib
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import transformers


def sent_to_words(sentences, min_len=2, max_len=15):
    # tokenize words
    import gensim
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True, 
                                            min_len=min_len, max_len=max_len)  # deacc=True removes punctuations


def remove_stopwords(texts, default='english', extensions=None):
    # nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = []
    if default is not None:
        stop_words.extend(stopwords.words(default))
    if extensions is not None:
        stop_words.extend(extensions)
    import gensim
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(data_words):
    import gensim
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshld fewer phrases
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in data_words], bigram, bigram_mod


def make_trigrams(data_words, bigram, bigram_mod):
    import gensim
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[bigram_mod[doc]] for doc in data_words], trigram, trigram_mod


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    '''
    Lemmatization for LDA topic modeling.
    '''
    import spacy
    """https://spacy.io/api/annotation"""
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        # do lemmatization and only keep the types of tokens in allowed_postags
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def lemmatization2(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    '''
    Lemmatization for BERT. 
    Although BERT has its own tokenizer, we need match the words for BERT and LDA.
    '''
    import spacy
    """https://spacy.io/api/annotation"""
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        # for tokens whose types in allowed_postages do lemmatization otherwise keep the original form
        texts_out.append([str(token.lemma_) if token.pos_ in allowed_postags else token for token in doc])
    return texts_out


def lemmatization3(texts):
    '''
    Lemmatization for leave-out estimator
    '''
    import spacy
    """https://spacy.io/api/annotation"""
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        # for all tokens do lemmatization and keep all tokens
        texts_out.append([str(token.lemma_) for token in doc])
    return texts_out

def create_dict_corpus(data_words):
    import gensim.corpora as corpora
    
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)

    # Create Corpus
    texts = data_words

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    return corpus, id2word

def preprocessing_lda(data):
    import re
    
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    # tokenize words and clean-up text
    data_words = list(sent_to_words(data))

    # remove stop words
    # need to remove the news source names
    data_words_nostops = remove_stopwords(data_words, 
                                          extensions=['from', 'subject', 're', 'edu', 
                                                       'use', 'rt', 'cnn', 'fox', 'huffington', 'breitbart'])

    # form bigrams
    data_words_bigrams, _, _ = make_bigrams(data_words_nostops)

    #  do lemmatization keeping only noun, adj, vb, adv, propnoun
    # other tokens are not useful for topic modeling
    data_lematized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN'])
    
    corpus, id2word = create_dict_corpus(data_lematized)

    return data_lematized, corpus, id2word

def preprocessing_bert(data):
    import re
    
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # tokenize words and clean-up text
    data_words = list(sent_to_words(data,min_len=1, max_len=30))

    # remove stop words
    data_words_nostops = remove_stopwords(data_words, default=None,
                                          extensions=['cnn', 'fox', 'huffington', 'breitbart'])

    # form bigrams
    data_words_bigrams, _, _ = make_bigrams(data_words)

    #  do lemmatization for only noun, adj, vb, adv propnoun, following the lemmatization for LDA
    #  keep the others which will be used as context
    data_lematized = lemmatization2(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN'])
    
    return data_lematized

def train_lda_model(data_path, corpus, id2word, texts, model_type='lda', start=10, limit=50, step=3):
    import os
    import gensim
    import numpy as np
    from gensim.models import CoherenceModel
    
    if model_type == 'mallet' and not os.path.exists('mallet-2.0.8.zip'):
        os.system('wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip')
        os.system('unzip mallet-2.0.8.zip')
    mallet_path = 'mallet-2.0.8/bin/mallet'
    
    import threading
    class MyThread(threading.Thread):
        def __init__(self, func, args=()):
            super(MyThread, self).__init__()
            self.func = func
            self.args = args

        def run(self):
            self.result = self.func(*self.args)

        def get_result(self):
            try:
                return self.result
            except Exception:
                return None
    
    def lda_model_func(corpus, id2word, num_topics):
        print(f'Training model with {num_topics} topics')
        model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=42,
                                           chunksize=100,
                                           passes=10,
                                           per_word_topics=True)
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=id2word, coherence='c_v')
        coh_v = coherence_model.get_coherence()
        print(f'num of topic: {num_topics}/{limit}, coherence value: {coh_v}')
        return model, coh_v
    
    
    def mallet_model_func(mallet_path, corpus, num_topics, id2word):
        print(f'Training model with {num_topics} topics')
        model = gensim.models.wrappers.LdaMallet(mallet_path, 
                                                     corpus=corpus, 
                                                     num_topics=num_topics, 
                                                     id2word=id2word, 
                                                     iterations=50,
                                                     workers=1,
                                                     random_seed=42)
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=id2word, coherence='c_v')
        coh_v = coherence_model.get_coherence()
        print(f'num of topic: {num_topics}/{limit}, coherence value: {coh_v}')
        return model, coh_v
    
    coherence_values_dict = {}
    coherence_values = []
    model_dict = {}
    for num_topics in range(start, limit+1, step):
        if model_type == 'lda':
            model, coh_v = lda_model_func(corpus, id2word, num_topics)
        else:
            model, coh_v = mallet_model_func(mallet_path, corpus, num_topics, id2word)
        model_dict[num_topics] = model
        coherence_values_dict[num_topics] = coh_v
        coherence_values.append(coh_v)
    
    import matplotlib.pyplot as plt
    plt.plot(range(start, limit+1, step), coherence_values, marker='o')
    plt.xlabel('n_topics')
    plt.ylabel('coherence score')
    plt.savefig(os.path.join(data_path, f'{model_type}_coherence_values_topics.png'))
    plt.close()
    
    max_idx = np.argmax(coherence_values)
    
    return model_dict, coherence_values_dict, max_idx+start

def search_in_list(list1, list2):
    '''
    search the indices of the topic keywords in the tokenized document -- a list of tokens
    '''
    idxes = []
    for i in range(len(list2)):
        if list1[0] == list2[i]:
            if (i + len(list1) <= len(list2)) and list2[i:i+len(list1)] == list1:
                idxes += list(range(i, i+len(list1)))
    return idxes


def get_topic_masks(data, topics):
    
    return topic_masks

# rank topics by the mass of probabilities
def rank_topics(lda_model, corpus):
    idx_prob = [[i, 0] for i in range(lda_model.num_topics)]
    for idx_doc, rows in enumerate(lda_model[corpus]):
        document_topics = rows[0]
        for j, (idx_topic, prob) in enumerate(document_topics):
            idx_prob[idx_topic][1] += prob
    idx_prob.sort(key=lambda x:x[1], reverse=True)
    return idx_prob

# figure out the the probability that each document is associated with each topic
# only keep those pairs with prob>=threshold (0.15 here)
def get_doc2topics(ldamodel, corpus, threshold=0.15):
    data = []
    for idx_doc in range(len(corpus)):
        corpus_doc = corpus[idx_doc]
        document_topics = ldamodel.get_document_topics(corpus_doc, minimum_probability=threshold, per_word_topics=False)
        for j, (idx_topic, prob) in enumerate(document_topics):
            if prob < threshold:
                continue
            data.append([idx_doc, idx_topic, prob])
    df = pd.DataFrame(data, columns=['idx_doc', 'idx_topic', 'prob'])
    return df

def topic_modelling(data_path, df_news, train=True):
    texts = pickle.load(open(os.path.join(data_path, 'texts_processed_lda.pkl'), 'rb'))
    corpus, id2word = pickle.load(open(os.path.join(data_path, 'corpus_lda.pkl'), 'rb'))
    # train LDA topic models with different K (n_topics)
    # lda_models is a dictionary {K:model_K, K+1: model_K+1, .....}
    if not os.path.exists(os.path.join(data_path, 'lda_models.pkl')):
        lda_models, coh_values, max_idx = train_lda_model(data_path, corpus, id2word, texts, model_type='lda', start=10, limit=50, step=1)
        pickle.dump((lda_models, coh_values, max_idx), open(os.path.join(data_path, 'lda_models.pkl'), 'wb'))
    else:
        lda_models, coh_values, max_idx = pickle.load(open(os.path.join(data_path, 'lda_models.pkl'), 'rb'))

    # print the coherecen values of different K (n_topics)
    # K=15 gives the best coherence value but the topics are not good after manual inspection
    df_ntopics_coh = pd.DataFrame(list(coh_values.items()), columns=['n_topics', 'coh_val'])
    # choose LDA with highest coh_val
    highest_coh_lda_n_topics = int(df_ntopics_coh.sort_values(by='coh_val', ascending=False).iloc[0]['n_topics'])

    
    lda_model = lda_models[highest_coh_lda_n_topics]

    # index each topic
    topics = {x:y for x,y in lda_model.show_topics(num_topics=-1, num_words=10, formatted=False)}
    df_topic = []
    for i in range(len(topics)):
        df_topic.append([i, [each[0] for each in topics[i]]])

    topic_ranks = rank_topics(lda_model, corpus)

    for i in range(highest_coh_lda_n_topics):
        df_topic[i].append(topic_ranks[i][1])
    df_topic = pd.DataFrame(df_topic, columns=['topic_idx', 'topic_stems', 'probs'])

    if train:
        max_threshold = 0.8
        threshold = 0.15
        while threshold < max_threshold:
            df_doc_topic = get_doc2topics(lda_model, corpus, threshold=threshold)
            if len(df_doc_topic['idx_doc'].unique()) > 0.9 * len(df_news['idx'].unique()):
                threshold += 0.05
                continue
            else:
                break
            
        # create a validation set for finetuning the language model
        # articles that is not assigned to any topic (prob < 0.15) are in this set
        idxes_doc_val = set(df_news['idx'].unique().tolist()) - set(df_doc_topic['idx_doc'].unique().tolist())

        assert len(idxes_doc_val) > 0, 'No validation set is created. Please check the threshold of the topic probability.'
        assert len(idxes_doc_val) > 0.1 * len(df_news), 'The validation set is too small. Please check the threshold of the topic probability.'
        pickle.dump(idxes_doc_val, open(os.path.join(data_path, 'idxes_val.pkl'), 'wb'))
    else:
        threshold = 0.15
        df_doc_topic = get_doc2topics(lda_model, corpus, threshold=threshold)

    # save the data
    df_doc_topic.to_csv(os.path.join(data_path, 'df_doc_topic.csv'), index=False)
    df_topic.to_csv(os.path.join(data_path, 'df_topics.csv'), index=False)
    pickle.dump(topic_ranks, open(os.path.join(data_path, 'topic_ranks.pkl'), 'wb'))
    pickle.dump(lda_model, open(os.path.join(data_path, 'lda_model.pkl'), 'wb'))
    pickle.dump((lda_models, coh_values, max_idx), open(os.path.join(data_path, 'lda_models.pkl'), 'wb'))
    pickle.dump(lda_model.show_topics(num_topics=-1, num_words=10, formatted=False), 
                open(os.path.join(data_path, 'topics.pkl'), 'wb'))

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    texts_bert = pickle.load(open(os.path.join(data_path, 'texts_processed_bert.pkl'), 'rb'))
    text_encodings = tokenizer(pd.Series(texts_bert).apply(lambda x: ' '.join(x)).tolist(), padding=True, 
                            truncation=True)['input_ids']
    text_encodings = pd.Series(text_encodings)

    topic_masks = np.zeros((text_encodings.shape[0], 512))  # (n_docs, 512)
    for topic in lda_model.show_topics(num_topics=-1, num_words=50, formatted=False):
        idx = topic[0]
        topic_stems = [each[0] for each in topic[1]]
        stem_probs = [each[1] for each in topic[1]]
        stem_probs = np.array(stem_probs)
        stem_probs /= stem_probs.sum()  # normalize the weights of the top-n keywords
        stem_encodings = tokenizer(topic_stems, truncation=True)['input_ids'] # encode topic keywords using BERT
        doc_idxes = df_doc_topic[df_doc_topic['idx_topic'] == idx]['idx_doc'].to_list() # find the documents associated with the topic
        doc_encodings = text_encodings[doc_idxes]
        for doc_idx in doc_idxes:
            doc_encoding = doc_encodings[doc_idx]
            topic_mask = topic_masks[doc_idx]
            for stem_input_ids, stem_prob in zip(stem_encodings, stem_probs):
                idxes = search_in_list(stem_input_ids[1:-1], doc_encoding) # search each keyword in the document
                if idxes:
                    # if found multiple occurrences of the keywords, 
                    # then the weight of each occurrence will be devalued
                    topic_mask[idxes] += stem_prob / len(idxes) 
            if topic_mask.mean() > 0:
                topic_mask /= topic_mask.sum()   # normalize the mask
    pickle.dump(topic_masks.tolist(), open(os.path.join(data_path, 'topic_masks.pkl'), 'wb'))


class NewsDataset(Dataset):
    def __init__(self, texts, topic_masks, labels=None):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=512)
        self.encodings = encodings
        self.topic_masks = topic_masks
        if labels is not None:
            self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if hasattr(self, 'labels'):
            item['labels'] = torch.tensor(self.labels[idx])
        item['topic_masks'] = torch.tensor(self.topic_masks[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


class FC(nn.Module):
    def __init__(self, n_in, n_out):
        super(FC, self).__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.fc(x)


class Engine:
    def __init__(self, data_path, model_path=None, init_train=True, lr=1e-5, batch_size=24, epochs=50, shuffle=False, unfinetuned=False):
        # gpu
        # os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs('ckp', exist_ok=True)

        # dataset
        print('Loading data....')
        texts_processed = pd.Series(pickle.load(open(os.path.join(data_path, 'texts_processed_bert.pkl'), 'rb')))
        texts = texts_processed.apply(lambda x: ' '.join(x))
        df_news = pd.read_csv(os.path.join(data_path, 'df_news.csv'))
        if init_train:
            labels = df_news['label'].map(lambda l: {'favor': 1, 'against': 0}[l])
            if shuffle: # shuffle the labels to serve as the baseline, where the languge model cannot learn partisanship
                labels = labels.sample(frac=1)
        del df_news
        
        topic_masks = pd.Series(pickle.load(open(os.path.join(data_path, 'topic_masks.pkl'), 'rb')))
        if init_train:
            val_idexes = pickle.load(open(os.path.join(data_path, 'idxes_val.pkl'), 'rb'))
            train_idexes = set(list(range(len(texts_processed)))) - val_idexes
            # train_idexes = range(len(texts_processed))
            # val_idexes = range(len(texts_processed))

            train_idexes = np.array(list(train_idexes))
            val_idexes = np.array(list(val_idexes))
            train_mask = np.isin(np.arange(len(texts_processed)), train_idexes)
            val_mask = np.isin(np.arange(len(texts_processed)), val_idexes)
            print('Done.')

            texts_train = texts[train_mask].tolist()
            texts_val = texts[val_mask].tolist()
            labels_train = labels[train_mask].tolist()
            labels_val = labels[val_mask].tolist()
            topic_masks_train = topic_masks[train_mask].tolist()
            topic_masks_val = topic_masks[val_mask].tolist()
            print('Done\n')

            print('Preparing dataset....')
            train_dataset = NewsDataset(texts_train, topic_masks_train, labels=labels_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset = NewsDataset(texts_val, topic_masks_val, labels=labels_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        texts = texts.tolist()
        if init_train:
            labels = labels.tolist()
        else:
            labels = None
        topic_masks = topic_masks.tolist()
        dataset = NewsDataset(texts, topic_masks, labels=labels)
        loader = DataLoader(dataset, batch_size=int(1.5*batch_size))

        print('Done\n')

        # model
        print('Initializing model....')
        from transformers import AutoModelForSequenceClassification, AdamW
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        print('Done\n')
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        os.makedirs(os.path.join(data_path, 'ckp'), exist_ok=True)

        if model_path is None:
            if not shuffle:
                model_path = os.path.join(data_path, 'ckp', 'model.pt')
            else:
                model_path = os.path.join(data_path, 'ckp', 'model_shuffle.pt')

        self.device = device
        self.model = model
        self.loader = loader
        if init_train:
            self.optimizer = optimizer
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.labels = labels
        self.texts = texts
        self.model_path = model_path
        self.data_path = data_path
        self.shuffle = shuffle
        self.unfinetuned = unfinetuned
        self.epochs = epochs

    def train(self):

        if (not os.path.exists(self.model_path)) and (not self.unfinetuned):
            best_epoch_loss = float('inf')
            best_epoch_f1 = 0
            best_epoch = 0
            import copy
            best_state_dict = copy.deepcopy(self.model.state_dict())
            for epoch in range(self.epochs):
                print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")
                loss = self.train_epoch()
                acc, f1 = self.eval()

                if f1 > best_epoch_f1:
                    best_epoch = epoch
                    best_epoch_loss = loss
                    best_epoch_f1 = f1
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                print(
                    f'Epoch {epoch + 1}, Loss: {loss:.3f}, Acc: {acc:.3f}, F1: {f1:.3f}, '
                    f'Best Epoch:{best_epoch + 1}, '
                    f'Best Epoch F1: {best_epoch_f1:.3f}\n')

                if epoch - best_epoch >= 5:
                    break

            print('Saving the best checkpoint....')
            torch.save(best_state_dict, self.model_path)
            print(
                f'Best Epoch: {best_epoch + 1}, Best Epoch F1: {best_epoch_f1:.3f}, Best Epoch Loss: {best_epoch_loss:.3f}')
        self.calc_embeddings(True)
        self.calc_embeddings()

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
            loss = outputs[0].mean()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            if i % (len(self.train_loader) // 20) == 0:
                print(f'Batch: {i + 1}/{len(self.train_loader)}\tloss:{loss.item():.3f}')

        return epoch_loss / len(self.train_loader)

    def eval(self):
        self.model.eval()
        y_pred = []
        y_true = []
        print('Evaluating f1....')
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
                y_pred.append(outputs[1].detach().to('cpu').argmax(dim=1).numpy())
                y_true.append(labels.detach().to('cpu').numpy())
                if i % (len(self.val_loader) // 10) == 0:
                    print(f"{i}/{len(self.val_loader)}")
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return acc, f1

    # def load_model(self, model_path):
    #     print('Initializing model....')
    #     from transformers import AutoModelForSequenceClassification
    #     model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    #     model = nn.DataParallel(model)
    #     model_weights = torch.load(model_path)
    #     model.load_state_dict(model_weights)

    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     model.to(device)
    #     self.model = model

    def calc_embeddings(self, topic_emb=False):
        '''
        Calculate the embeddings of all documents
        :param topic_emb: boolean. If False, then the document embedding is the original BERT embedding ([CLS] embedding)
            If True, the document embedding is the document-contextualized topic embedding, with more focus on the topic keywords.
        :return: the embeddings of all documents
        '''

        os.makedirs(os.path.join(self.data_path, 'embeddings'), exist_ok=True)
        prediction_path = os.path.join(self.data_path, 'embeddings', f'predictions.pkl')
        if not topic_emb:
            embedding_path = os.path.join(self.data_path, 'embeddings', f'embeddings_unfinetuned={self.unfinetuned}.pkl')
        else:
            embedding_path = os.path.join(self.data_path, 'embeddings', f'topic_embeddings_unfinetuned={self.unfinetuned}.pkl')
        if self.shuffle:
            embedding_path = embedding_path[:-4] + '_shuffle.pkl'
        if (not os.path.exists(embedding_path)) or (not os.path.exists(prediction_path)):
            if not self.unfinetuned:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            embeddings = []
            predictions = []
            self.model.eval()
            print('Calculating embedding....')
            with torch.no_grad():
                for i, batch in enumerate(self.loader):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    preds_ = outputs['logits'].argmax(dim=1)
                    predictions.append(preds_.detach().to('cpu').numpy())
                    if not topic_emb:
                        embeddings_ = outputs['hidden_states'][-1][:, 0].detach().to('cpu').numpy()
                    else:
                        topic_masks = batch['topic_masks'].to(self.device).reshape(input_ids.shape[0],
                                                                                   input_ids.shape[1], -1)
                        embeddings_ = (topic_masks * outputs['hidden_states'][-1]).sum(dim=1).detach().to('cpu').numpy()
                    embeddings.append(embeddings_)
                    if i % 50 == 0:
                        print(f"{i}/{len(self.loader)}")
            print('Done')
            embeddings = np.concatenate(embeddings, axis=0)
            predictions = np.concatenate(predictions, axis=0)
            pickle.dump(embeddings, open(embedding_path, 'wb'))
            pickle.dump(predictions, open(prediction_path, 'wb'))
        else:
            embeddings = pickle.load(open(embedding_path, 'rb'))

        return embeddings

    def plot_embeddings(self, topic_emb=False, dim_reduction='pca'):
        '''
        Plot the document embeddings.
        :param topic_emb: see "calc_embeddings()"
        :param dim_reduction: which dimension reduction method to use. PCA or TSNE or UMAP
        :return:
        '''
        os.makedirs(os.path.join(self.data_path, 'embeddings'), exist_ok=True)
        print('Plotting....')
        print('Reducing dimension....')
        if not topic_emb:
            embedding_path = os.path.join(self.data_path, 'embeddings', f'embeddings_{dim_reduction}_unfinetuned={self.unfinetuned}.pkl')
        else:
            embedding_path = os.path.join(self.data_path, 'embeddings', f'topic_embeddings_{dim_reduction}_unfinetuned={self.unfinetuned}.pkl')
        if self.shuffle:
            embedding_path = embedding_path[:-4] + '_shuffle.pkl'
        if not os.path.exists(embedding_path):
            embeddings = self.calc_embeddings(topic_emb)
            if dim_reduction == 'pca':
                from sklearn.decomposition import PCA
                embeddings2 = PCA(n_components=2).fit_transform(embeddings)
            elif dim_reduction == 'tsne':
                from sklearn.manifold import TSNE
                embeddings2 = TSNE(n_components=2).fit_transform(embeddings)
            else:
                from umap import UMAP
                embeddings2 = UMAP(n_neighbors=15, n_components=2, min_dist=0, metric='cosine').fit_transform(embeddings)
            pickle.dump(embeddings2, open(embedding_path, 'wb'))
        else:
            embeddings2 = pickle.load(open(embedding_path, 'rb'))
        print('Done')
        data = pd.DataFrame(embeddings2, columns=['x', 'y'])
        if hasattr(self, 'labels'):
            data['labels'] = self.labels
        df_doc_topic = pd.read_csv(os.path.join(self.data_path, 'df_doc_topic.csv'))
        df_doc_topic = df_doc_topic.sort_values(by=['prob'], ascending=False).drop_duplicates(subset='idx_doc',
                                                                                              keep='first')

        data['cluster_labels'] = -1
        data['cluster_labels'][df_doc_topic['idx_doc'].tolist()] = df_doc_topic['idx_topic'].tolist()
        # only plot the documents in the 10 labeled topics
        data = data[data['cluster_labels'].isin([1, 2, 8, 9, 10, 11, 12, 27, 30, 33])]

        import matplotlib.pyplot as plt
        clustered = data[data['cluster_labels'] != -1]
        if 'labels' in data.columns:
            clustered1 = clustered[clustered['labels'] == 0][:200]
            clustered2 = clustered[clustered['labels'] == 1][:200]

        from matplotlib.backends.backend_pdf import PdfPages
        os.makedirs(os.path.join(self.data_path, 'fig'), exist_ok=True)
        if not topic_emb:
            fig_name = os.path.join(self.data_path, 'fig', f'embeddings_{dim_reduction}_unfinetuned={self.unfinetuned}.pdf')
            print(fig_name)
        else:
            fig_name = os.path.join(self.data_path, 'fig', f'topic_embeddings_{dim_reduction}_unfinetuned={self.unfinetuned}.pdf')
            print(fig_name)

        with PdfPages(fig_name) as pdf:
            _, _ = plt.subplots(figsize=(5, 5))
            if 'labels' in data.columns:
                plt.scatter(clustered1.x, clustered1.y, c=clustered1['cluster_labels'], marker='o', s=30, cmap='hsv_r', alpha=0.2,
                            label='liberal')
                plt.scatter(clustered2.x, clustered2.y, c=clustered2['cluster_labels'], marker='x', s=30, cmap='hsv_r', alpha=0.5,
                            label='conservative')
            else:
                plt.scatter(clustered.x, clustered.y, c=clustered['cluster_labels'], marker='o', s=30, cmap='hsv_r', alpha=0.2)

            plt.xlabel('dim_1', fontsize=12)
            plt.ylabel('dim_2', fontsize=12)
            plt.legend(fontsize=12)
            pdf.savefig()

    def get_polarization(self, polarization='emb', min_docs=10, max_docs=10):
        '''
        calculate the polarization score for each topic and save the ranking
        '''

        def select_docs(df_doc_topic, topic_idx, max_docs=10, min_docs=2, label=None):
            '''
            output the top-n documents from each source for each topic
            '''
            if label is not None:
                df_doc_topic = df_doc_topic[(df_doc_topic['label'] == label)]

            df = df_doc_topic[(df_doc_topic['idx_topic'] == topic_idx)].sort_values(by=['prob'], ascending=False).head(max_docs)
            if df.shape[0] >= min_docs:
                return df['idx_doc'].tolist(), df['prob'].tolist()
            return [], []

        def calc_corpus_embedding(text_embeddings, text_probs):
            # calculate corpus-contextualized document embeddings
            # text_probs: the probabilities of a doc associated with the topic
            if len(text_embeddings) != 0:
                text_probs = np.array(text_probs)
                text_probs /= text_probs.mean()
                text_probs = text_probs.reshape(-1, 1)
                return (text_probs * text_embeddings).mean(axis=0)
            else:
                return np.zeros(768)

        topics = pickle.load(open(os.path.join(self.data_path, 'topics.pkl'), 'rb'))
        topic_stems = [[each[0] for each in each1[1]] for each1 in topics]

        if polarization in ['emb', 'emb_pairwise']:
            doc_embeddings = self.calc_embeddings(True)
        elif polarization == 'emb_doc':
            doc_embeddings = self.calc_embeddings()
        else:
            doc_embeddings = None, None

        df_doc_topic = pd.read_csv(os.path.join(self.data_path, 'df_doc_topic.csv'))
        topic_ranks = pickle.load(open(os.path.join(self.data_path, 'topic_ranks.pkl'), 'rb'))
        topic_idxes = [each[0] for each in topic_ranks]

        if polarization in ['emb', 'emb_pairwise', 'emb_doc']:
            corpus, id2word = None, None
        else:
            corpus, id2word = pickle.load(open(os.path.join(self.data_path, 'corpus_lo.pkl'), 'rb'))

        data = []
        from sklearn.metrics.pairwise import cosine_similarity
        for topic_idx in topic_idxes:
            print(f"{'*' * 10}Topic: {topic_idx}{'*' * 10}")
            row = [','.join(topic_stems[topic_idx]), f'topic_{topic_idx}']

            if hasattr(self, 'labels'):
                idxes_docs1, text_probs1 = select_docs(df_doc_topic, topic_idx,
                                                        max_docs, min_docs, label=1)
                idxes_docs2, text_probs2 = select_docs(df_doc_topic, topic_idx,
                                                        max_docs, min_docs, label=0)
                min_len = min(len(idxes_docs1), len(idxes_docs2))
                idxes_docs1_ = idxes_docs1[:min_len]
                idxes_docs2_ = idxes_docs2[:min_len]
                text_probs1 = np.array(text_probs1[:min_len])
                text_probs2 = np.array(text_probs2[:min_len])
                text_probs1 /= text_probs1.mean()
                text_probs2 /= text_probs2.mean()

                if polarization in ['emb', 'emb_pairwise', 'emb_doc']:

                    if polarization in ['emb', 'emb_doc']:
                        emb1 = calc_corpus_embedding(doc_embeddings[idxes_docs1_], text_probs1)
                        emb2 = calc_corpus_embedding(doc_embeddings[idxes_docs2_], text_probs2)
                        cos_sim = cosine_similarity([emb1], [emb2])[0][0]
                    else:
                        embs1_ = doc_embeddings[idxes_docs1_]
                        embs2_ = doc_embeddings[idxes_docs2_]
                        if embs1_.sum() != 0 and embs2_.sum() != 0:
                            pairwise_cossim = cosine_similarity(embs1_, embs2_)
                            weight_mat = np.matmul(np.array(text_probs1).reshape(-1, 1),
                                                    np.array(text_probs2).reshape(1, -1))
                            # weight_mat = np.ones((min_len, min_len))
                            weight_mat = weight_mat / weight_mat.mean()
                            cos_sim = (pairwise_cossim * weight_mat).mean()
                        else:
                            cos_sim = float('nan')
                    pola_score = 0.5 * (-cos_sim + 1)

                elif polarization == 'lo':
                    corpus1 = pd.Series(corpus)[idxes_docs1]
                    corpus2 = pd.Series(corpus)[idxes_docs2]
                    pola_score, pol_score_random, n_articles = get_leaveout_score(corpus1, corpus2, id2word,
                                                                                    min_docs=min_docs,
                                                                                    max_docs=max_docs)
                    # pola_score = 1 - 2 * pola_score
                else:  # ground true
                    raise ValueError('Invalid polarization method')
            else:
                idxes_docs, text_probs = select_docs(df_doc_topic, topic_idx,
                                                        max_docs, min_docs)
                if len(idxes_docs) < min_docs:
                    continue
                min_len = len(idxes_docs)
                text_probs = np.array(text_probs)
                text_probs /= text_probs.mean()

                if polarization in ['emb', 'emb_pairwise', 'emb_doc']:

                    if polarization in ['emb', 'emb_doc']:
                        raise ValueError('Invalid polarization method')
                    else:
                        embs = doc_embeddings[idxes_docs]
                        if embs.sum() != 0:
                            pairwise_cossim = cosine_similarity(embs, embs)
                            weight_mat = np.matmul(np.array(text_probs).reshape(-1, 1),
                                                    np.array(text_probs).reshape(1, -1))
                            # weight_mat = np.ones((min_len, min_len))
                            weight_mat = weight_mat / weight_mat.mean()
                            cos_sim = (pairwise_cossim * weight_mat).mean()
                        else:
                            cos_sim = float('nan')
                        pola_score = 0.5 * (-cos_sim + 1)

                elif polarization == 'lo':
                    corpus = pd.Series(corpus)[idxes_docs]
                    pola_score, pol_score_random, n_articles = get_leaveout_score(corpus, corpus, id2word,
                                                                                    min_docs=min_docs,
                                                                                    max_docs=max_docs)
                    # pola_score = 1 - 2 * pola_score
                else:  # ground true
                    raise ValueError('Invalid polarization method')

            # pola_score = float('nan') if pola_score == 0 else pola_score
            row.append(pola_score)

            data.append([row[1], row[2], row[0]])

        os.makedirs(os.path.join(self.data_path, 'results'), exist_ok=True)
        file_name = f"all{'_unfinetuned' if self.unfinetuned else ''}_{polarization}" \
                    f"_{max_docs}_{min_docs}.csv"
        if self.shuffle:
            file_name = file_name[:-5] + '_shuffle.csv'
        data.sort(key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(data, columns=['topic_idx', 'pola'] + ['topic_words'])
        return df

def get_news_token_counts(corpus, idx2word):
    row_idx = []
    col_idx = []
    data = []
    for i, doc in enumerate(corpus):
        for j, count in doc:
            row_idx.append(i)
            col_idx.append(j)
            data.append(count)
    return sp.csr_matrix((data, (row_idx, col_idx)), shape=(len(corpus), len(idx2word)))



def get_party_q(party_counts, exclude_user_id = None):
    user_sum = party_counts.sum(axis=0)
    if exclude_user_id:
        user_sum -= party_counts[exclude_user_id, :]
    total_sum = user_sum.sum()
    return user_sum / total_sum


def get_rho(dem_q, rep_q):
    return (rep_q / (dem_q + rep_q)).transpose()


def get_token_user_counts(party_counts):
    no_tokens = party_counts.shape[1]
    nonzero = sp.find(party_counts)[:2]
    user_t_counts = collections.Counter(nonzero[1])  # number of users using each term
    party_t = np.ones(no_tokens)  # add one smoothing
    for k, v in user_t_counts.items():
        party_t[k] += v
    return party_t


def mutual_information(dem_t, rep_t, dem_not_t, rep_not_t, dem_no, rep_no):
    no_users = dem_no + rep_no
    all_t = dem_t + rep_t
    all_not_t = no_users - all_t + 4
    mi_dem_t = dem_t * np.log2(no_users * (dem_t / (all_t * dem_no)))
    mi_dem_not_t = dem_not_t * np.log2(no_users * (dem_not_t / (all_not_t * dem_no)))
    mi_rep_t = rep_t * np.log2(no_users * (rep_t / (all_t * rep_no)))
    mi_rep_not_t = rep_not_t * np.log2(no_users * (rep_not_t / (all_not_t * rep_no)))
    return (1 / no_users * (mi_dem_t + mi_dem_not_t + mi_rep_t + mi_rep_not_t)).transpose()[:, np.newaxis]


def chi_square(dem_t, rep_t, dem_not_t, rep_not_t, dem_no, rep_no):
    no_users = dem_no + rep_no
    all_t = dem_t + rep_t
    all_not_t = no_users - all_t + 4
    chi_enum = no_users * (dem_t * rep_not_t - dem_not_t * rep_t) ** 2
    chi_denom = all_t * all_not_t * (dem_t + dem_not_t) * (rep_t + rep_not_t)
    return (chi_enum / chi_denom).transpose()[:, np.newaxis]


def calculate_polarization(dem_counts, rep_counts, measure="posterior", leaveout=True):
    dem_user_total = dem_counts.sum(axis=1)
    rep_user_total = rep_counts.sum(axis=1)

    dem_user_distr = (sp.diags(1 / dem_user_total.A.ravel())).dot(dem_counts)  # get row-wise distributions
    rep_user_distr = (sp.diags(1 / rep_user_total.A.ravel())).dot(rep_counts)
    dem_no = dem_counts.shape[0]
    rep_no = rep_counts.shape[0]
    assert (set(dem_user_total.nonzero()[0]) == set(range(dem_no)))  # make sure there are no zero rows
    assert (set(rep_user_total.nonzero()[0]) == set(range(rep_no)))  # make sure there are no zero rows
    if measure not in ('posterior', 'mutual_information', 'chi_square'):
        print('invalid method')
        return
    dem_q = get_party_q(dem_counts)
    rep_q = get_party_q(rep_counts)
    dem_t = get_token_user_counts(dem_counts)
    rep_t = get_token_user_counts(rep_counts)
    dem_not_t = dem_no - dem_t + 2  # because of add one smoothing
    rep_not_t = rep_no - rep_t + 2  # because of add one smoothing
    func = mutual_information if measure == 'mutual_information' else chi_square

    # apply measure without leave-out
    if not leaveout:
        if measure == 'posterior':
            token_scores_rep = get_rho(dem_q, rep_q)
            token_scores_dem = 1. - token_scores_rep
        else:
            token_scores_dem = func(dem_t, rep_t, dem_not_t, rep_not_t, dem_no, rep_no)
            token_scores_rep = token_scores_dem
        dem_val = 1 / dem_no * dem_user_distr.dot(token_scores_dem).sum()
        rep_val = 1 / rep_no * rep_user_distr.dot(token_scores_rep).sum()
        return 1/2 * (dem_val + rep_val)

    # apply measures via leave-out
    dem_addup = 0
    rep_addup = 0
    dem_leaveout_no = dem_no - 1
    rep_leaveout_no = rep_no - 1
    for i in range(dem_no):
        if measure == 'posterior':
            dem_leaveout_q = get_party_q(dem_counts, i)
            token_scores_dem = 1. - get_rho(dem_leaveout_q, rep_q)
        else:
            dem_leaveout_t = dem_t.copy()
            excl_user_terms = sp.find(dem_counts[i, :])[1]
            for term_idx in excl_user_terms:
                dem_leaveout_t[term_idx] -= 1
            dem_leaveout_not_t = dem_leaveout_no - dem_leaveout_t + 2
            token_scores_dem = func(dem_leaveout_t, rep_t, dem_leaveout_not_t, rep_not_t, dem_leaveout_no, rep_no)
        dem_addup += dem_user_distr[i, :].dot(token_scores_dem)[0, 0]
    for i in range(rep_no):
        if measure == 'posterior':
            rep_leaveout_q = get_party_q(rep_counts, i)
            token_scores_rep = get_rho(dem_q, rep_leaveout_q)
        else:
            rep_leaveout_t = rep_t.copy()
            excl_user_terms = sp.find(rep_counts[i, :])[1]
            for term_idx in excl_user_terms:
                rep_leaveout_t[term_idx] -= 1
            rep_leaveout_not_t = rep_leaveout_no - rep_leaveout_t + 2
            token_scores_rep = func(dem_t, rep_leaveout_t, dem_not_t, rep_leaveout_not_t, dem_no, rep_leaveout_no)
        rep_addup += rep_user_distr[i, :].dot(token_scores_rep)[0, 0]
    rep_val = 1 / rep_no * rep_addup
    dem_val = 1 / dem_no * dem_addup
    return 1/2 * (dem_val + rep_val)


def get_leaveout_score(corpus1, corpus2, id2word, token_partisanship_measure='posterior',
                       leaveout=True, default_score=0.5, min_docs=10, max_docs=999):
    """
    Measure polarization.
    :param event: name of the event
    :param data: dataframe with 'text' and 'user_id'
    :param token_partisanship_measure: type of measure for calculating token partisanship based on user-token counts
    :param leaveout: whether to use leave-out estimation
    :param between_topic: whether the estimate is between topics or tokens
    :param default_score: default token partisanship score
    :return:
    """
    import gc

    dem_counts = get_news_token_counts(corpus1, id2word)
    rep_counts = get_news_token_counts(corpus2, id2word)

    dem_user_len = dem_counts.shape[0]
    rep_user_len = rep_counts.shape[0]

    # return these values when there is not enough data to make predictions on
    if dem_user_len < min_docs or rep_user_len < min_docs:
        return default_score, default_score, dem_user_len + rep_user_len

    if max_docs < dem_user_len:
        dem_counts = dem_counts[:dem_user_len]
    if max_docs < rep_user_len:
        rep_counts = rep_counts[:rep_user_len]

    import random
    RNG = random.Random()  # make everything reproducible
    RNG.seed(42)
    # make the prior neutral (i.e. make sure there are the same number of Rep and Dem users)
    dem_user_len = dem_counts.shape[0]
    rep_user_len = rep_counts.shape[0]
    if dem_user_len > rep_user_len:
        dem_subset = np.array(RNG.sample(range(dem_user_len), rep_user_len))
        dem_counts = dem_counts[dem_subset, :]
        dem_user_len = dem_counts.shape[0]
    elif rep_user_len > dem_user_len:
        rep_subset = np.array(RNG.sample(range(rep_user_len), dem_user_len))
        rep_counts = rep_counts[rep_subset, :]
        rep_user_len = rep_counts.shape[0]
    assert (dem_user_len == rep_user_len)

    all_counts = sp.vstack([dem_counts, rep_counts])

    wordcounts = all_counts.nonzero()[1]

    # filter words used by fewer than 2 people
    all_counts = all_counts[:, np.array([(np.count_nonzero(wordcounts == i) > 1) for i in range(all_counts.shape[1])])]

    dem_counts = all_counts[:dem_user_len, :]
    rep_counts = all_counts[dem_user_len:, :]
    del wordcounts
    del all_counts
    gc.collect()

    dem_nonzero = set(dem_counts.nonzero()[0])
    rep_nonzero = set(rep_counts.nonzero()[0])
    # filter users who did not use words from vocab
    dem_counts = dem_counts[np.array([(i in dem_nonzero) for i in range(dem_counts.shape[0])]), :]
    rep_counts = rep_counts[np.array([(i in rep_nonzero) for i in range(rep_counts.shape[0])]), :]
    del dem_nonzero
    del rep_nonzero
    gc.collect()

    actual_val = calculate_polarization(dem_counts, rep_counts, token_partisanship_measure, leaveout)

    all_counts = sp.vstack([dem_counts, rep_counts])
    del dem_counts
    del rep_counts
    gc.collect()

    index = np.arange(all_counts.shape[0])
    RNG.shuffle(index)
    all_counts = all_counts[index, :]

    random_val = calculate_polarization(all_counts[:dem_user_len, :], all_counts[dem_user_len:, :],
                                        token_partisanship_measure, leaveout)
    print(actual_val, random_val, dem_user_len + rep_user_len)
    sys.stdout.flush()
    del all_counts
    gc.collect()

    return actual_val, random_val, dem_user_len + rep_user_len



def get_leaveout_emb_score(dem_counts, rep_counts, token_partisanship_measure='posterior',
                       leaveout=True, default_score=0.5, min_docs=10, max_docs=999):
    """
    Measure polarization.
    :param event: name of the event
    :param data: dataframe with 'text' and 'user_id'
    :param token_partisanship_measure: type of measure for calculating token partisanship based on user-token counts
    :param leaveout: whether to use leave-out estimation
    :param between_topic: whether the estimate is between topics or tokens
    :param default_score: default token partisanship score
    :return:
    """
    if dem_counts.sum() == 0 or rep_counts.sum() == 0:
        return 0.5, 0, 0
    from sklearn.metrics.pairwise import cosine_similarity
    sim_mat = cosine_similarity(dem_counts, rep_counts)
    # import ipdb; ipdb.set_trace()
    # print(sim_mat)
    return sim_mat.mean(), 0, 0


class PaCTE:
    def __init__(self):
        pass

    def train(
            self,
            docs, 
            labels,
            seed=42, 
            lr=1e-5, 
            batch_size=24, 
            epochs=50,
            unfinetuned=False,
            shuffle=False
        ):
        self._run(
            docs,
            labels=labels,
            train=True,
            seed=seed,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            unfinetuned=unfinetuned,
            shuffle=shuffle,
            polarization=None,
            plotting=False
        )

    def fit_transform(
            self,
            docs, 
            seed=42, 
            model_path=None,
            polarization='emb', 
            n_topics=10, 
            min_docs=10, 
            max_docs=10, 
            dim_reduction='tsne',
            plotting=False
        ):
        return self._run(
            docs,
            train=False,
            seed=seed,
            model_path=model_path,
            polarization=polarization,
            n_topics=n_topics,
            min_docs=min_docs,
            max_docs=max_docs,
            dim_reduction=dim_reduction,
            plotting=plotting
        )

    def _run(
            self,
            docs, 
            labels=None,
            train=False,
            seed=42, 
            lr=1e-5, 
            batch_size=24, 
            epochs=50,
            unfinetuned=False,
            shuffle=False,
            model_path=None,
            polarization='emb', 
            n_topics=10, 
            min_docs=10, 
            max_docs=10, 
            dim_reduction='tsne',
            plotting=False
        ):
        data = docs

        hashable_docs = tuple(sorted(docs))
        dataset_hash = hashlib.md5(str(hashable_docs).encode()).hexdigest()
        data_path = f'./data/pacte/{dataset_hash}'
        os.makedirs(data_path, exist_ok=True)
        # https://github.com/zihaohe123/pacte-polarized-topics-detection
        texts_processed_lda, corpus_lda, id2word_lda = preprocessing_lda(data)
        pickle.dump(texts_processed_lda, open(os.path.join(data_path, 'texts_processed_lda.pkl'), 'wb'))
        pickle.dump((corpus_lda, id2word_lda), open(os.path.join(data_path, 'corpus_lda.pkl'), 'wb'))


        text_processed_bert = preprocessing_bert(data)
        text_processed_bert = [[str(x) for x in y] for y in text_processed_bert]
        pickle.dump(text_processed_bert, open(os.path.join(data_path, 'texts_processed_bert.pkl'), 'wb'))

        df_news = pd.DataFrame(data, columns=['text'])
        df_news.insert(0, 'idx', np.arange(df_news.shape[0]))
        if labels:
            df_news['label'] = labels
        df_news.to_csv(os.path.join(data_path, 'df_news.csv'), index=False)

        topic_modelling(data_path, df_news, train=train)

        assert dim_reduction in ['pca', 'tsne', 'umap']
        assert polarization in [None, 'emb', 'lo', 'emb_pairwise', 'gt', 'emb_doc']
        # parser.add_argument('--source1', nargs='+', default=['cnn', 'huff', 'nyt'], help='the left sources')
        # parser.add_argument('--source2', nargs='+', default=['fox', 'breit', 'nyp'], help='the right sources')

        if polarization in ['lo', 'gt']:
            unfinetuned = False
            init_train = False
            plotting = False

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        engine = Engine(
            data_path, 
            model_path=model_path,
            init_train=train, 
            lr=lr, 
            batch_size=batch_size,
            epochs=epochs,
            shuffle=shuffle,
            unfinetuned=unfinetuned
        )
        if train:
            engine.train()
            print(f"Saved model to {engine.model_path}")
        if polarization:
            target_df = engine.get_polarization(polarization=polarization, min_docs=min_docs, max_docs=max_docs)
        if plotting:
            engine.plot_embeddings(dim_reduction='pca')
            engine.plot_embeddings(dim_reduction='tsne')
            engine.plot_embeddings(topic_emb=True, dim_reduction='pca')
            engine.plot_embeddings(topic_emb=True, dim_reduction='tsne')

        if polarization:
            self.target_info = target_df

            predictions = pickle.load(open(os.path.join(engine.data_path, 'embeddings', f'predictions.pkl'), 'rb'))

            df = pd.read_csv(os.path.join(engine.data_path, 'df_doc_topic.csv'))
            topic_docs_df = df.groupby('idx_doc').agg({'idx_topic': list, 'prob': list}).reset_index()
            topic_docs_df = topic_docs_df.rename(columns={'idx_topic': 'idx_topics', 'prob': 'probs'})
            docs_df = pd.DataFrame({'text': docs, 'idx_doc': np.arange(len(docs))})
            docs_df = docs_df.merge(topic_docs_df, on='idx_doc', how='left')

            # fill docs with no assigned topics
            docs_df['idx_topics'] = docs_df['idx_topics'].apply(lambda x: x if isinstance(x, list) else [])
            docs_df['probs'] = docs_df['probs'].apply(lambda x: x if isinstance(x, list) else [])
            assert docs_df.shape[0] == len(data)
            assert len(predictions) == len(data)
            docs_df['pred'] = predictions
            doc_targets = []
            probs = np.zeros((len(docs), len(self.target_info)))
            polarity = np.zeros((len(docs), len(self.target_info)))
            for i, row in docs_df.iterrows():
                topic_nums = row['idx_topics']
                topic_probs = row['probs']

                if len(topic_nums) == 0:
                    doc_targets.append([])
                    continue

                pred = row['pred']
                # get highest probablity topic
                topic_idx = sorted(zip(topic_nums, topic_probs), key=lambda x: x[1], reverse=True)[0][0]
                top_target_row = target_df[target_df['topic_idx'] == f'topic_{topic_idx}']
                targets = top_target_row['topic_words'].values
                if len(targets) == 0:
                    doc_targets.append([])
                else:
                    doc_targets.append(targets[0])
                for topic_num, topic_prob in zip(topic_nums, topic_probs):
                    target_row = target_df[target_df['topic_idx'] == f'topic_{topic_num}']
                    if len(target_row) == 0:
                        continue
                    target_idx = target_row.index[0]
                    probs[row['idx_doc'], target_idx] = topic_prob
                    polarity[row['idx_doc'], target_idx] = pred
            return doc_targets, probs, polarity
        
    def get_target_info(self):
        return self.target_info.rename(columns={'topic_words': 'noun_phrase'})


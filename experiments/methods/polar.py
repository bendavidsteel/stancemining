import datetime
import hashlib
import json
import os
import pickle

import numpy as np
import pandas as pd
import requests
import spacy

from polar.attitude.syntactical_sentiment_attitude import SyntacticalSentimentAttitudePipeline
from polar.news_corpus_collector import NewsCorpusCollector
from polar.actor_extractor import EntityExtractor, NounPhraseExtractor
from polar.topic_identifier import TopicIdentifier
from polar.coalitions_and_conflicts import FellowshipExtractor, InsufficientSignedEdgesException, DipoleGenerator, TopicAttitudeCalculator
from polar.sag_generator import SAGGenerator

class Polar:
    def __init__(self):
        pass

    def fit_transform(self, docs):
        # https://github.com/dpasch01/polarlib
        hashable_docs = tuple(docs)
        dataset_hash = hashlib.md5(str(hashable_docs).encode()).hexdigest()
        output_dir = f"./data/polar/{dataset_hash}/"
        os.makedirs(output_dir, exist_ok=True)
        nlp = spacy.load("en_core_web_sm")

        # News Corpus Collection
        corpus_collector = NewsCorpusCollector(output_dir=output_dir, from_date=datetime.date.today(), to_date=datetime.date.today(), keywords=[])

        article_dir = os.path.join(output_dir, 'articles', '0')
        os.makedirs(article_dir, exist_ok=True)
        for idx, doc in enumerate(docs):
            uid_name = f"doc_{idx}.json"
            with open(os.path.join(article_dir, uid_name), 'w') as f:
                json.dump({'text': doc, 'uid': str(idx)}, f)
        corpus_collector.pre_process_articles()

        # Entity and NP Extraction
        entity_extractor = EntityExtractor(output_dir=output_dir)

        # check that spotlight is accessible
        text = "This is a test"
        confidence = 0.5
        spotlight_url='http://127.0.0.1:2222/rest/annotate'
        req_data      = {'lang': 'en', 'text': str(text), 'confidence': confidence, 'types': ['']}
        ret = requests.post(spotlight_url, data=req_data, headers={"Accept": "application/json"})
        if ret.status_code != 200:
            raise Exception("Spotlight API is not accessible. Please setup")

        entity_extractor.extract_entities()
        noun_phrase_extractor = NounPhraseExtractor(output_dir=output_dir)
        noun_phrase_extractor.extract_noun_phrases()

        # Discussion Topic Identification
        topic_identifier = TopicIdentifier(output_dir=output_dir)
        topic_identifier.encode_noun_phrases()
        topic_identifier.noun_phrase_clustering(threshold=0.6)

        # Sentiment Attitude Classification
        mpqa_path = "./models/polar/subjclueslen1-HLTEMNLP05.tff"
        sentiment_attitude_pipeline = SyntacticalSentimentAttitudePipeline(output_dir=output_dir, nlp=nlp, mpqa_path=mpqa_path)
        sentiment_attitude_pipeline.calculate_sentiment_attitudes()

        # Sentiment Attitude Graph Construction
        sag_generator = SAGGenerator(output_dir)
        sag_generator.load_sentiment_attitudes()
        bins = sag_generator.calculate_attitude_buckets(verbose=True)
        sag_generator.convert_attitude_signs(
            bin_category_mapping={
                "NEGATIVE": [bins[0], bins[1], bins[2], bins[3]],
                "NEUTRAL":  [bins[4], bins[5]],
                "POSITIVE": [bins[6], bins[7], bins[8], bins[9]]
            },
            minimum_frequency=3,
            verbose=True
        )
        sag_generator.construct_sag()

        # Entity Fellowship Extraction
        try:
            os.makedirs('./data/polar/simap/', exist_ok=True)
            fellowship_extractor = FellowshipExtractor(output_dir)
            fellowships = fellowship_extractor.extract_fellowships(
                n_iter     = 1,
                resolution = 0.05,
                merge_iter = 1,
                jar_path   ='../signed-community-detection/target/',
                jar_name   ='signed-community-detection-1.1.4.jar',
                tmp_path   ='./data/polar/simap/',
                verbose    = True
            )   

            # Fellowship Dipole Generation
            dipole_generator = DipoleGenerator(output_dir)
            dipoles = dipole_generator.generate_dipoles(f_g_thr=0.7, n_r_thr=0.5)

            # Dipole Topic Polarization
            topic_attitude_calculator = TopicAttitudeCalculator(output_dir)
            topic_attitude_calculator.load_sentiment_attitudes()
            topic_attitudes = topic_attitude_calculator.get_topic_attitudes()

            self.ngrams = [t['topic']['nps'][0] for t in topic_attitudes]

            doc_targets = []
            probs = np.zeros((len(docs), len(self.ngrams)))
            polarity = np.zeros((len(docs), len(self.ngrams)))

            assert len(sag_generator.attitude_path_list) == len(docs)
            for doc_idx, (doc, file_path) in enumerate(zip(docs, sorted(sag_generator.attitude_path_list))):
                with open(file_path, 'rb') as f:
                    d = pickle.load(f)
                
                # check if entities or nouns are in any of the dipoles
                # and use this to tag the document with a target, stance, and topic
                # if not, tag with None

                if len(d['attitudes']) == 0:
                    doc_targets.append(None)
                    continue

                for attitude in d['attitudes']:
                    for entity in attitude['entities']:
                        for dipole in dipoles:
                            if entity['title'] in list(dipole[1]['d_ij'].nodes()):
                                # set probs of doc belonging to dipole target
                                # if document features an entity from a dipole, it seems likely that the document is relevant to the topics discussed in the dipole
                                # but polarity is neutral
                                doc_topic_idxs = [i for i, t in enumerate(topic_attitudes) if t['dipole'] == dipole[0]]
                                if len(doc_topic_idxs) > 0:
                                    for doc_topic_idx in doc_topic_idxs:
                                        probs[doc_idx, doc_topic_idx] += 1


                    for noun_phrase in attitude['noun_phrases']:
                        for topic_idx, topic in enumerate(topic_attitudes):
                            if noun_phrase['ngram'] in topic['topic']['nps']:
                                # set probs of doc belonging to noun phrase topic
                                # if the document features a noun phrase that is considered polarized in a dipole, it seems likely that the document is relevant to the topics discussed in the dipole
                                # but polarity is neutral
                                probs[doc_idx, topic_idx] += 1

                topic_vals = probs[doc_idx, :]
                num_topics = (topic_vals > 0).sum()
                if num_topics == 0:
                    doc_targets.append(None)
                elif num_topics == 1:
                    topic_idx = np.argmax(topic_vals)
                    doc_targets.append(self.ngrams[topic_idx])
                else:
                    # check if two topic probs are the same
                    num_max = (topic_vals == np.max(topic_vals)).sum()
                    if num_max == 1:
                        topic_idx = np.argmax(topic_vals)
                        doc_targets.append(self.ngrams[topic_idx])
                    else:
                        # return all that are max
                        topic_idxs = np.where(topic_vals == np.max(topic_vals))[0]
                        doc_targets.append([self.ngrams[topic_idx] for topic_idx in topic_idxs])

        except InsufficientSignedEdgesException:
            fellowships = []
            dipoles = []
            topic_attitudes = []

            self.ngrams = []

            doc_targets = [None] * len(docs)
            probs = np.zeros((len(docs), 0))
            polarity = np.zeros((len(docs), 0))

        
        # normalize probs
        probs[doc_idx, :] = probs[doc_idx, :] / np.maximum(np.sum(probs[doc_idx, :]), 1)


        return doc_targets, probs, polarity
    
    def get_target_info(self):
        return pd.DataFrame(self.ngrams, columns=['ngram'])
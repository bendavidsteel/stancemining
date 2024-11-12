import datetime
import json
import os

import numpy as np
import pandas as pd
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
        output_dir = "./data/polar"
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

            self.ngrams = topic_attitudes
            doc_targets = topic_attitudes
            probs = np.zeros((len(docs), len(topic_attitudes)))
            polarity = np.zeros((len(docs), len(topic_attitudes)))

        except InsufficientSignedEdgesException:
            fellowships = []
            dipoles = []
            topic_attitudes = []

            self.ngrams = []

            doc_targets = [None] * len(docs)
            probs = np.zeros((len(docs), 0))
            polarity = np.zeros((len(docs), 0))

        return doc_targets, probs, polarity
    
    def get_target_info(self):
        return pd.DataFrame(self.ngrams, columns=['ngram'])
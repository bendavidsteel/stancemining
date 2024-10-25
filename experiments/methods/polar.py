import spacy

from polar.attitude.syntactical_sentiment_attitude import SyntacticalSentimentAttitudePipeline
from polar.news_corpus_collector import NewsCorpusCollector
from polar.actor_extractor import EntityExtractor, NounPhraseExtractor
from polar.topic_identifier import TopicIdentifier
from polar.coalitions_and_conflicts import FellowshipExtractor, DipoleGenerator, TopicAttitudeCalculator
from polar.sag_generator import SAGGenerator

def polar(docs):
    # https://github.com/dpasch01/polarlib
    output_dir = "./data/polar"
    nlp = spacy.load("en_core_web_sm")

    # News Corpus Collection
    corpus_collector = NewsCorpusCollector(output_dir=output_dir)

    # Entity and NP Extraction
    entity_extractor = EntityExtractor(output_dir=output_dir)
    noun_phrase_extractor = NounPhraseExtractor(output_dir=output_dir)

    # Discussion Topic Identification
    topic_identifier = TopicIdentifier(output_dir=output_dir)

    # Sentiment Attitude Classification
    sentiment_attitude_pipeline = SyntacticalSentimentAttitudePipeline(output_dir=output_dir, nlp=nlp)

    # Sentiment Attitude Graph Construction
    sag_generator = SAGGenerator(output_dir)
    sag_generator.load_sentiment_attitudes()
    sag_generator.convert_attitude_signs(bin_category_mapping={})

    # Entity Fellowship Extraction
    fellowship_extractor = FellowshipExtractor(output_dir)
    fellowships = fellowship_extractor.extract_fellowships()

    # Fellowship Dipole Generation
    dipole_generator = DipoleGenerator(output_dir)
    dipoles = dipole_generator.generate_dipoles()

    # Dipole Topic Polarization
    topic_attitude_calculator = TopicAttitudeCalculator(output_dir)
    topic_attitude_calculator.load_sentiment_attitudes()
    topic_attitudes = topic_attitude_calculator.get_topic_attitudes()

    return sag_generator, fellowships, dipoles, topic_attitudes
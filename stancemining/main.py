from typing import List

from bertopic import BERTopic
import numpy as np
import pandas as pd
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

from stancemining import llms, finetune, prompting
from stancemining.ngram_gen import NGramGeneration

keyword_prompt = """<|system|>You are a helpful, respectful and honest assistant for labeling topics..</s>
<|user|>
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.</s>
<|assistant|>"""

def get_var_and_max_var_target(documents_df, target_info):
    target_info_df = pl.DataFrame(target_info)
    if 'topic_id' in target_info_df.columns:
        target_info_df = target_info_df.group_by('noun_phrase')\
            .agg(pl.col('topic_id'), pl.col('polarity'))
    else:
        target_info_df = target_info_df.group_by('noun_phrase')\
            .agg(pl.col('polarity'))
    target_info_df = target_info_df.with_columns([
        pl.col('polarity').list.mean().alias('mean'),
        pl.when(pl.col('polarity').list.len() > 1)\
            .then(pl.col('polarity').list.var())\
            .otherwise(0)
            .alias('var')
    ])

    documents_df = documents_df.join(
        documents_df.explode('Targets')\
            .join(target_info_df, left_on='Targets', right_on='noun_phrase', how='left')\
            .group_by('ID')\
            .agg(pl.all().sort_by('var').last())\
            .with_columns(pl.col('Targets').alias('Target'))\
            .select(['ID', 'Target']),
        on='ID',
        how='left',
        maintain_order='left'
    )
    return documents_df, target_info_df


class StanceMining:
    def __init__(self, 
                 method='llm', 
                 llm_method='zero-shot',
                 num_representative_docs=5,
                 model_lib='transformers', 
                 model_name='microsoft/Phi-3.5-mini-instruct', 
                 model_kwargs={}, 
                 tokenizer_kwargs={},
                 finetune_kwargs={}
                ):
        self.method = method
        assert llm_method in ['zero-shot', 'finetuned'], f"LLM method must be either 'zero-shot' or 'finetuned', not '{llm_method}'"
        self.llm_method = llm_method

        self.num_representative_docs = num_representative_docs

        self.model_lib = model_lib
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.finetune_kwargs = finetune_kwargs

        # lazily load generator if using in between fine-tuned models
        self.generator = self._get_generator()

        self.embedding_model = 'paraphrase-MiniLM-L6-v2'

    def _get_embedding_model(self):
        model = SentenceTransformer(self.embedding_model)
        return model

    def _get_embeddings(self, docs, model=None):
        if model is None:
            model = self._get_embedding_model()
        embeddings = model.encode(docs)
        return embeddings

    
    def _ask_llm_stance_target(self, docs: List[str]):
        num_samples = 3
        if self.llm_method == 'zero-shot':
            return prompting.ask_llm_zero_shot_stance_target(self.generator, docs, {'num_samples': num_samples})
        elif self.llm_method == 'finetuned':
            model_name = self.finetune_kwargs['model_name'].replace('/', '-')
            topic_extraction_path = f'./models/wiba/{model_name}-topic-extraction-vast-ezstance'
            df = pl.DataFrame({'Text': docs})
            generate_kwargs = {}
            generate_kwargs['num_beams'] = num_samples * 5
            generate_kwargs['num_return_sequences'] = num_samples
            generate_kwargs['num_beam_groups'] = num_samples
            generate_kwargs['diversity_penalty'] = 0.5
            generate_kwargs['no_repeat_ngram_size'] = 2
            generate_kwargs['do_sample'] = False
            self.generator.unload_model()
            results = finetune.get_predictions("topic-extraction", df, topic_extraction_path, self.finetune_kwargs, generate_kwargs=generate_kwargs)
            self.generator.load_model()
            if isinstance(results[0], str):
                results = [[r] for r in results]
            # lower case all results
            results = [[r.lower() for r in res] for res in results]
            # remove exact duplicates
            results = [list(set(res)) for res in results]
            return results

    def _ask_llm_stance(self, docs, stance_targets):
        if self.llm_method == 'zero-shot':
            return prompting.ask_llm_zero_shot_stance(self.generator, docs, stance_targets)
        elif self.llm_method == 'finetuned':
            model_name = self.finetune_kwargs['model_name'].replace('/', '-')
            stance_classification_path = f'./models/wiba/{model_name}-stance-classification-vast-ezstance'
            data = pl.DataFrame({'Text': docs, 'Target': stance_targets})
            self.generator.unload_model()
            results = finetune.get_predictions("stance-classification", data, stance_classification_path, self.finetune_kwargs)
            self.generator.load_model()
            results = [r.upper() for r in results]
            return results

    def _filter_similar_phrases(self, phrases_list: List[List[str]], embedding_model=None, similarity_threshold: float = 0.8) -> List[List[str]]:
        """
        Filter out similar phrases from a list of lists based on embedding similarity,
        only comparing phrases within the same sublist.
        
        Args:
            phrases_list: List of lists containing phrases to filter
            embedding_fn: Function that takes a list of strings and returns numpy array of embeddings
            similarity_threshold: Threshold above which phrases are considered similar (default: 0.8)
            
        Returns:
            List of lists with similar phrases removed
        """
        # If input is empty or has less than 2 items, return as is
        if not phrases_list or len(sum(phrases_list, [])) < 2:
            return phrases_list
            
        # Flatten list to compute embeddings efficiently
        flat_phrases = sum(phrases_list, [])
        
        # Get embeddings for all phrases at once
        try:
            all_embeddings = self._get_embeddings(flat_phrases, model=embedding_model)
        except Exception as e:
            raise ValueError(f"Error computing embeddings: {str(e)}")
        
        # Keep track of sublist boundaries
        boundaries = [0]
        for sublist in phrases_list:
            boundaries.append(boundaries[-1] + len(sublist))
        
        # Process each sublist separately
        filtered_lists = []
        for i in range(len(phrases_list)):
            start, end = boundaries[i], boundaries[i+1]
            
            # Skip if sublist has less than 2 items
            if end - start < 2:
                filtered_lists.append(phrases_list[i])
                continue
                
            # Get embeddings for current sublist
            embeddings = all_embeddings[start:end]
            
            # Compute cosine similarity matrix for current sublist
            norms = np.linalg.norm(embeddings, axis=1)
            similarity = np.dot(embeddings, embeddings.T) / np.outer(norms, norms)
            
            # Get upper triangular part to avoid duplicate comparisons
            similarity = np.triu(similarity, k=1)
            
            # Find indices of similar phrases within this sublist
            similar_indices = set(int(i) for i in np.where(similarity > similarity_threshold)[0])
            
            # Filter current sublist
            filtered_sublist = [
                phrase for j, phrase in enumerate(phrases_list[i])
                if j not in similar_indices
            ]
            filtered_lists.append(filtered_sublist)
        
        return filtered_lists

    def _get_base_topic_model(self, bertopic_kwargs):
        return BERTopic(**bertopic_kwargs)

    def _topic_llm_fit_transform(self, docs, embeddings=None, bertopic_kwargs={}):
        embedding_model = self._get_embedding_model()
        if embeddings is None:
            embeddings = self._get_embeddings(docs, model=embedding_model)

        topic_model = self._get_base_topic_model(bertopic_kwargs)
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
        documents = pd.DataFrame({"Document": docs, "ID": range(len(docs)), "Topic": topics})
        repr_docs, _, _, _ = topic_model._extract_representative_docs(
            topic_model.c_tf_idf_,
            documents,
            topic_model.topic_representations_,
            nr_samples=500,
            nr_repr_docs=self.num_representative_docs,
        )
        topic_model.representative_docs_ = repr_docs
        topic_info = topic_model.get_topic_info()
        target_info = []

        document_df = pl.from_pandas(documents)
        topic_info_df = pl.from_pandas(topic_info)

        # get stance targets for outliers
        outliers_df = document_df.filter(pl.col('Topic') == -1)
        stance_targets = self._ask_llm_stance_target(outliers_df['Document'])
        stance_targets = self._filter_similar_phrases(stance_targets, embedding_model=embedding_model)
        outliers_df = outliers_df.with_columns(pl.Series(name='Targets', values=stance_targets))

        # get stance targets for non-outliers
        stance_targets = [prompting.ask_llm_multi_doc_targets(self.generator, r['Representative_Docs']) for r in tqdm(topic_info_df[['Representative_Docs', 'Representation']].to_dicts(), desc="Getting topic noun phrases")]
        stance_targets = self._filter_similar_phrases(stance_targets, embedding_model=embedding_model)
        topic_info_df = topic_info_df.with_columns(pl.Series(name='Targets', values=stance_targets))

        # get stance targets for hierarchical topics
        hierarchical_topics_df = pl.from_pandas(topic_model.hierarchical_topics(docs))
        hierarchical_topics_df = hierarchical_topics_df.explode('Topics')\
            .join(topic_info_df.select(['Topic', 'Targets']), left_on='Topics', right_on='Topic', how='left')\
            .group_by(['Parent_ID', 'Parent_Name'])\
            .agg(pl.col('Targets').flatten().alias('Targets'), pl.col('Topics'))
        hierarchical_topics_df = hierarchical_topics_df.with_columns(pl.col('Parent_Name').str.split('_').alias('Representation'))
        stance_targets = [prompting.ask_llm_target_aggregate(self.generator, r['Targets'], r['Representation']) for r in tqdm(hierarchical_topics_df.to_dicts(), desc="Getting hierarchical topic noun phrases")]
        stance_targets = self._filter_similar_phrases(stance_targets, embedding_model=embedding_model)
        hierarchical_topics_df = hierarchical_topics_df.with_columns(pl.Series(name='Parent_Targets', values=stance_targets))

        non_outliers_df = document_df.filter(pl.col('Topic') != -1)
        non_outliers_df = non_outliers_df.join(topic_info_df, on='Topic', how='left')

        # add in parent targets
        non_outliers_df = non_outliers_df.join(hierarchical_topics_df.explode('Topics').select(['Topics', 'Parent_Targets']), left_on='Topic', right_on='Topics', how='left')\
            .group_by(non_outliers_df.columns)\
            .agg(pl.col('Parent_Targets').flatten().alias('Parent_Targets'))\
            .with_columns(pl.col('Parent_Targets').fill_null([]))\
            .with_columns(pl.concat_list(pl.col('Targets'), pl.col('Parent_Targets')).alias('Targets'))\
            .drop('Parent_Targets')
        
        # drop similar phrases between base targets and parent targets
        targets = non_outliers_df['Targets'].to_list()
        targets = self._filter_similar_phrases(targets, embedding_model=embedding_model)
        non_outliers_df = non_outliers_df.with_columns(pl.Series(name='Targets', values=targets))

        document_df = document_df.join(
            pl.concat([outliers_df, non_outliers_df], how='diagonal_relaxed'),
            on='ID',
            how='left',
            maintain_order='left'
        )

        # get stances for all targets
        targets_df = document_df.explode('Targets')
        targets_df = targets_df.with_columns(
            pl.Series(name='Stance', values=self._ask_llm_stance(targets_df['Document'], targets_df['Targets']))
        )
        targets_df = targets_df.with_columns(pl.col('Stance').replace_strict({'FAVOR': 1, 'AGAINST': -1, 'NEUTRAL': 0}).alias('Polarity'))
        target_info.extend(targets_df[['Targets', 'Polarity']].rename({'Targets': 'noun_phrase', 'Polarity': 'polarity'}).to_dicts())

        # join targets back to documents
        document_df = document_df.join(
            targets_df.group_by('ID')\
                .agg(pl.col('Targets'), pl.col('Polarity'))\
                .rename({'Targets': 'Targets', 'Polarity': 'Polarities'}),
            on='ID',
            how='left',
            maintain_order='left'
        )

        document_df, target_info_df = get_var_and_max_var_target(document_df, target_info)
        self.target_info = target_info_df

        return document_df
    
    def _get_base_targets(self, docs, embedding_model):
        documents_df = pl.DataFrame({"Document": docs, "ID": range(len(docs))})

        stance_targets = self._ask_llm_stance_target(documents_df['Document'])
        stance_targets = self._filter_similar_phrases(stance_targets, embedding_model=embedding_model)
        documents_df = documents_df.with_columns(pl.Series(name='Targets', values=stance_targets, dtype=pl.List(pl.String)))
        return documents_df

    def _llm_topic_fit_transform(self, docs, bertopic_kwargs={}):
        embedding_model = self._get_embedding_model()
        documents_df = self._get_base_targets(docs, embedding_model)

        # cluster initial stance targets
        topic_model = self._get_base_topic_model(bertopic_kwargs)
        base_targets_df = documents_df.explode('Targets').unique('Targets').drop_nulls()[['Targets']]
        topics, probs = topic_model.fit_transform(base_targets_df['Targets'].to_list())
        base_targets_df = base_targets_df.with_columns([
            pl.Series(name='Topic', values=topics),
            pl.col('Targets').alias('Document')
        ]).with_row_index(name='ID')
        repr_docs, _, _, _ = topic_model._extract_representative_docs(
            topic_model.c_tf_idf_,
            base_targets_df.to_pandas(),
            topic_model.topic_representations_,
            nr_samples=500,
            nr_repr_docs=self.num_representative_docs,
        )
        topic_model.representative_docs_ = repr_docs
        topic_info = topic_model.get_topic_info()

        # get higher level stance targets for each topic
        topic_info = pl.from_pandas(topic_info)
        no_outliers_df = topic_info.filter(pl.col('Topic') != -1)
        stance_targets = [prompting.ask_llm_target_aggregate(self.generator, r['Representative_Docs'], r['Representation']) for r in tqdm(no_outliers_df[['Representative_Docs', 'Representation']].to_dicts(), desc="Getting topic noun phrases")]
        stance_targets = self._filter_similar_phrases(stance_targets, embedding_model=embedding_model)
        no_outliers_df = no_outliers_df.with_columns(
            pl.Series(name='noun_phrases', values=stance_targets, dtype=pl.List(pl.String))
        )
        topic_info = pl.concat([topic_info.filter(pl.col('Topic') == -1), no_outliers_df], how='diagonal_relaxed')\
            .with_columns(pl.col('noun_phrases').fill_null([]))

        # map documents to new noun phrases via topics
        target_df = documents_df.explode('Targets')
        target_df = target_df.join(base_targets_df, on='Targets', how='left')
        target_df = target_df.filter(pl.col('Topic') != -1)
        target_df = target_df.join(topic_info, on='Topic', how='left')

        documents_df = documents_df.join(
                target_df.group_by('ID')\
                    .agg(pl.col('noun_phrases').flatten())\
                    .rename({'noun_phrases': 'NewTargets'}), 
            on='ID', 
            how='left',
            maintain_order='left'
        )
        
        documents_df = documents_df.with_columns(
            pl.when(pl.col('NewTargets').is_not_null())\
                .then(pl.concat_list(pl.col('Targets'), pl.col('NewTargets')))\
                .otherwise(pl.col('Targets')))
        documents_df = documents_df.drop(['NewTargets'])

        # remove targets that are too similar
        targets = documents_df['Targets'].to_list()
        targets = self._filter_similar_phrases(targets, embedding_model=embedding_model)
        documents_df = documents_df.drop('Targets').with_columns(pl.Series(name='Targets', values=targets))

        target_df = documents_df.explode('Targets').rename({'Targets': 'Target'})
        target_df = target_df.with_columns(pl.Series(name='stance', values=self._ask_llm_stance(target_df['Document'], target_df['Target'])))
        target_df = target_df.with_columns(pl.col('stance').replace_strict({'FAVOR': 1, 'AGAINST': -1, 'NEUTRAL': 0}).alias('polarity'))
        target_info = target_df.rename({'Target': 'noun_phrase'}).select(['noun_phrase', 'polarity']).to_dicts()

        documents_df = documents_df.join(
            target_df.group_by('ID')\
                .agg(pl.col('Target').alias('Targets'), pl.col('polarity').alias('Polarities')),
            on='ID',
            how='left',
            maintain_order='left'
        )

        documents_df, target_info_df = get_var_and_max_var_target(documents_df, target_info)
        self.target_info = target_info_df

        return documents_df
    
    def _embedding_fit_transform():
        raise NotImplementedError("Method 'steeringvectors' not implemented")
        if embeddings is None:
            embeddings = get_embeddings(docs)

        model, tokenizer = get_generator()

        # TODO potentially use other topic modelling allowing doc to have multiple topics
        # or use probs to allow docs to fit into multiple topics
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
        self.topic_info = topic_model.get_topic_info()

        for idx, topic in topic_info.iterrows():
            # get generic vector, check projections of docs along that vector
            # then for multiple points in the space, take the mean difference vector, and use that as a steering vector
            # to produce generations of suggestions for noun_phrases
            noun_phrases = ask_llm(topic['representative_docs'])
            topic_id = topic['id']
            topic_docs = [docs[i] for i in range(len(docs)) if topics[i] == topic_id]
            noun_phrase_probs = []
            means = []
            vars = []
            for noun_phrase in noun_phrases:
                # calculate probability of doc along vector
                # https://arjunbansal.substack.com/p/llms-know-more-than-what-they-say
                # get noun_phrase specific target vector, and check projection of doc activations along that vector
                probs = ask_llm(noun_phrase, topic_docs)
                noun_phrase_probs.append(probs)
                means.append(np.mean(probs))
                vars.append(np.var(probs))

            self.topic_info.at[idx, 'noun_phrase_probs'] = noun_phrase_probs
            self.topic_info.at[idx, 'noun_phrase_means'] = means
            self.topic_info.at[idx, 'noun_phrase_vars'] = vars

        return documents
    
    def _get_targets_probs_polarity(self, documents):
        all_targets = self.target_info['noun_phrase'].to_list()
        target_to_idx = {target: idx for idx, target in enumerate(all_targets)}
        doc_targets = documents['Target'].to_list()
        doc_targets = [[t] if t is not None else [] for t in doc_targets]

        num_targets = len(self.target_info)
        
        polarities = np.full((len(documents), num_targets), np.nan)
        probs = np.zeros((len(documents), num_targets))
        polarities_dict = documents['Polarities'].to_list()
        for i, doc_data in enumerate(documents.to_dicts()):
            doc_num_targets = len(doc_data['Targets'])
            for doc_target, doc_polarity in zip(doc_data['Targets'], doc_data['Polarities']):
                idx = target_to_idx[doc_target]
                polarities[i, idx] = doc_polarity
                probs[i, idx] = 1/doc_num_targets

        return doc_targets, probs, polarities

    def fit_transform(self, docs, embeddings=None, bertopic_kwargs={}):
        # could also compute generalized stance (i.e. sentiment), then find vector
        # then use steering vector to find the difference along the vector
        # https://www.lesswrong.com/posts/ndyngghzFY388Dnew/implementing-activation-steering

        # the same idea more fleshed out for evaluation 
        # https://arjunbansal.substack.com/p/llms-know-more-than-what-they-say
        if self.method == 'topicllm':
            documents = self._topic_llm_fit_transform(docs, embeddings, bertopic_kwargs)
        elif self.method == 'llmtopic':
            documents = self._llm_topic_fit_transform(docs, bertopic_kwargs=bertopic_kwargs)
        elif self.method == 'steeringvectors':
            documents = self._embedding_fit_transform(docs, embeddings)
        else:
            raise ValueError(f"Method '{self.method}' not implemented")
        
        return self._get_targets_probs_polarity(documents)

    def get_topic_info(self):
        return self.topic_info

    def get_target_info(self):
        return self.target_info

    def _get_generator(self, lazy=False):
        if self.model_lib == 'transformers':
            return llms.Transformers(self.model_name, self.model_kwargs, self.tokenizer_kwargs, lazy=lazy)
        else:
            raise ValueError(f"LLM library '{self.model_lib}' not implemented")
        
    def _get_vector_probs(self, noun_phrase, vector, topic_docs, model, tokenizer):
        # for each noun_phrase, get probability of doc along vector
        prompt = "{text}. Targeted towards {noun_phrase}, is the preceeding text more likely to be {sent_a} or {sent_b}?".format(text="{text}", noun_phrase=noun_phrase, sent_a=vector.sent_a, sent_b=vector.sent_b)
        # get probability of doc along vector
        probs = self.generator.get_prompt_response_probs(prompt, topic_docs, model, tokenizer, vector.sent_a, vector.sent_b)
        return probs

                
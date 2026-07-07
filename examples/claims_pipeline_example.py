"""End-to-end claims pipeline: extract atomic claims from a large corpus, cluster
them into canonical claims, then retrieve and stance-classify the documents that
discuss a selected set of claims.

The general method (each step maps to a StanceMining knob):

    [1] Claim extraction     finetuned claim-extraction model (stance_target_type='claims')
    [2] Embed claims         encode at full dim, Matryoshka-truncate to a small dim
    [3] Cluster              HDBSCAN on the truncated embeddings
    [4] Canonical claims     an LLM generalizes each cluster's exemplars into 1-few claims
    [5] Select top claims    inspect get_target_info(), pick the claims of interest
    [6] Stance / entailment  classify each retrieved document toward the selected claims

Steps [1]-[4] discover and canonicalize claims across the whole corpus; steps
[5]-[6] focus stance classification on the claims you care about, using
`retrieve_documents_for_targets` to find the documents that discuss them.

Notes / knobs worth knowing:

  * Matryoshka embeddings. `embedding_dim=N` encodes at the model's full
    dimensionality and truncates to the leading N dims (cheaper clustering/dedup).
    The native dim and whether to re-normalize are inferred from the model; set
    `embedding_normalize=...` only to override the inference.

  * Prompted vs finetuned stance. `stance_detection_llm_method='prompting'`
    classifies stance with the large `model_name` model using the built-in 4-way
    prompt {supporting, refuting, discussing, irrelevant}, while target *extraction*
    stays on the finetuned claim model. Leave it unset (and set
    `claim_entailment_task` + `stance_detection_finetune_kwargs`) to use a small
    finetuned entailment head instead.

  * Clustering space. StanceMining normally runs PaCMAP(->5d) before HDBSCAN. Pass
    the `_IdentityReducer` below to cluster on the truncated embeddings directly;
    drop it to keep the default PaCMAP-then-cluster behaviour.

  * Structured claims. StanceMining claims are flat sentences; if you need typed
    claim structure (claimant/subject/object/type) you'd supply your own extractor.
"""

import polars as pl

import stancemining


class _IdentityReducer:
    """No-op dimensionality reducer so BERTopic/HDBSCAN cluster on the input
    (already Matryoshka-truncated) embeddings directly, rather than a
    further-reduced space."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


def build_hdbscan(min_cluster_size=150):
    """HDBSCAN sized for large corpora. Uses GPU cuML if available, else
    fast_hdbscan/hdbscan on CPU."""
    try:
        from cuml.cluster import HDBSCAN
        return HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_cluster_size,   # min_samples defaults to min_cluster_size
            cluster_selection_epsilon=0.0,
            cluster_selection_method="eom",
        )
    except ImportError:
        try:
            from fast_hdbscan import HDBSCAN
        except ImportError:
            from hdbscan import HDBSCAN
        return HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_cluster_size,
            cluster_selection_epsilon=0.0,
            cluster_selection_method="eom",
        )


def main():
    # --- Input -----------------------------------------------------------------
    # Any DataFrame with a text column (and optionally a time column for trends).
    doc_df = pl.read_csv("./tests/data/active_bluesky_sample.csv")

    # --- Configure StanceMining ------------------------------------------------
    model = stancemining.StanceMining(
        # [1] Claims, not noun-phrases.
        stance_target_type="claims",
        model_inference="vllm",

        # [4] + [6] One large model, used BOTH to aggregate each cluster's exemplars
        #     into canonical claims AND (via stance_detection_llm_method='prompting'
        #     below) to classify stance. Serve the largest reasoning model you can.
        model_name="Qwen/Qwen3-30B-A3B",

        # [1] Finetuned claim extraction. The published StanceMining claim extractors
        #     are domain-suffixed; pick the closest to your domain, or point at your own.
        llm_method="finetuned",
        target_extraction_model="bendavidsteel/Qwen3-1.7B-claim-extraction-ezstance",

        # [6] Stance via the large *prompted* model above rather than a small finetuned
        #     head. The prompted claim-stance path uses the 4-way scheme
        #     {supporting, refuting, discussing, irrelevant}, so no label config is needed.
        stance_detection_llm_method="prompting",

        # [2] Matryoshka embeddings: encode at the model's full dim, truncate to 32d.
        #     Native dim and re-normalization are inferred; we only set the truncation target.
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embedding_model_inference="sentence-transformers",  # or 'vllm'
        embedding_dim=32,

        # [3] Topic model backend (BERTopic wrapping HDBSCAN).
        topic_model="bertopic",

        verbose=True,
    )

    # --- [3] Clustering knobs: HDBSCAN on the truncated embeddings --------------
    topic_model_kwargs = {
        "hdbscan_model": build_hdbscan(min_cluster_size=150),
        "umap_model": _IdentityReducer(),  # cluster on the 32d vectors; drop for default PaCMAP(->5d)
    }

    # === Phase 1: discover claims ==============================================
    # Extract claims -> cluster -> generate canonical claims. Stance is deferred
    # (get_stance=False) so we can first curate the claims we care about.
    document_df = model.fit_transform(
        doc_df,
        text_column="text",
        get_stance=False,                     # defer stance to Phase 3
        generate_targets=True,                # [1] extract claims
        generate_higher_level_targets=True,   # [3] + [4] cluster -> canonical claims
        deduplicate_all_targets=True,         # merge near-duplicate claims across the corpus
        topic_model_kwargs=topic_model_kwargs,
        max_layers=1,                         # single flat clustering layer
    )

    # === Phase 2: select the top claims (Step [5]) =============================
    # Inspect the canonical claims and pick the ones of interest. Here we approximate
    # by taking the highest-volume claims; in practice you might hand-pick a list.
    target_info = model.get_target_info().sort("Count", descending=True)
    print(target_info.head(25))
    top_claims = target_info.head(20)["Target"].to_list()

    # === Phase 3: retrieve documents for the selected claims, then classify ====
    # Find the documents that actually discuss each selected claim, then run
    # stance/entailment on them. `retrieve_documents_for_targets` matches in the
    # same 32d Matryoshka space used for clustering.
    pairs_df = model.retrieve_documents_for_targets(
        document_df,
        targets=top_claims,
        text_column="text",
        similarity_threshold=0.5,   # threshold on claim-claim cosine; tune to taste
        match="claims",             # match selected claims against each doc's extracted claims
    )
    stance_df = model.get_stance(pairs_df, text_column="text")   # [6]

    # [optional] Stance-over-time trends, if your DataFrame has a datetime column:
    # trend_df, _ = stancemining.estimate.infer_stance_trends_for_all_targets(
    #     stance_df, time_column="created_at")

    stance_df.write_parquet("./claim_document_stances.parquet")
    print(f"Done. {len(stance_df)} document/claim stance rows.")


if __name__ == "__main__":
    main()

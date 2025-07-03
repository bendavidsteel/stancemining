API Reference
================================================

Stance Target Extraction and Stance Detection
------------------------------------------------

.. autoclass:: stancemining.main.StanceMining
    :members:

Stance Mean and Trend Estimation
---------------------------------------------------

.. autofunction:: stancemining.estimate.infer_stance_trends_for_all_targets
.. autofunction:: stancemining.estimate.infer_stance_trends_for_target

.. autofunction:: stancemining.estimate.infer_stance_normal_for_all_targets
.. autofunction:: stancemining.estimate.infer_stance_normal_for_target

Stance Visualization
-------------------------------------------------

.. autofunction:: stancemining.plot.plot_semantic_map
.. autofunction:: stancemining.plot.plot_trend_map

Multimodal Utilities
------------------------------------------------------

.. autofunction:: stancemining.utils.get_transcripts_from_audio_files
.. autofunction:: stancemining.utils.get_transcripts_from_video_files

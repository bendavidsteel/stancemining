Model Collections
=================

StanceMining provides a collection of fine-tuned language models hosted on Hugging Face for various stance-related tasks. These models are designed to work together in the stance mining pipeline.

All models are available at: https://huggingface.co/bendavidsteel/collections

Stance Detection Models
-----------------------

Models for detecting stance (favor, against, neutral) toward noun-phrases (topics/entities).

.. list-table::
   :header-rows: 1
   :widths: 50 15 35

   * - Model
     - Size
     - Hugging Face Link
   * - ``SmolLM2-135M-Instruct-stance-detection``
     - 0.1B
     - `Link <https://huggingface.co/bendavidsteel/SmolLM2-135M-Instruct-stance-detection>`_
   * - ``SmolLM2-360M-Instruct-stance-detection``
     - 0.4B
     - `Link <https://huggingface.co/bendavidsteel/SmolLM2-360M-Instruct-stance-detection>`_
   * - ``Qwen3-0.6B-stance-detection``
     - 0.6B
     - `Link <https://huggingface.co/bendavidsteel/Qwen3-0.6B-stance-detection>`_
   * - ``Qwen3-1.7B-stance-detection``
     - 2B
     - `Link <https://huggingface.co/bendavidsteel/Qwen3-1.7B-stance-detection>`_


Stance Target Extraction Models
-------------------------------

Models for extracting stance targets (noun-phrases) from documents expressing stance on those targets.

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Model
     - Hugging Face Link
   * - ``SmolLM2-360M-Instruct-stance-target-extraction``
     - `Link <https://huggingface.co/bendavidsteel/SmolLM2-360M-Instruct-stance-target-extraction>`_
   * - ``Qwen3-0.6B-stance-target-extraction``
     - `Link <https://huggingface.co/bendavidsteel/Qwen3-0.6B-stance-target-extraction>`_


Claim Extraction Models
-----------------------

Models for extracting atomic claims from social media posts.

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Model
     - Hugging Face Link
   * - ``Qwen3-0.6B-claim-extraction-ezstance``
     - `Link <https://huggingface.co/bendavidsteel/Qwen3-0.6B-claim-extraction-ezstance>`_
   * - ``Qwen3-1.7B-claim-extraction-ezstance``
     - `Link <https://huggingface.co/bendavidsteel/Qwen3-1.7B-claim-extraction-ezstance>`_
   * - ``Qwen3-1.7B-claim-extraction-romain``
     - `Link <https://huggingface.co/bendavidsteel/Qwen3-1.7B-claim-extraction-romain>`_


Claim Entailment Models
-----------------------

Models for claim entailment, also known as 'rumour stance detection'. These models determine the relationship between a claim and a piece of evidence.

.. list-table::
   :header-rows: 1
   :widths: 50 15 35

   * - Model
     - Size
     - Hugging Face Link
   * - ``Qwen3-0.6B-claim-entailment-stanceosaurus``
     - 0.6B
     - `Link <https://huggingface.co/bendavidsteel/Qwen3-0.6B-claim-entailment-stanceosaurus>`_
   * - ``Qwen3-1.7B-claim-entailment-5-labels``
     - 2B
     - `Link <https://huggingface.co/bendavidsteel/Qwen3-1.7B-claim-entailment-5-labels>`_


Using Custom Models
-------------------

You can use any of these models with StanceMining by specifying the model name:

.. code-block:: python

    import stancemining

    # Use a specific stance detection model
    model = stancemining.StanceMining(
        stance_detection_model="bendavidsteel/Qwen3-1.7B-stance-detection"
    )

    # Use a specific stance target extraction model
    model = stancemining.StanceMining(
        stance_target_extraction_model="bendavidsteel/Qwen3-0.6B-stance-target-extraction"
    )

Model Selection Guide
---------------------

- **For speed/low resource environments**: Use the SmolLM2-135M or SmolLM2-360M models
- **For balanced performance**: Use the Qwen3-0.6B models
- **For best accuracy**: Use the Qwen3-1.7B models

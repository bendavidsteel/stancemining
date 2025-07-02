Getting Started
==============================================

To get stance targets and stance from a corpus of documents:

.. code-block:: python

    import stancemining

    model = stancemining.StanceMining()
    docs = [
        "The government should invest more in renewable energy.",
        "I believe that climate change is a serious issue.",
        "Renewable energy sources are the future of our planet."
    ]
    document_df = model.fit_transform(docs)
    target_info_df = model.get_target_info()

To get stance target, stance, and stance trends from a corpus of documents:

.. code-block:: python

    import stancemining

    # dataframe containing documents with a 'text' column and a 'createtime' column

    model = stancemining.StanceMining()
    document_df = model.fit_transform(df)
    trend_df = stancemining.get_trends_for_all_targets(document_df)

To deploy stancemining app
If you need authentication for the app, you can set the environment variable `STANCE_AUTH_URL_PATH` to the URL of your authentication service (e.g., `myauth.com/login`). That path must accept a POST request with a JSON body containing `username` and `password` fields, and return a JSON response with a `token` field.
If you do not need authentication, you can leave the environment variable unset.

.. code-block:: bash

    # Install Docker and Docker Compose if not already installed

    # Run the StanceMining app with Docker Compose
    # Replace <your-data-path> with the path to your data directory
    # Replace <your-auth-url/login> with your authentication URL if needed

    export STANCE_DATA_PATH=<your-data-path>
    export STANCE_AUTH_URL_PATH=<your-auth-url/login>
    docker compose -f ./app/compose.yaml up


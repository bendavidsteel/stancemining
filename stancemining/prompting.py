from .llms import Transformers

def parse_generated_targets(outputs):
    outputs = [o.replace('stance', '').strip() for o in outputs if o != 'none' and o != None and o != '']
    outputs = list(set(outputs))
    return outputs

def ask_llm_zero_shot_stance_target(generator: Transformers, docs, generate_kwargs):
    
    prompts = []
    for doc in docs:
        prompt = [
            "You are an expert at analyzing documents for stance detection. ",
            """Your task is to identify the primary stance target in the given document, if one exists. A stance target is the specific topic, entity, or concept that is being supported or opposed in the text.

            Instructions:
            1. Carefully read the entire document.
            2. Determine if there is a clear stance being expressed towards any specific target.
            3. If a stance target exists:
            - Extract it as a noun phrase (e.g., "gun control", "climate change policy", "vaccination requirements")
            - Ensure the noun phrase captures the specific target being discussed
            - Do not include stance indicators (support/oppose) in the target
            4. If no clear stance target exists, return "None"

            Output format:
            Stance target: [noun phrase or "None"]

            Reasoning: [Brief explanation of why this is the stance target or why no stance target was found]
            ---
            Examples:

            Input: 'We must act now to reduce carbon emissions. The future of our planet depends on immediate action to address this crisis.'""",
            """Output:
            Stance target: climate change action
            Reasoning: The text clearly takes a position on the need for reducing carbon emissions and addressing climate issues.""",

            """Input: 'The weather was beautiful yesterday. I went for a long walk in the park and saw many birds.'""",
            """Output:
            Stance target: None
            Reasoning: This text is purely descriptive and does not express a stance towards any particular target.""",
            f"""---
            Input: '{doc}'""",
            """Output:
            Stance target: """
        ]
        prompts.append(prompt)

    if 'max_new_tokens' not in generate_kwargs:
        generate_kwargs['max_new_tokens'] = 7
    if 'num_samples' not in generate_kwargs:
        generate_kwargs['num_samples'] = 3

    generate_kwargs['add_generation_prompt'] = False
    generate_kwargs['continue_final_message'] = True
    
    docs_outputs = generator.generate(prompts, **generate_kwargs)

    all_outputs = []
    for outputs in docs_outputs:
        # remove reasoning and none responses
        outputs = [o.split('Reasoning:')[0].split('\n')[0].strip().lower() for o in outputs]
        outputs = parse_generated_targets(outputs)
        all_outputs.append(outputs)
    return all_outputs

def ask_llm_multi_doc_targets(generator, docs):
    # Multi-Document Stance Target Extraction Prompt
    formatted_docs = '\n'.join(docs)
    prompt = [
        "You are an expert at analyzing discussions across multiple documents.",
        """Your task is to identify a common stance target that multiple documents are expressing opinions about.

        Instructions:
        1. Read all provided documents
        2. Identify topics that appear across multiple documents
        3. Determine if there is a shared target that documents are taking stances on
        4. Express the target as a clear noun phrase

        Input:
        Documents: [list of texts]

        Output:
        Stance target: [noun phrase or "None"]
        Reasoning: [2-3 sentences explaining the choice]

        Examples:

        Example 1:
        Documents:
        "The council's new parking fees are excessive. Downtown businesses will suffer as shoppers avoid the area."
        "Increased parking rates will encourage public transit use. This is exactly what our city needs."
        "Local restaurant owners report 20% fewer customers since the parking fee increase."
        """,
        """
        Output:
        Stance target: downtown parking fees
        Reasoning: All three documents discuss the impact of new parking fees, though from different angles. The documents show varying stances on this policy change's effects on business and transportation behavior.""",
        """
        Example 2:
        Documents:
        "Beijing saw clear skies yesterday as wind cleared the air."
        "Traffic was unusually light on Monday due to the holiday."
        "New subway line construction continues on schedule."
        """,
        """
        Output:
        Stance target: None
        Reasoning: While all documents relate to urban conditions, they discuss different aspects with no common target for stance-taking. The texts are primarily descriptive rather than expressing stances.
        """,
        """
        Example 3:
        Documents:
        "AI art tools make creativity accessible to everyone."
        "Generated images lack the soul of human-made art."
        "Artists demand proper attribution when AI models use their work."
        """,
        """
        Output:
        Stance target: AI-generated art
        Reasoning: The documents all address AI's role in art creation, discussing its benefits, limitations, and ethical implications. While covering different aspects, they all take stances on AI's place in artistic creation.
        """,
        f"""---
        Documents:
        {formatted_docs}
        """,
        """Output:
        Stance target: """
    ]

    prompt_outputs = generator.generate([prompt], max_new_tokens=7, num_samples=3, add_generation_prompt=False, continue_final_message=True)
    outputs = prompt_outputs[0]
    # remove reasoning and none responses
    outputs = [o.split('Reasoning:')[0].split('\n')[0].strip().lower() for o in outputs]
    outputs = parse_generated_targets(outputs)
    return outputs

def ask_llm_zero_shot_stance(generator, docs, stance_targets):
    all_outputs = []
    for doc, stance_target in zip(docs, stance_targets):
        # Stance Classification Prompt
        prompt = [
            "You are an expert at analyzing stances in documents.",
            """Your task is to determine the stance expressed towards a specific target in the given document. Consider both explicit and implicit indicators of stance.

            Instructions:
            1. Carefully read the document while focusing on content related to the provided stance target.
            2. Classify the stance as one of:
            - FAVOR: Supporting, promoting, or agreeing with the target
            - AGAINST: Opposing, criticizing, or disagreeing with the target
            - NEUTRAL: Presenting balanced or objective information about the target

            Input:
            Document: [text]
            Stance target: [noun phrase]

            Output format:
            Stance: [FAVOR/AGAINST/NEUTRAL]
            Reasoning: [Brief explanation citing specific evidence from the text]

            Examples:

            Input:
            Document: "Research shows that diverse communities have higher rates of innovation. Cities with more international residents see increased patent filings and startups."
            Stance target: immigration""",
            """Output:
            Stance: FAVOR
            Reasoning: Text implicitly supports immigration by highlighting its positive economic impacts through innovation and business creation.
            """,
            """Input:
            Document: "Tech companies want self-governance of social media, while lawmakers push for oversight. Recent polls show the public remains divided."
            Stance target: social media regulation""",
            """Output:
            Stance: NEUTRAL
            Reasoning: Presents both industry and government perspectives without favoring either side.
            """,
            """Input:
            Document: "Standardized test scores correlate more with family income than academic ability, while countries using alternative assessments report better outcomes."
            Stance target: standardized testing""",
            """Output:
            Stance: AGAINST
            Reasoning: Implies tests are flawed by linking them to wealth rather than ability and noting superior alternatives.
            """,
            """Input:
            Document: "Some remote workers report higher productivity, others struggle with collaboration. Companies are testing hybrid models."
            Stance target: remote work""",
            """Output:
            Stance: NEUTRAL
            Reasoning: Balances positive and negative aspects of remote work without taking a position.
            """,
            f"""---
            Document: "{doc}"
            Stance target: {stance_target}""",
            """Output:
            Stance: """
        ]
        outputs = generator.generate([prompt], max_new_tokens=2, num_samples=1, add_generation_prompt=False, continue_final_message=True)[0]
        outputs = [o.split('Reasoning:')[0].split('\n')[0].strip() for o in outputs]
        all_outputs.append(outputs[0])
    return all_outputs


def ask_llm_target_aggregate(generator: Transformers, repr_docs, keywords):
    # Stance Target Topic Generalization Prompt
    repr_docs_s = ', '.join(f'"{d}"' for d in repr_docs)
    keywords_s = ', '.join(f'"{k}"' for k in keywords)
    prompt = [
        "You are an expert at analyzing and categorizing topics.",
        f"""Your task is to generate a list of appropriately specific generalized stance targets that best represent a cluster of related stance targets.

        Instructions:
        1. Review the provided stance targets and keywords that characterize the stance target cluster
        2. Identify the common specific stance issues these targets relate to
        3. Generate 1-3 appropriately specific noun phrases that:
        - Are specific enough to meaningfully represent the stance being taken
        - Are general enough to apply to the whole cluster
        - Include relevant context like policy domains, affected groups, or implementation approaches
        - Each noun phrase should be max 5 words
        - If inputs are low quality (just emojis, Twitter handles, random phrases), return []

        Input:
        Representative stance targets: [list of stance targets]
        Top keywords: [list of high tf-idf terms]

        Output format:
        Generalized target: ["appropriately specific noun phrase 1", "appropriately specific noun phrase 2", "appropriately specific noun phrase 3"]
        Reasoning: [1-2 sentences explaining why these specific generalizations fit]

        Examples:

        Input:
        Representative stance targets: ["EU vaccine passports", "mandatory covid shots", "provincial immunization requirements"]
        Top keywords: ["mandatory", "requirement", "public health", "immunization", "vaccination"]

        Output:
        Generalized target: ["mandatory healthcare worker vaccination", "cross-border vaccination verification", "regional immunization requirements"]
        Reasoning: These capture specific vaccination contexts and policy approaches across different regional systems without overly precise details.

        Input:
        Representative stance targets: ["London congestion charge", "Oslo car-free zones", "Paris emissions restrictions"]
        Top keywords: ["emissions", "vehicles", "urban", "electric", "pollution"]

        Output:
        Generalized target: ["urban vehicle restriction zones", "metropolitan emissions policies", "city center traffic regulations"]
        Reasoning: These targets identify specific urban policy approaches across different cities and implementation contexts without committing to exact timeframes.

        Input:
        Representative stance targets: ["content moderation", "online censorship", "platform guidelines"]
        Top keywords: ["social media", "guidelines", "content", "moderation", "posts"]

        Output:
        Generalized target: ["political content removal policies", "international platform regulations", "global speech moderation standards"]
        Reasoning: These specify content types and regulatory mechanisms across international contexts without naming specific platforms or exact policies.

        Input:
        Representative stance targets: ["Mediterranean migration crisis", "Canadian immigration system", "Schengen border controls"]
        Top keywords: ["migration", "borders", "refugees", "policy", "asylum"]

        Output:
        Generalized target: ["refugee processing protocols", "international border management", "asylum application systems"]
        Reasoning: These targets specify aspects of migration management across different regions without overly specific geographic or numerical details.

        Input:
        Representative stance targets: ["Ontario teacher contracts", "Ontario education spending", "Ontario school funding"]
        Top keywords: ["education", "funding", "teachers", "schools", "budget"]

        Output:
        Generalized target: ["Ontario public education funding models", "Ontario teacher collective agreements", "Ontario classroom resource availability"]
        Reasoning: These identify specific aspects of education funding for the Canadian province of Ontario without committing to specific percentage increases or exact amounts.

        Input:
        Representative stance targets: ["😂😂😂", "@JohnDoe2023", "@RealUserXYZ", "lol omg"]
        Top keywords: ["lol", "omg", "user", "haha"]

        Output:
        Generalized target: []
        Reasoning: The inputs contain only emojis, specific Twitter usernames, and generic expressions without substantive content related to any stance target.

        Input:
        Representative stance targets: ["just saying", "idk maybe", "whatever", "cool story"]
        Top keywords: ["just", "maybe", "whatever", "cool", "story"]

        Output:
        Generalized target: []
        Reasoning: The inputs are random short phrases without substantive content or specific topics that could be formed into meaningful stance targets.

        Input:
        Representative stance targets: ["ndp leader jagmeet singh", "jagmeet singh and j", "the jagmeet singh", "jagmeet singh", "@jagmeet singh"]
        Top keywords: ["singh", "khan", "dr", "jagmeet", "mohammed", "manana", "khans", "abdulla", "abu", "nan"]

        Output:
        Generalized target: ["jagmeet singh"]
        Reasoning: The inputs are all about one person, the common stance target is jagmeet singh.

        Input:
        Representative stance targets: ["rent control rent", "rent housing", "rent de l\'écart", "rent", "rent room rental rental"]
        Top keywords: ["rent", "rental", "landlord", "tenant", "lease", "renters", "landlords", "shortterm", "rentals", "tenants"]

        Output:
        Generalized target: ["rent control", "rent increase caps", "renter protections"]
        Reasoning: The inputs are about renting and landlords, and these outputs are all renter policy specific stance targets.
        
        Input:
        Representative stance targets: ["israeli", "israeli israeli,", "israeli israeli israeli", "israeli israeli", "israeli israel"]
        Top keywords: ["israeli", "palestinian", "israel", "gaza", "palestine", "israels", "palestinians", "conflict", "war", "attacks"]

        Output:
        Generalized target: ["state of israel", "ceasefire in palestine", "ceasefire in gaza"]
        Reasoning: The inputs are about israel and palestine, and these outputs are all israel-palestine specific stance targets.
        
        Input:
        Representative stance targets: [{repr_docs_s}]
        Top keywords: [{keywords_s}]

        Output:
        Generalized target: """
    ]
    raw_outputs = generator.generate([prompt], max_new_tokens=20, num_samples=1, add_generation_prompt=False, continue_final_message=True)[0]
    def parse_output(o):
        if ('\n' not in o) and (', ' not in o) and (']' not in o):  # incomplete generation
            # incomplete generation
            return []
        o = o.split('Reasoning:')[0].split('\n')[0].strip('\n').strip().strip('[]').split(', ')
        o = [t.strip('"') for t in o]
        return o
    outputs = parse_output(raw_outputs[0])
    outputs = [o for o in outputs if o != None and o != '']
    outputs = parse_generated_targets(outputs)
    return outputs
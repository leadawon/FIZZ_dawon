def orca_format(text, num_shot=10):
    system_message = (
        "You are Orca, a helpful assistant AI. You are given a summary sentence and several document sentences "
        "with their NLI scores. Based on the evidence, determine whether the summary is factual (1) or hallucinated (0)."
    )

    examples = [
        {
            "summary": "The president held a press conference today.",
            "evidence": [
                ("The president addressed the nation from the White House.", 0.85, 0.05, 0.10),
                ("The media gathered in the briefing room.", 0.65, 0.10, 0.25),
            ],
            "label": "1"
        },
        {
            "summary": "The Queen announced she would retire next year.",
            "evidence": [
                ("The Queen spoke at an event celebrating her reign.", 0.40, 0.45, 0.15),
                ("She discussed future plans but made no mention of retirement.", 0.10, 0.60, 0.30),
            ],
            "label": "0"
        },
        {
            "summary": "Heavy rains caused floods in northern India.",
            "evidence": [
                ("Monsoon season has begun in northern India.", 0.50, 0.05, 0.45),
                ("Local villages are submerged under water.", 0.88, 0.03, 0.09),
            ],
            "label": "1"
        },
        {
            "summary": "Elon Musk was arrested in Los Angeles.",
            "evidence": [
                ("Elon Musk visited a Tesla factory in California.", 0.15, 0.70, 0.15),
                ("There were rumors online but no official report.", 0.20, 0.65, 0.15),
            ],
            "label": "0"
        },
        {
            "summary": "The new iPhone was released last week.",
            "evidence": [
                ("Apple hosted an event unveiling the new iPhone.", 0.85, 0.02, 0.13),
                ("It is now available in stores worldwide.", 0.92, 0.01, 0.07),
            ],
            "label": "1"
        },
        {
            "summary": "A new species of bird was discovered in the Amazon.",
            "evidence": [
                ("Scientists have begun documenting a new bird species in Brazil.", 0.70, 0.15, 0.15),
                ("The bird has distinct plumage not seen before.", 0.80, 0.05, 0.15),
            ],
            "label": "1"
        },
        {
            "summary": "NASA cancelled the next Mars mission.",
            "evidence": [
                ("NASA announced budget changes.", 0.30, 0.45, 0.25),
                ("The next Mars rover is under construction.", 0.20, 0.65, 0.15),
            ],
            "label": "0"
        },
        {
            "summary": "The Prime Minister visited Canada this week.",
            "evidence": [
                ("The Prime Minister gave a speech in Ottawa.", 0.82, 0.04, 0.14),
                ("He also met with Canadian officials.", 0.78, 0.06, 0.16),
            ],
            "label": "1"
        },
        {
            "summary": "The company lost $1 billion in revenue.",
            "evidence": [
                ("The company reported a slight drop in quarterly profits.", 0.30, 0.50, 0.20),
                ("Revenue remained stable year over year.", 0.25, 0.60, 0.15),
            ],
            "label": "0"
        },
        {
            "summary": "The volcano erupted yesterday.",
            "evidence": [
                ("Smoke and lava were seen at the crater.", 0.88, 0.04, 0.08),
                ("Local authorities issued evacuation warnings.", 0.83, 0.03, 0.14),
            ],
            "label": "1"
        },
    ]

    fewshot_section = ""
    for example in examples[:num_shot]:
        fewshot_section += "<|im_start|>user\n"
        fewshot_section += f"Summary: {example['summary']}\n"
        fewshot_section += "Document Sentences with NLI Scores:\n"
        for i, (sent, ent, cont, neut) in enumerate(example["evidence"], 1):
            fewshot_section += f"  [{i}] {sent}\n"
            fewshot_section += f"     Entailment: {ent:.2f}, Contradiction: {cont:.2f}, Neutral: {neut:.2f}\n"
        fewshot_section += "<|im_end|>\n<|im_start|>assistant\n"
        fewshot_section += example["label"] + "\n<|im_end|>\n"

    prompt = f"""<|im_start|>system
{system_message}<|im_end|>
{fewshot_section}<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""
    return prompt

def orca_format_minus(text, num_shot=10):
    system_message = (
        "You are Orca, a helpful assistant AI. You are given a summary sentence and several document sentences "
        "with their entailment - contradiction (E-C) scores from an NLI model. Based on the scores, determine "
        "whether the summary is factual (1) or hallucinated (0)."
    )

    examples = [
        {
            "summary": "The president held a press conference today.",
            "evidence": [
                ("The president addressed the nation from the White House.", 0.75),
                ("The media gathered in the briefing room.", 0.55),
            ],
            "label": "1"
        },
        {
            "summary": "The Queen announced she would retire next year.",
            "evidence": [
                ("The Queen spoke at an event celebrating her reign.", -0.05),
                ("She discussed future plans but made no mention of retirement.", -0.50),
            ],
            "label": "0"
        },
        {
            "summary": "Heavy rains caused floods in northern India.",
            "evidence": [
                ("Monsoon season has begun in northern India.", 0.45),
                ("Local villages are submerged under water.", 0.85),
            ],
            "label": "1"
        },
        {
            "summary": "Elon Musk was arrested in Los Angeles.",
            "evidence": [
                ("Elon Musk visited a Tesla factory in California.", -0.55),
                ("There were rumors online but no official report.", -0.40),
            ],
            "label": "0"
        },
        {
            "summary": "The new iPhone was released last week.",
            "evidence": [
                ("Apple hosted an event unveiling the new iPhone.", 0.83),
                ("It is now available in stores worldwide.", 0.91),
            ],
            "label": "1"
        },
        {
            "summary": "A new species of bird was discovered in the Amazon.",
            "evidence": [
                ("Scientists have begun documenting a new bird species in Brazil.", 0.55),
                ("The bird has distinct plumage not seen before.", 0.75),
            ],
            "label": "1"
        },
        {
            "summary": "NASA cancelled the next Mars mission.",
            "evidence": [
                ("NASA announced budget changes.", -0.15),
                ("The next Mars rover is under construction.", -0.45),
            ],
            "label": "0"
        },
        {
            "summary": "The Prime Minister visited Canada this week.",
            "evidence": [
                ("The Prime Minister gave a speech in Ottawa.", 0.76),
                ("He also met with Canadian officials.", 0.74),
            ],
            "label": "1"
        },
        {
            "summary": "The company lost $1 billion in revenue.",
            "evidence": [
                ("The company reported a slight drop in quarterly profits.", -0.35),
                ("Revenue remained stable year over year.", -0.40),
            ],
            "label": "0"
        },
        {
            "summary": "The volcano erupted yesterday.",
            "evidence": [
                ("Smoke and lava were seen at the crater.", 0.84),
                ("Local authorities issued evacuation warnings.", 0.79),
            ],
            "label": "1"
        },
    ]

    fewshot_section = ""
    for example in examples[:num_shot]:
        fewshot_section += "<|im_start|>user\n"
        fewshot_section += f"Summary: {example['summary']}\n"
        fewshot_section += "Document Sentences with E-C Scores:\n"
        for i, (sent, ec_score) in enumerate(example["evidence"], 1):
            fewshot_section += f"  [{i}] {sent}\n"
            fewshot_section += f"     E-C Score: {ec_score:.2f}\n"
        fewshot_section += "<|im_end|>\n<|im_start|>assistant\n"
        fewshot_section += example["label"] + "\n<|im_end|>\n"

    prompt = f"""<|im_start|>system
{system_message}<|im_end|>
{fewshot_section}<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""
    return prompt



def zephyr_format(text):
    messages = [
{
    "role": "system",
    "content": "You are a helpful assistant. Please give me a list of atomic facts of the following texts."
},
{"role": "user", "content": "lisa courtney, of hertfordshire, has spent most of her life collecting pokemon memorabilia."},
{"role": "assistant", "content": """
- Lisa Courtney is from Hertfordshire.
- Lisa Courtney has spent most of her life collecting Pokémon memorabilia."""},
{"role": "user", "content": "prince jan zylinski said he was fed up with discrimination against poles living in britain."},
{"role": "assistant", "content": """
- Prince Jan Zylinski made a statement.
- The statement made by Prince Jan Zylinski was about discrimination.
- The statement made by Prince Jan Zylinski was regarding Poles living in Britain.
- Prince Jan Zylinski expressed feeling fed up with this type of discrimination."""},
{"role": "user", "content": "no charges were filed, there will be no travel ban."},
{"role": "assistant", "content": """
- No charges were filed.
- There will be no travel ban."""},
{"role": "user", "content": "rudd has pleaded guilty to threatening to kill and possession of drugs in a court."},
{"role": "assistant", "content": """
- Rudd has pleaded guilty.
- Rudd has pleaded guilty to threatening to kill.
- Rudd has pleaded guilty to possession of drugs."""},
{"role": "user", "content": "Lee made his acting debut in the film The Moon is the Sun's Dream (1992), and continued to appear in small and supporting roles throughout the 1990s."},
{"role": "assistant", "content": """
- Lee made his acting debut in The Moon is the Sun's Dream.
- The Moon is the Sun’s Dream is a film.
- The Moon is the Sun’s Dream was released in 1992.
- After Lee's acting debut, he appeared in small and supporting roles throughout the 1990s."""},
{"role": "user", "content": "In 1963, Collins became one of the third group of astronauts selected by NASA and he served as the back-up Command Module Pilot for the Gemini 7 mission."},
{"role": "assistant", "content": """
- Collins became an astronaut.
- Collins became one of the third group of astronauts selected by NASA in 1963.
- Collins served as the Command Module Pilot for the Gemini 7 mission."""},
{"role": "user", "content": "In addition to his acting roles, Bateman has written and directed two short films and is currently in development on his feature debut."},
{"role": "assistant", "content": """
- Bateman has acting roles.
- Bateman has written two short films.
- Bateman has directed two short films.
- Bateman is currently in development on his feature debut."""},
{"role": "user", "content": "Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969."},
{"role": "assistant", "content": """
- Michael Collins was born on October 31, 1930.
- Michael Collins is retired.
- Michael Collins is an American.
- Michael Collins was an astronaut.
- Michael Collins was a test pilot.
- Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969."""},
{"role": "user", "content": "{}\n".format(text)}]
    return messages

def mistral_format(text):
    messages = [
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: lisa courtney, of hertfordshire, has spent most of her life collecting pokemon memorabilia."},
        {"role": "assistant", "content": """
- Lisa Courtney is from Hertfordshire.
- Lisa Courtney has spent most of her life collecting Pokémon memorabilia."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: prince jan zylinski said he was fed up with discrimination against poles living in britain."},
        {"role": "assistant", "content": """
- Prince Jan Zylinski made a statement.
- The statement made by Prince Jan Zylinski was about discrimination.
- The statement made by Prince Jan Zylinski was regarding Poles living in Britain.
- Prince Jan Zylinski expressed feeling fed up with this type of discrimination."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: no charges were filed, there will be no travel ban."},
        {"role": "assistant", "content": """
- No charges were filed.
- There will be no travel ban."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: rudd has pleaded guilty to threatening to kill and possession of drugs in a court."},
        {"role": "assistant", "content": """
- Rudd has pleaded guilty.
- Rudd has pleaded guilty to threatening to kill.
- Rudd has pleaded guilty to possession of drugs."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: Lee made his acting debut in the film The Moon is the Sun's Dream (1992), and continued to appear in small and supporting roles throughout the 1990s."},
        {"role": "assistant", "content": """
- Lee made his acting debut in The Moon is the Sun's Dream.
- The Moon is the Sun’s Dream is a film.
- The Moon is the Sun’s Dream was released in 1992.
- After Lee's acting debut, he appeared in small and supporting roles throughout the 1990s."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: In 1963, Collins became one of the third group of astronauts selected by NASA and he served as the back-up Command Module Pilot for the Gemini 7 mission."},
        {"role": "assistant", "content": """
- Collins became an astronaut.
- Collins became one of the third group of astronauts selected by NASA in 1963.
- Collins served as the Command Module Pilot for the Gemini 7 mission."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: In addition to his acting roles, Bateman has written and directed two short films and is currently in development on his feature debut."},
        {"role": "assistant", "content": """
- Bateman has acting roles.
- Bateman has written two short films.
- Bateman has directed two short films.
- Bateman is currently in development on his feature debut."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969."},
        {"role": "assistant", "content": """
- Michael Collins was born on October 31, 1930.
- Michael Collins is retired.
- Michael Collins is an American.
- Michael Collins was an astronaut.
- Michael Collins was a test pilot.
- Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: {}".format(text)}]
    return messages
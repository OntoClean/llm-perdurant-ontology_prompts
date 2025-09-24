
README: Supplementary Materials for Meta-Property Classification via Prompting Strategies

These materials support the paper titled:

"Automating Perduring Subsumption Hierarchy Verification"

This repository contains Python scripts, input data, annotated datasets, and output files that demonstrate and compare multiple prompting strategies for assigning four key ontological meta-properties:
- Cumulativity
- Homeomericity
- Temporal Extent
- Agentivity

------------------------------------------------------------------------
CONTENTS
------------------------------------------------------------------------

[1] INPUT FILES
---------------
- 161_FrameNet.csv: FrameNet-based event definitions used for general-domain classification.
- Human_annotated_dataset.csv: Gold standard manual annotations for meta-property labels.

[2] OUTPUT FILES
----------------
- 161_Direct_prompting.csv: Output from direct prompting classification.
- 161_CoT_prompting.csv: Output from chain-of-thought (CoT) prompting.
- 161_FewShot_prompting.csv: Output from few-shot prompting.
- 161_analogical.csv: Output from analogical prompting.
- 161_meta_cognitive_prompting.csv: Output from meta-cognitive prompting.
- Military_Strategic_CoT_prompting.csv: Output for strategic-level military domain prompting.

[3] PROMPTING SCRIPTS
---------------------
Each script classifies meta-properties for event definitions using a different prompting technique via OpenAI’s GPT model.

- Direct_prompting.py: One-shot direct question-answer format.
- CoT_prompting.py: Structured chain-of-thought reasoning with intermediate reflection.
- Few-shot-Prompting.py: Uses examples (few-shot) to demonstrate prior label reasoning.
- Analogical_prompting.py: Maps input to ontologically analogous events.
- Meta-cognitive-prompting.py: Guides classification through reflective multi-step self-assessment.
- Military_Domain_Specific.py: Uses custom military event definitions and prompts for domain-specific assessment.
- Self_generated.py: Auxiliary or testing script for local prompting experiments.

[4] SUPPORTING FILES
--------------------
- requirements.txt: Python dependencies needed to run the scripts (see below).

------------------------------------------------------------------------
INSTRUCTIONS FOR REPRODUCIBILITY
------------------------------------------------------------------------

1. Ensure the following packages are installed:
   - Python 3.8+
   - pandas
   - openai



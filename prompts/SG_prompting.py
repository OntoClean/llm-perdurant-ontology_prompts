"""
Self-Generated Prompting (SGP) for perdurant meta-property
classification: INDEPENDENT EXECUTION.

Implements the SGP strategy specified in the Chapter 6 appendix
(Section 'Prompt Template: Self-Generated Prompting (SGP)').

This script issues all four property-specific SGP prompts
INDEPENDENTLY for every event, with no information passed
between prompts and no constraint forcing applied. Each
property is classified as a separate task, exactly four LLM
calls per event.

Each SGP prompt is a single API call. The self-generation
block inside the prompt instructs the model to:
    1. construct an internal classification framework for the
       property from the formal definition (Step 1);
    2. apply the framework to the target event silently
       (Step 2);
    3. emit only the final post-application label.
The framework, the application trace, and the candidate label
are not part of the model output, so the output format
matches the DP, FSP, CoT v2, MC v2, and AP independent runs
(single label per call, no reasoning trace).

This run is the SGP equivalent of the independent Direct
Prompting baseline reported in
Section app:ch6-dp-results-independent of the appendix.
The stage-wise pipeline using IC1, IC2, and the
contrapositive of IC3 is a separate run produced by a
different script.

Design lessons baked in from the CoT v1-v2 and MC v1-v2
empirical cycles:
  - Characteristic-of-type framing throughout. The granularity
    preamble's closing sentence asks what the type's
    definition entails, not whether a single counterexample
    can be exhibited.
  - Compact two-step structure (generate-framework then
    apply-framework). Long instruction lists fail
    (CoT v1 atomicity collapse).
  - No active witness-search instructions. The self-generation
    directive names structural elements but does not
    instruct the model to "attempt to identify a sub-event"
    or "search for a non-conforming sub-interval", which is
    the wording that destroyed CoT v1 atomicity and
    homeomericity.
  - System message includes an explicit "perform silently
    and emit only the final label" instruction to address
    the silent-execution risk.
  - Agentivity directive lists orthogonality as one of the
    structural elements the framework must evaluate, rather
    than as a separate post-hoc reminder.

Output CSV columns per event:
  EventType, Generic_Definition,
  Cumulativity, Atomicity, Homeomericity, Agentivity

Usage:
    1. Set the OPENAI_API_KEY environment variable:
           export OPENAI_API_KEY=sk-...    (macOS/Linux)
           setx OPENAI_API_KEY sk-...      (Windows, then reopen terminal)
    2. Run from a terminal:
           python sgp_independent_local.py \
               --input  /path/to/161_FrameNet.csv \
               --output /path/to/161_SGP_independent.csv

Security note. The script reads the key from the
OPENAI_API_KEY environment variable, so the key never
needs to live inside this file. Never hard-code an API
key in a file you intend to share or commit.
"""

import argparse
import os
import time

import pandas as pd
from openai import OpenAI


# =====================================================================
# OPENAI API KEY
# =====================================================================
# The key is read from the OPENAI_API_KEY environment variable.
# Get a key at https://platform.openai.com/api-keys, then set it
# in your shell before running:
#     export OPENAI_API_KEY=sk-...    (macOS/Linux)
#     setx OPENAI_API_KEY sk-...      (Windows, then reopen terminal)
# Never hard-code the key into this file.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


# =====================================================================
# CLIENT
# =====================================================================
client = OpenAI(api_key=OPENAI_API_KEY)


# =====================================================================
# SHARED PROMPT ELEMENTS
# =====================================================================

SYSTEM_MESSAGE = (
    "You are an ontology-engineering assistant that classifies "
    "perdurant (event) types against formally defined "
    "meta-properties drawn from foundational ontology. When a "
    "prompt asks you to construct and apply an internal "
    "classification framework, perform the framework "
    "construction and application silently, and emit only the "
    "final label. You return exactly one label per query: no "
    "explanation, no punctuation, no prefix, no reasoning "
    "trace, no framework description."
)

GRANULARITY_PREAMBLE = (
    "Interpretive context. You are classifying a single perdurant "
    "(event) type relative to one meta-property. All judgements below "
    "are made under the following declared context; finer or coarser "
    "granularities are out of scope for this classification.\n\n"
    "  Domain: general-purpose ontology\n"
    "  Modelling purpose: common-sense reasoning over events\n"
    "  Granularity: human-perceptual scale "
    "(seconds to days, depending on the event)\n"
    "  Individuation criterion (what counts as one occurrence): "
    "one contextually salient occurrence as typically reported in "
    "natural language\n\n"
    "All claims that follow are type-level. Where the prompt asks "
    "what the type's definition entails, it asks whether the property "
    "is a characteristic feature of the type as defined, not whether "
    "an unusual instance can be constructed that breaks the property."
)


# =====================================================================
# PROPERTY-DEFINITION BLOCKS (identical to DP, FSP, CoT, MC, AP)
# =====================================================================

property_definitions = {

    "Cumulativity": (
        "Property definition: Cumulativity.\n"
        "Cumulativity concerns whether the combination of two separate "
        "occurrences of the event type is itself a single occurrence "
        "of the same type. The two type-level outcomes are defined as "
        "follows.\n\n"
        "  Cumulative. For any two occurrences x and y of the event "
        "type, their mereological sum (the aggregate formed by "
        "considering the two occurrences together) is also a single "
        "occurrence of the same event type.\n\n"
        "  Anti-Cumulative. For any two occurrences x and y of the "
        "event type that are mutually independent (that is, neither "
        "x is a part of y, nor y is a part of x), their mereological "
        "sum is not a single occurrence of the same event type."
    ),

    "Atomicity": (
        "Property definition: Atomicity.\n"
        "Atomicity concerns whether occurrences of the event type can "
        "be decomposed into strictly smaller proper sub-events at the "
        "declared granularity. The two type-level outcomes are "
        "defined as follows.\n\n"
        "  Atomic. Every occurrence of the event type has no proper "
        "parts at the declared granularity. A proper part is a "
        "strictly smaller, identifiable constituent sub-event of the "
        "occurrence.\n\n"
        "  Anti-Atomic. Every occurrence of the event type has at "
        "least one proper part at the declared granularity."
    ),

    "Homeomericity": (
        "Property definition: Homeomericity.\n"
        "Homeomericity concerns whether the temporal sub-events of an "
        "occurrence (the smaller events that take place during "
        "continuous portions of the occurrence's duration) are "
        "themselves complete occurrences of the same event type. The "
        "two type-level outcomes are defined as follows.\n\n"
        "  Homeomeric. For every occurrence of the event type, every "
        "temporal sub-event of that occurrence (every event that "
        "takes place during a continuous portion of the occurrence's "
        "duration) is itself a complete occurrence of the same event "
        "type.\n\n"
        "  Anti-Homeomeric. For every occurrence of the event type, "
        "there is at least one temporal sub-event of that occurrence "
        "that is not itself a complete occurrence of the same event "
        "type. Common cases include preparation phases or culminating "
        "phases that differ in kind from the whole."
    ),

    "Agentivity": (
        "Property definition: Agentivity.\n"
        "Agentivity concerns whether occurrences of the event type "
        "are, by definition, intentionally initiated by an agent "
        "acting toward a culmination. An agent is a person, group, "
        "or organisation that is an enduring entity, distinct from "
        "the event itself. To intentionally initiate an event is "
        "for the agent to bring the event about with a goal in mind, "
        "and the event must contain a culminating part at which "
        "that goal is realised. The three type-level outcomes are "
        "defined as follows.\n\n"
        "  Agentive. Every occurrence of the event type is, by "
        "definition, intentionally initiated by some agent acting "
        "toward a culmination.\n\n"
        "  Anti-Agentive. No occurrence of the event type is "
        "intentionally initiated by an agent. The event arises "
        "through natural, mechanical, or reflexive processes.\n\n"
        "  Non-Agentive. The event type admits both intentionally "
        "initiated occurrences and occurrences that are not "
        "intentionally initiated. Whether agency is present "
        "depends on the context of the particular occurrence "
        "(for example, sneezing may be either voluntary or "
        "reflexive).\n\n"
        "Important. Judge agentivity on the definitional content "
        "of the event type itself. Do not infer agentivity from "
        "the event's mereological profile, that is, from whether "
        "the event is atomic, cumulative, or homeomeric. Two "
        "event types with identical mereological profiles can "
        "differ in agentivity."
    ),
}


# =====================================================================
# SGP SELF-GENERATION BLOCKS
# Each block names the structural elements of the formal predicate
# that the self-generated framework must evaluate, with a compact
# two-step generate-then-apply structure. The form of the
# framework is unconstrained; only the structural commitments are
# specified.
# =====================================================================

self_generation_blocks = {

    # -----------------------------------------------------------------
    # CUMULATIVITY
    # Structural elements: closure under mereological sum,
    # independence guard, four anti-cumulativity triggers.
    # -----------------------------------------------------------------
    "Cumulativity": (
        "Self-generation (silent). Carry out the following two steps "
        "internally, in order. Do not include the framework, the "
        "application, or any reasoning trace in your output. Emit "
        "only the final label.\n\n"

        "Step 1 (generate the framework).\n"
        "Construct an internal classification framework for "
        "Cumulativity, drawing on the property definition above. "
        "Your framework must evaluate the following structural "
        "elements of the formal predicate:\n"
        "  - whether the type's definition entails closure under "
        "mereological sum (combining two separate occurrences yields "
        "a single longer occurrence of the same type);\n"
        "  - the independence guard for the negative case "
        "(occurrences are mutually independent rather than parts of "
        "one another);\n"
        "  - the structural triggers of anti-cumulativity that the "
        "type's definition may carry: a culmination point at which an "
        "occurrence is complete; an event-external individuation "
        "criterion under which each occurrence is counted "
        "separately; a uniqueness condition built into the type's "
        "identity criteria; a punctual character with no temporal "
        "extent.\n"
        "The framework's form is unconstrained: it may be a decision "
        "procedure, a check-list, a mapping from structural features "
        "to labels, or any other operationalisation of the "
        "predicate.\n\n"

        "Step 2 (apply the framework).\n"
        "Apply the framework constructed in Step 1 to the target "
        "event type's definition, and produce the final label."
    ),

    # -----------------------------------------------------------------
    # ATOMICITY
    # Structural elements: granularity-relativity, PP predicate,
    # universal-quantifier scope, characteristic atomic profiles
    # (achievement, decision-point, state-entry).
    # No active "search for sub-event" wording (CoT v1 lesson).
    # -----------------------------------------------------------------
    "Atomicity": (
        "Self-generation (silent). Carry out the following two steps "
        "internally, in order. Do not include the framework, the "
        "application, or any reasoning trace in your output. Emit "
        "only the final label.\n\n"

        "Step 1 (generate the framework).\n"
        "Construct an internal classification framework for "
        "Atomicity at the granularity declared in the "
        "interpretive-context preamble. Your framework must evaluate "
        "the following structural elements of the formal predicate:\n"
        "  - the granularity-relativity of the classification (the "
        "framework applies at the declared granularity only; finer or "
        "coarser granularities are out of scope);\n"
        "  - whether the type's definition characteristically entails "
        "proper sub-events at the declared granularity (a proper "
        "sub-event is a strictly smaller, identifiable constituent "
        "event, distinct from a participant or role);\n"
        "  - the universal-quantifier scope: the framework classifies "
        "the type, not a particular occurrence; the question is "
        "whether the type's definition entails the property for the "
        "type's extension as a whole;\n"
        "  - canonical atomic profiles at human-perceptual "
        "granularity: achievements (single boundary transitions), "
        "decision-points (single discrete cognitive acts), "
        "state-entries (the moment of entering a state); and "
        "canonical anti-atomic profiles: phased accomplishments, "
        "distributed activities with concurrent sub-acts, composite "
        "processes with parallel coordinated strands.\n"
        "The framework's form is unconstrained.\n\n"

        "Step 2 (apply the framework).\n"
        "Apply the framework constructed in Step 1 to the target "
        "event type's definition, and produce the final label."
    ),

    # -----------------------------------------------------------------
    # HOMEOMERICITY
    # Structural elements: PT predicate, universal closure, canonical
    # profiles. No "search for non-conforming sub-interval" wording.
    # -----------------------------------------------------------------
    "Homeomericity": (
        "Self-generation (silent). Carry out the following two steps "
        "internally, in order. Do not include the framework, the "
        "application, or any reasoning trace in your output. Emit "
        "only the final label.\n\n"

        "Step 1 (generate the framework).\n"
        "Construct an internal classification framework for "
        "Homeomericity. Your framework must evaluate the following "
        "structural elements of the formal predicate:\n"
        "  - the temporal-part predicate: a temporal sub-event is an "
        "event that takes place during a continuous portion of an "
        "occurrence's duration; this is distinct from a participant, "
        "role, or component of the occurrence;\n"
        "  - whether the type's definition characteristically entails "
        "closure under temporal-part formation (every continuous "
        "temporal sub-interval of an occurrence is itself an "
        "occurrence of the same type);\n"
        "  - canonical homeomeric profiles: persistent states (such "
        "as being seated, holding, containing, surrounding), and "
        "homogeneous processes (such as running, walking, breathing, "
        "carrying);\n"
        "  - canonical anti-homeomeric profiles: phase-distinct "
        "accomplishments (foundation, walls, and roof in building a "
        "house); goal-directed activities with preparation, "
        "execution, and culmination; composite distributed "
        "activities whose strands are themselves of different types.\n"
        "The framework's form is unconstrained.\n\n"

        "Step 2 (apply the framework).\n"
        "Apply the framework constructed in Step 1 to the target "
        "event type's definition, and produce the final label."
    ),

    # -----------------------------------------------------------------
    # AGENTIVITY
    # Structural elements: agent-typing, initiation-typing,
    # culminating-part, three-way label space, orthogonality.
    # Orthogonality is named as a structural element rather than
    # as a separate reminder, addressing the agentivity-by-mereo-
    # logical-cue over-prediction documented in DP independent.
    # -----------------------------------------------------------------
    "Agentivity": (
        "Self-generation (silent). Carry out the following two steps "
        "internally, in order. Do not include the framework, the "
        "application, or any reasoning trace in your output. Emit "
        "only the final label.\n\n"

        "Step 1 (generate the framework).\n"
        "Construct an internal classification framework for "
        "Agentivity. Your framework must evaluate the following "
        "structural elements of the formal predicate:\n"
        "  - the agent-typing requirement: an enduring agent (a "
        "person, group, or organisation) distinct from the perdurant; "
        "the agent's enduring nature distinguishes it from a "
        "participant role within the event;\n"
        "  - the initiation-typing requirement: the agent "
        "intentionally initiates the occurrence with a goal in "
        "mind;\n"
        "  - the culminating-part requirement: the occurrence "
        "contains a part at which the agent's goal is realised;\n"
        "  - the three-way label space: Agentive (every occurrence "
        "is intentionally initiated), Anti-Agentive (the type's "
        "definition characteristically excludes intentional "
        "initiation, such as natural, mechanical, or reflexive "
        "processes), and Non-Agentive (the type's definition is "
        "neutral on intentional initiation, admitting both "
        "deliberate and non-deliberate cases);\n"
        "  - the orthogonality of agentivity to the mereological "
        "profile: two event types with identical mereological "
        "profiles (same combinations of cumulativity, atomicity, "
        "and homeomericity) can differ in agentivity. The framework "
        "must judge agentivity from definitional content, not from "
        "mereological cues such as the presence of a culmination, "
        "an animate participant, or a goal-directed framing.\n"
        "The framework's form is unconstrained.\n\n"

        "Step 2 (apply the framework).\n"
        "Apply the framework constructed in Step 1 to the target "
        "event type's definition, and produce the final label."
    ),
}


# =====================================================================
# TASK FOOTERS (per property)
# =====================================================================

footer_blocks = {
    "Cumulativity": (
        "Task. Return exactly one of the valid labels, with no "
        "explanation and no reasoning trace. The label you return "
        "must follow from the application of your self-generated "
        "framework to the target event.\n"
        "Valid answers: cumulative, anti-cumulative"
    ),
    "Atomicity": (
        "Task. Return exactly one of the valid labels, with no "
        "explanation and no reasoning trace. The label you return "
        "must follow from the application of your self-generated "
        "framework to the target event.\n"
        "Valid answers: atomic, anti-atomic"
    ),
    "Homeomericity": (
        "Task. Return exactly one of the valid labels, with no "
        "explanation and no reasoning trace. The label you return "
        "must follow from the application of your self-generated "
        "framework to the target event.\n"
        "Valid answers: homeomeric, anti-homeomeric"
    ),
    "Agentivity": (
        "Task. Return exactly one of the valid labels, with no "
        "explanation and no reasoning trace. The label you return "
        "must follow from the application of your self-generated "
        "framework to the target event.\n"
        "Valid answers: agentive, anti-agentive, non-agentive"
    ),
}


# =====================================================================
# VALID LABEL SETS (for normalisation)
# =====================================================================

valid_labels = {
    "Cumulativity":   {"cumulative", "anti-cumulative"},
    "Atomicity":      {"atomic", "anti-atomic"},
    "Homeomericity":  {"homeomeric", "anti-homeomeric"},
    "Agentivity":     {"agentive", "anti-agentive", "non-agentive"},
}


# =====================================================================
# PROMPT CONSTRUCTION AND QUERYING
# =====================================================================

def construct_sgp_prompt(event_name: str,
                          event_definition: str,
                          meta_property: str) -> str:
    """Build the full SGP user-prompt for a target event."""
    event_block = (
        f"Event type to classify.\n"
        f"  Name: {event_name}\n"
        f"  Definition: {event_definition}"
    )
    return "\n\n".join([
        GRANULARITY_PREAMBLE,
        property_definitions[meta_property],
        event_block,
        self_generation_blocks[meta_property],
        footer_blocks[meta_property],
    ])


def normalise_label(raw: str, allowed: set) -> str:
    """Map raw model output to one of the allowed labels."""
    cleaned = raw.strip().lower().rstrip(".").replace(" ", "-")
    if cleaned in allowed:
        return cleaned
    # Tolerate hyphen-vs-no-hyphen variants
    for label in allowed:
        if cleaned.replace("-", "") == label.replace("-", ""):
            return label
    return "error"


def query_sgp(event_name: str,
               event_definition: str,
               meta_property: str) -> str:
    """Issue one SGP classification call. Returns a normalised label."""
    try:
        prompt = construct_sgp_prompt(event_name, event_definition,
                                       meta_property)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=20,
            temperature=0.2,
            top_p=1.0,
        )
        raw = response.choices[0].message.content
        return normalise_label(raw, valid_labels[meta_property])
    except Exception as exc:
        print(f"  [{meta_property}] error for '{event_name}': {exc}")
        return "error"


# =====================================================================
# MAIN LOOP
# All four prompts are issued for every event, independently.
# =====================================================================

def main(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path, encoding="ISO-8859-1")
    df = df.drop_duplicates(subset=["EventType", "Generic_Definition"])
    df = df.reset_index(drop=True)

    meta_properties = ["Cumulativity", "Atomicity",
                       "Homeomericity", "Agentivity"]

    for col in meta_properties:
        if col not in df.columns:
            df[col] = ""

    for i, row in df.iterrows():
        event_type = row["EventType"]
        definition = row["Generic_Definition"]
        print(f"[{i+1}/{len(df)}] Processing: {event_type}")

        for meta_property in meta_properties:
            label = query_sgp(event_type, definition, meta_property)
            df.at[i, meta_property] = label
            print(f"    {meta_property}: {label}")
            time.sleep(1)  # gentle rate-limiting

        # Save after each event so progress survives interruption.
        df.to_csv(output_path, index=False)

    total_calls = len(df) * len(meta_properties)
    print()
    print("SGP independent classification complete.")
    print(f"  Events processed: {len(df)}")
    print(f"  Total LLM calls:  {total_calls} "
          f"({len(df)} events x {len(meta_properties)} properties)")
    print(f"  Output:           {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Independent Self-Generated Prompting for "
                    "perdurant meta-property classification."
    )
    parser.add_argument(
        "--input",
        default="161_FrameNet.csv",
        help="Input CSV with EventType and Generic_Definition columns.",
    )
    parser.add_argument(
        "--output",
        default="161_SGP_independent.csv",
        help="Output CSV with the classification results.",
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        raise SystemExit(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before running, e.g.\n"
            "    export OPENAI_API_KEY=sk-...    (macOS/Linux)\n"
            "    setx OPENAI_API_KEY sk-...      (Windows, then reopen terminal)"
        )

    main(args.input, args.output)
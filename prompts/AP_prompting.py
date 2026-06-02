"""
Analogical Prompting (AP) for perdurant meta-property
classification: INDEPENDENT EXECUTION.

Implements the AP strategy specified in the Chapter 6 appendix
(Section 'Prompt Template: Analogical Prompting (AP)').

This script issues all four property-specific AP prompts
INDEPENDENTLY for every event, with no information passed
between prompts and no constraint forcing applied. Each
property is classified as a separate task, exactly four LLM
calls per event.

Each AP prompt presents one or two non-event structural
analogues drawn from a domain related to the target
meta-property (mass-vs-count nouns for cumulativity,
mereological atoms for atomicity, substance-vs-complex
object for homeomericity, deliberate-vs-incidental
causation for agentivity), and asks the model to transfer
the analogical structure to the target event.

Design lessons baked in from the CoT v1-v2 and MC v1-v2
empirical cycles:
  - Characteristic-of-type framing throughout (no "single
    counterexample defeats" wording).
  - The transfer instruction is a single sentence at the
    end of each block, not a multi-step diagnostic. This
    keeps AP structurally distinct from CoT and MC and
    avoids the long-instruction-list failure mode that
    destroyed CoT v1.
  - The system message instructs the model to perform the
    analogical mapping internally and emit only the final
    label, addressing the silent-reasoning risk.
  - The granularity preamble uses characteristic-of-type
    framing (matching CoT v2 and MC v2).
  - For agentivity, the analogue explicitly states that
    the two contrast cases share the same mereological
    profile, addressing the agentivity-by-mereological-cue
    over-prediction documented in DP independent results.

This run is the AP equivalent of the independent Direct
Prompting baseline reported in
Section app:ch6-dp-results-independent of the appendix.
The stage-wise pipeline using IC1, IC2, and the
contrapositive of IC3 is a separate run produced by a
different script.

Output CSV columns per event:
  EventType, Generic_Definition,
  Cumulativity, Atomicity, Homeomericity, Agentivity

Usage:
    1. Set the OPENAI_API_KEY environment variable:
           export OPENAI_API_KEY=sk-...    (macOS/Linux)
           setx OPENAI_API_KEY sk-...      (Windows, then reopen terminal)
    2. Run from a terminal:
           python ap_independent_local.py \
               --input  /path/to/161_FrameNet.csv \
               --output /path/to/161_AP_independent.csv

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
    "prompt presents a structural analogue from a non-event "
    "domain, identify the structural correspondence to the "
    "target event silently and emit only the final label. You "
    "return exactly one label per query: no explanation, no "
    "punctuation, no prefix, no reasoning trace."
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
# PROPERTY-DEFINITION BLOCKS (identical to DP, FSP, CoT, MC)
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
# AP ANALOGICAL BLOCKS
# Each block presents one or two non-event analogues drawn from a
# structurally related domain, with the analogue's classification,
# and a single transfer instruction at the end. None of the
# analogues is an event type, and none is a MAVEN entry.
# =====================================================================

analogical_blocks = {

    # -----------------------------------------------------------------
    # CUMULATIVITY: 2 analogues from mass-vs-count noun semantics
    # -----------------------------------------------------------------
    "Cumulativity": (
        "Structural analogue for Cumulativity (drawn from "
        "mass-versus-count noun semantics).\n\n"

        "Analogue 1 (cumulative-type, physical mass-noun): water.\n"
        "The mass noun water denotes a substance whose two portions "
        "combine to yield a single longer portion of the same "
        "substance. Pour the contents of one glass of water into "
        "a second glass; the result is one larger portion of "
        "water, not two. The mass-noun classification of water is "
        "therefore cumulative: the combination of two portions is "
        "itself a portion of the same substance.\n\n"

        "Analogue 2 (anti-cumulative-type, count noun): chair.\n"
        "The count noun chair denotes a discrete individual whose "
        "two instances cannot combine into a single chair. Two "
        "chairs side by side are two chairs, not one chair. The "
        "count-noun classification of chair is therefore "
        "anti-cumulative: the combination of two individuals is "
        "not itself a single individual of the same kind.\n\n"

        "Structural mapping for the target. An event type behaves "
        "like a mass noun (cumulative) if combining two separate "
        "occurrences of the type yields a single longer occurrence "
        "of the same type. An event type behaves like a count noun "
        "(anti-cumulative) if combining two separate occurrences "
        "yields two distinct occurrences rather than one. Identify "
        "which analogue the target event type's definition "
        "characteristically resembles, and apply the corresponding "
        "label."
    ),

    # -----------------------------------------------------------------
    # ATOMICITY: 2 analogues from mereological-atom theory
    # -----------------------------------------------------------------
    "Atomicity": (
        "Structural analogue for Atomicity (drawn from mereological "
        "atom theory).\n\n"

        "Analogue 1 (atomic-type, fundamental particle): an "
        "electron.\n"
        "An electron is a fundamental physical particle that "
        "cannot be decomposed into smaller constituent particles "
        "at the resolution at which physics treats electrons as "
        "elementary. The mereological classification of an "
        "electron is therefore atomic: it has no proper parts at "
        "the declared resolution.\n\n"

        "Analogue 2 (anti-atomic-type, complex object): a "
        "bicycle.\n"
        "A bicycle is a physical object that consists of "
        "identifiable proper parts (wheels, handlebars, frame, "
        "pedals, chain). Each part is a strictly smaller, "
        "identifiable constituent of the whole bicycle. The "
        "mereological classification of a bicycle is therefore "
        "anti-atomic: it has at least one proper part, "
        "identifiable as a distinct constituent of the same kind "
        "of object (a physical artefact) but smaller than the "
        "whole.\n\n"

        "Structural mapping for the target. An event type behaves "
        "like a fundamental particle (atomic) if its definition "
        "makes the event a single indivisible act at the declared "
        "granularity, with no proper sub-event of the same kind "
        "identifiable. An event type behaves like a complex object "
        "(anti-atomic) if its definition makes the event have "
        "identifiable proper sub-events, each a strictly smaller "
        "constituent of the whole. Identify which analogue the "
        "target event type's definition characteristically "
        "resembles at the declared granularity, and apply the "
        "corresponding label."
    ),

    # -----------------------------------------------------------------
    # HOMEOMERICITY: 1 analogue from substance-vs-complex-object
    # -----------------------------------------------------------------
    "Homeomericity": (
        "Structural analogue for Homeomericity (drawn from "
        "physical-mereology of substances and complex objects).\n\n"

        "Analogue (substance vs complex object): water versus an "
        "automobile.\n"
        "A spatial part of a portion of water is itself a portion "
        "of water: any sub-region of a water-filled volume is "
        "itself water. The substance water is therefore spatially "
        "homeomeric: every spatial part of a portion is itself an "
        "instance of the substance.\n"
        "A spatial part of an automobile is not itself an "
        "automobile: the engine is an engine, not an automobile; "
        "the wheel is a wheel, not an automobile. The complex "
        "object automobile is therefore spatially anti-homeomeric: "
        "at least one spatial part is not itself an instance of "
        "the object's kind.\n\n"

        "Structural mapping for the target. An event type behaves "
        "like a substance (homeomeric) if its definition makes "
        "every continuous temporal sub-interval of an occurrence "
        "itself an occurrence of the same type. Persistent states "
        "(being seated, holding, containing) and homogeneous "
        "processes (running, walking, breathing) are the canonical "
        "event analogues of substances. An event type behaves "
        "like a complex object (anti-homeomeric) if its definition "
        "makes at least one temporal sub-interval of an occurrence "
        "qualitatively different from the whole. Phase-distinct "
        "accomplishments, goal-directed activities with "
        "preparation and culmination, and composite distributed "
        "activities are the canonical event analogues of complex "
        "objects. Identify which analogue the target event type's "
        "definition characteristically resembles, and apply the "
        "corresponding label."
    ),

    # -----------------------------------------------------------------
    # AGENTIVITY: 1 analogue from deliberate-vs-incidental causation,
    # constructed to share the same mereological profile (addresses
    # the agentivity-by-mereological-cue over-prediction failure)
    # -----------------------------------------------------------------
    "Agentivity": (
        "Structural analogue for Agentivity (drawn from "
        "deliberate-versus-incidental causation in legal and "
        "philosophical theories of action).\n\n"

        "Analogue (deliberate vs incidental causation): "
        "deliberate construction versus a natural collapse.\n"
        "A deliberate construction is brought about by an agent "
        "who intends to produce the constructed object, acting "
        "toward the culmination of the object being completed. "
        "The agent is enduring (a person, a company, a public "
        "authority) and distinct from the construction event "
        "itself; the agent's intention is constitutive of the "
        "type. The causal type deliberate construction is "
        "therefore agentive: every occurrence requires "
        "intentional initiation by a distinct agent, with a "
        "culminating part at which the agent's goal is "
        "realised.\n"
        "A natural collapse arises from physical processes "
        "(material fatigue, weather, geological force) without "
        "intentional initiation. There is no agent who brings "
        "about the collapse with a goal in mind; the occurrence "
        "proceeds from the material conditions of the collapsing "
        "object. The causal type natural collapse is therefore "
        "anti-agentive: no occurrence is intentionally initiated "
        "by an agent.\n"
        "Crucially, the deliberate-construction and "
        "natural-collapse causal types share the same "
        "mereological profile (both have proper temporal phases, "
        "both are anti-cumulative, both are anti-homeomeric). "
        "The difference between them is purely a difference in "
        "the role of intentional initiation in the type's "
        "definition.\n\n"

        "Structural mapping for the target. An event type "
        "behaves like deliberate causation (agentive) if its "
        "definition characteristically requires intentional "
        "initiation of every occurrence by an enduring agent, "
        "acting toward a culmination at which the agent's goal "
        "is realised. An event type behaves like incidental "
        "causation (anti-agentive) if its definition "
        "characteristically excludes intentional initiation, with "
        "occurrences arising from natural, mechanical, or "
        "reflexive processes. An event type is non-agentive if "
        "its definition admits both deliberate and incidental "
        "occurrences, or is itself neutral on intentional "
        "initiation. Identify which analogue the target event "
        "type's definition characteristically resembles, judging "
        "from the type's definitional content rather than from "
        "its mereological profile."
    ),
}


# =====================================================================
# TASK FOOTERS (per property)
# =====================================================================

footer_blocks = {
    "Cumulativity": (
        "Task. The structural distinction illustrated by the "
        "analogue maps onto the meta-property classification of "
        "the event type. Identify the structural correspondence "
        "and apply the same classification logic to the target "
        "event. Return exactly one of the valid labels, with no "
        "explanation and no reasoning trace.\n"
        "Valid answers: cumulative, anti-cumulative"
    ),
    "Atomicity": (
        "Task. The structural distinction illustrated by the "
        "analogue maps onto the meta-property classification of "
        "the event type. Identify the structural correspondence "
        "and apply the same classification logic to the target "
        "event. Return exactly one of the valid labels, with no "
        "explanation and no reasoning trace.\n"
        "Valid answers: atomic, anti-atomic"
    ),
    "Homeomericity": (
        "Task. The structural distinction illustrated by the "
        "analogue maps onto the meta-property classification of "
        "the event type. Identify the structural correspondence "
        "and apply the same classification logic to the target "
        "event. Return exactly one of the valid labels, with no "
        "explanation and no reasoning trace.\n"
        "Valid answers: homeomeric, anti-homeomeric"
    ),
    "Agentivity": (
        "Task. The structural distinction illustrated by the "
        "analogue maps onto the meta-property classification of "
        "the event type. Identify the structural correspondence "
        "and apply the same classification logic to the target "
        "event. Return exactly one of the valid labels, with no "
        "explanation and no reasoning trace.\n"
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

def construct_ap_prompt(event_name: str,
                         event_definition: str,
                         meta_property: str) -> str:
    """Build the full AP user-prompt for a target event."""
    event_block = (
        f"Event type to classify.\n"
        f"  Name: {event_name}\n"
        f"  Definition: {event_definition}"
    )
    return "\n\n".join([
        GRANULARITY_PREAMBLE,
        property_definitions[meta_property],
        analogical_blocks[meta_property],
        event_block,
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


def query_ap(event_name: str,
              event_definition: str,
              meta_property: str) -> str:
    """Issue one AP classification call. Returns a normalised label."""
    try:
        prompt = construct_ap_prompt(event_name, event_definition,
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
            label = query_ap(event_type, definition, meta_property)
            df.at[i, meta_property] = label
            print(f"    {meta_property}: {label}")
            time.sleep(1)  # gentle rate-limiting

        # Save after each event so progress survives interruption.
        df.to_csv(output_path, index=False)

    total_calls = len(df) * len(meta_properties)
    print()
    print("AP independent classification complete.")
    print(f"  Events processed: {len(df)}")
    print(f"  Total LLM calls:  {total_calls} "
          f"({len(df)} events x {len(meta_properties)} properties)")
    print(f"  Output:           {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Independent Analogical Prompting for "
                    "perdurant meta-property classification."
    )
    parser.add_argument(
        "--input",
        default="161_FrameNet.csv",
        help="Input CSV with EventType and Generic_Definition columns.",
    )
    parser.add_argument(
        "--output",
        default="161_AP_independent.csv",
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
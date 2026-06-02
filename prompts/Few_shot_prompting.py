"""
Few-Shot Prompting (FSP) for perdurant meta-property classification:
INDEPENDENT EXECUTION.

Implements the FSP strategy specified in the Chapter 6 appendix
(Section 'Prompt Template: Few-Shot Prompting (FSP)').

This script issues all four property-specific FSP prompts
INDEPENDENTLY for every event, with no information passed between
prompts and no constraint forcing applied. Each property is
classified as a separate task, exactly four LLM calls per event.

This run is the FSP equivalent of the independent Direct Prompting
baseline reported in Section app:ch6-dp-results-independent of the
appendix. The stage-wise pipeline using IC1, IC2, and the
contrapositive of IC3 is a separate run produced by a different
script.

Output CSV columns per event:
  EventType, Generic_Definition,
  Cumulativity, Atomicity, Homeomericity, Agentivity

Usage:
    1. Open this file in a text editor.
    2. Find the line:
           OPENAI_API_KEY = "sk-REPLACE-WITH-YOUR-KEY"
       near the top of the file, and paste your real key between
       the quotes.
    3. Run from a terminal:
           python fsp_independent_local.py \
               --input  /path/to/161_FrameNet.csv \
               --output /path/to/161_FSP_independent.csv

Security note. Keep this file private once your key is pasted in.
Do not share, email, or commit this file with the key still inside.
"""

import argparse
import os
import time

import pandas as pd
from openai import OpenAI


# =====================================================================
# OPENAI API KEY
# =====================================================================
# Paste your OpenAI API key between the quotes below before running.
# Get a key at https://platform.openai.com/api-keys.
# Treat this key like a password: do not commit this file to a public
# repository, and do not share the file once the key is filled in.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


# =====================================================================
# CLIENT
# =====================================================================
client = OpenAI(api_key=OPENAI_API_KEY)


# =====================================================================
# SHARED PROMPT ELEMENTS (identical to DP)
# =====================================================================

SYSTEM_MESSAGE = (
    "You are an ontology-engineering assistant that classifies "
    "perdurant (event) types against formally defined meta-properties "
    "drawn from foundational ontology. You return exactly one label "
    "per query: no explanation, no punctuation, no prefix."
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
    "about 'every occurrence', it asks whether the property holds "
    "uniformly across all instances admitted by the type under the "
    "above context."
)


# =====================================================================
# PROPERTY-DEFINITION BLOCKS (identical to DP)
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
# FSP EXEMPLAR BLOCKS
# Drawn from the canonical philosophical / linguistic literature on
# the corresponding meta-property; none is itself a MAVEN event type.
# Counts: Cumulativity 6, Atomicity 7, Homeomericity 5, Agentivity 6.
# =====================================================================

exemplar_blocks = {

    # -----------------------------------------------------------------
    # CUMULATIVITY: 6 exemplars (2 Cumulative, 4 Anti-Cumulative)
    # -----------------------------------------------------------------
    "Cumulativity": (
        "Examples for Cumulativity:\n\n"

        "Exemplar 1 (Cumulative, open-ended activity): Running.\n"
        "Take any two separate occurrences of running, for example, "
        "an episode of running from 09:00 to 09:30 and a second "
        "episode from 09:30 to 10:00. Their mereological sum is "
        "itself a single occurrence of running spanning 09:00 to "
        "10:00. The same relationship holds for two temporally "
        "non-adjacent episodes: their sum is again an occurrence of "
        "running at the day-level granularity. Running is open-ended; "
        "it has no built-in stopping point at which the activity is "
        "complete, so adding more running yields more running.\n"
        "Classification: cumulative\n\n"

        "Exemplar 2 (Cumulative, state persistence): Being-Seated.\n"
        "Take any two separate occurrences of being seated. The "
        "mereological sum of these two occurrences is itself a single "
        "occurrence of being seated, because the type denotes the "
        "persistence of a condition (the agent is in a seated "
        "configuration) and the sum of two occurrences is a longer "
        "period during which that condition holds. Being-Seated has "
        "no internal completion criterion; a state continues to be "
        "of the type as long as the configuration is maintained.\n"
        "Classification: cumulative\n\n"

        "Exemplar 3 (Anti-Cumulative, culmination): "
        "Running-Five-Kilometres.\n"
        "Take any two separate occurrences of running five kilometres "
        "that are mutually independent (neither is a part of the "
        "other; they are two distinct five-kilometre runs). Their "
        "mereological sum is not a single occurrence of running five "
        "kilometres, but a ten-kilometre run, which fails the type. "
        "The five-kilometre mark functions as a culmination point at "
        "which the event is complete; once that culmination is "
        "reached, continuing to run does not extend the same event "
        "but begins a new one.\n"
        "Classification: anti-cumulative\n\n"

        "Exemplar 4 (Anti-Cumulative, count-noun individuation): "
        "Addressing Congress.\n"
        "Take any two separate occurrences of addressing Congress, "
        "for example, two speeches given on different days in two "
        "distinct parliamentary proceedings. The mereological sum of "
        "these two occurrences is not a single occurrence of "
        "addressing Congress; it is two separate addresses, and the "
        "count of two cannot be reduced to a count of one without "
        "losing the identity of each as a distinct address. Each "
        "occurrence is associated with a distinct parliamentary "
        "proceeding, and the type counts proceedings rather than "
        "aggregating speech-time.\n"
        "Classification: anti-cumulative\n\n"

        "Exemplar 5 (Anti-Cumulative, categorical-bound outcome): "
        "Winning a Race.\n"
        "Take any two separate occurrences of winning a race that "
        "are mutually independent (two distinct race victories in "
        "two distinct races). The mereological sum of these two "
        "occurrences is not a single occurrence of winning a race, "
        "because winning entails a uniqueness condition (the type "
        "admits exactly one winner per race) that two independent "
        "victories cannot share.\n"
        "Classification: anti-cumulative\n\n"

        "Exemplar 6 (Anti-Cumulative, punctual transition): "
        "Crossing the Finish Line.\n"
        "Take any two separate occurrences of crossing the finish "
        "line, for example, a runner crossing the finish line in "
        "two different races. The mereological sum of these two "
        "occurrences is not a single occurrence of crossing the "
        "finish line, because the type denotes an instantaneous "
        "transition between two states. At the type level the event "
        "has no extent that admits aggregation; the sum of two "
        "transitions is two separate transitions, not a longer "
        "single transition.\n"
        "Classification: anti-cumulative"
    ),

    # -----------------------------------------------------------------
    # ATOMICITY: 7 exemplars (3 Atomic, 3 Anti-Atomic, 1 boundary)
    # -----------------------------------------------------------------
    "Atomicity": (
        "Examples for Atomicity:\n\n"

        "Exemplar 1 (Atomic, achievement): Reaching the Summit.\n"
        "Under the declared granularity, every occurrence of "
        "reaching the summit is a single boundary event with no "
        "proper temporal parts. One cannot decompose the act of "
        "reaching the summit into strictly smaller, identifiable "
        "constituent sub-events of the same type; the act is the "
        "transition from not-having-reached to having-reached, and "
        "that transition has no internal phases at the declared "
        "granularity.\n"
        "Classification: atomic\n\n"

        "Exemplar 2 (Atomic, decision-point): Selecting an Option.\n"
        "Under the declared granularity, every occurrence of "
        "selecting an option from a set is a single discrete "
        "cognitive act, treated as indivisible. Although a finer "
        "granularity might admit deliberation phases as sub-events, "
        "at the declared human-perceptual granularity the act has "
        "no proper sub-events that are themselves of the type.\n"
        "Classification: atomic\n\n"

        "Exemplar 3 (Atomic, state-entry): Beginning a Lecture.\n"
        "Under the declared granularity, every occurrence of "
        "beginning a lecture is the single act of entering into the "
        "state of lecturing, treated as a unitary transition rather "
        "than as the lecture itself. The event marks the entry into "
        "a state of being under way; it has no internal phases that "
        "are themselves lecture-beginnings.\n"
        "Classification: atomic\n\n"

        "Exemplar 4 (Anti-Atomic, phased accomplishment): An Ascent.\n"
        "Under the declared granularity, every occurrence of an "
        "ascent necessarily consists of distinguishable sub-phases: "
        "departure from the starting point, the ascent proper, and "
        "arrival at the destination. Each sub-phase is a strictly "
        "smaller, identifiable constituent of the whole ascent.\n"
        "Classification: anti-atomic\n\n"

        "Exemplar 5 (Anti-Atomic, distributed activity): "
        "Heated Argument.\n"
        "Under the declared granularity, every occurrence of a "
        "heated argument consists of identifiable sub-acts performed "
        "by multiple participants: assertions, rebuttals, "
        "interruptions, and reactions. These sub-acts are not "
        "necessarily sequential, and many are concurrent.\n"
        "Classification: anti-atomic\n\n"

        "Exemplar 6 (Anti-Atomic, composite process): "
        "Surgical Procedure.\n"
        "Under the declared granularity, every occurrence of a "
        "surgical procedure consists of multiple distinguishable "
        "strands of activity (anaesthesia, the surgical action "
        "itself, monitoring, instrument handling, post-operative "
        "stabilisation) coordinated by an organising purpose and "
        "proceeding in parallel rather than purely in sequence.\n"
        "Classification: anti-atomic\n\n"

        "Exemplar 7 (granularity-dependent boundary case): Explosion.\n"
        "At human-perceptual granularity, every occurrence of an "
        "explosion is treated as a single indivisible event without "
        "identifiable sub-phases, and the type is classified as "
        "atomic. At engineering-simulation granularity, every "
        "occurrence decomposes into strictly smaller sub-events "
        "(initiation, shock propagation, fireball expansion, "
        "pressure decay), and the same type is classified as "
        "anti-atomic. The classification depends on the declared "
        "granularity at the top of the prompt.\n"
        "Classification at human-perceptual granularity: atomic"
    ),

    # -----------------------------------------------------------------
    # HOMEOMERICITY: 5 exemplars (2 Homeomeric, 3 Anti-Homeomeric)
    # -----------------------------------------------------------------
    "Homeomericity": (
        "Examples for Homeomericity:\n\n"

        "Exemplar 1 (Homeomeric, state persistence): Being-Seated.\n"
        "For every occurrence of being seated, every continuous "
        "temporal sub-event of that occurrence is itself a complete "
        "occurrence of being seated. If a person is seated from "
        "14:00 to 16:00, then they are seated from 14:30 to 15:00 "
        "and during every other continuous sub-interval of the "
        "occurrence. The type denotes the persistence of a "
        "condition.\n"
        "Classification: homeomeric\n\n"

        "Exemplar 2 (Homeomeric, process activity): Running.\n"
        "For every occurrence of running, every continuous temporal "
        "sub-event of that occurrence is itself a complete occurrence "
        "of running, even though the runner's instantaneous physical "
        "configuration changes across the sub-event. Homeomericity "
        "holds at the running-event level even when finer-grained "
        "physical states do not, because the type is running, not "
        "an instantaneous physical configuration.\n"
        "Classification: homeomeric\n\n"

        "Exemplar 3 (Anti-Homeomeric, phase-distinct accomplishment): "
        "Building-a-House.\n"
        "For every occurrence of building a house, there is at least "
        "one continuous temporal sub-event that is not itself a "
        "complete occurrence of building a house. The "
        "foundation-laying phase is a foundation-laying event, the "
        "wall-construction phase is a wall-construction event, and "
        "the roofing phase is a roofing event; none of these "
        "sub-phases is itself a house-building event.\n"
        "Classification: anti-homeomeric\n\n"

        "Exemplar 4 (Anti-Homeomeric, goal-directed): "
        "Performing a Symphony.\n"
        "For every occurrence of performing a symphony, there is at "
        "least one continuous temporal sub-event that is not itself "
        "a complete occurrence of performing a symphony. A symphony "
        "performance has distinct movements, each of which is a "
        "movement performance rather than a symphony performance; "
        "the tuning that precedes the first movement is a tuning "
        "event, not a symphony performance.\n"
        "Classification: anti-homeomeric\n\n"

        "Exemplar 5 (Anti-Homeomeric, composite distributed "
        "activity): Surgical Procedure.\n"
        "For every occurrence of a surgical procedure, there is at "
        "least one continuous temporal sub-event that is not itself "
        "a complete occurrence of a surgical procedure. The strands "
        "of activity that constitute the procedure (anaesthesia, "
        "the surgical action itself, monitoring, instrument "
        "handling) are different in kind from the procedure as a "
        "whole.\n"
        "Classification: anti-homeomeric"
    ),

    # -----------------------------------------------------------------
    # AGENTIVITY: 6 exemplars (2 Agentive, 2 Anti-Agentive,
    # 2 Non-Agentive). The Building-a-Bridge / Volcanic-Eruption
    # pair is the embedded orthogonality contrast.
    # -----------------------------------------------------------------
    "Agentivity": (
        "Examples for Agentivity:\n\n"

        "Exemplar 1 (Agentive, constructive accomplishment): "
        "Building-a-Bridge.\n"
        "Every occurrence of building a bridge requires, by "
        "definition, a distinct intentional agent acting with the "
        "goal that the occurrence culminate in a completed bridge. "
        "The culminating part at which the agent's goal is realised "
        "is essential to the type. The mereological profile of "
        "building a bridge is anti-cumulative, anti-atomic, and "
        "anti-homeomeric.\n"
        "Classification: agentive\n\n"

        "Exemplar 2 (Agentive, performative act): "
        "Signing a Contract.\n"
        "Every occurrence of signing a contract requires, by "
        "definition, an intentional agent who performs the signing "
        "with the explicit purpose of binding themselves to the "
        "contract. The act is constitutively intentional: an "
        "unconscious or compelled motion that resembles signing is "
        "not an instance of the type. No physical artifact is "
        "produced; what is produced is an institutional fact.\n"
        "Classification: agentive\n\n"

        "Exemplar 3 (Anti-Agentive, natural process): "
        "Volcanic-Eruption.\n"
        "No occurrence of a volcanic eruption is intentionally "
        "initiated by an agent. Every occurrence arises from a "
        "natural geological process, with no intentional initiator "
        "and no goal-directed culmination. The mereological profile "
        "of volcanic eruption is also anti-cumulative, anti-atomic, "
        "and anti-homeomeric, identical to that of building a "
        "bridge. The agentivity classification therefore cannot be "
        "inferred from the mereological profile; it must be read "
        "from the definitional content of the type.\n"
        "Classification: anti-agentive\n\n"

        "Exemplar 4 (Anti-Agentive, reflexive biological): "
        "Heart-Beating.\n"
        "No occurrence of heart-beating is intentionally initiated "
        "by an agent at the type level. The type denotes an "
        "autonomously regulated biological process; an agent has "
        "no intentional control over the occurrence of the next "
        "beat. The presence of an animate participant does not "
        "make an event agentive: agentivity depends on intentional "
        "initiation, not on the kind of entity that hosts the "
        "event.\n"
        "Classification: anti-agentive\n\n"

        "Exemplar 5 (Non-Agentive, context-dependent): Journey.\n"
        "The event type journey admits both intentionally initiated "
        "occurrences (a deliberately planned trip with a destination "
        "in mind) and occurrences that are not intentionally "
        "initiated (an unplanned wandering, a forced relocation, a "
        "sleepwalking episode that crosses ground). Whether agency "
        "is present depends on the context of the particular "
        "occurrence; both kinds of occurrence are admitted by the "
        "type.\n"
        "Classification: non-agentive\n\n"

        "Exemplar 6 (Non-Agentive, type-ambiguous): "
        "Causing-Harm-to-Another.\n"
        "The event type causing harm to another is neutral on "
        "whether an agent intentionally initiated the occurrence. "
        "The type covers deliberate harm, negligent harm, and "
        "incidental harm. The type's definition does not entail "
        "intentional initiation, nor does it entail the absence of "
        "intentional initiation. The ambiguity is built into the "
        "type rather than being context-dependent at the occurrence "
        "level.\n"
        "Classification: non-agentive"
    ),
}


# =====================================================================
# TASK FOOTERS (per property)
# =====================================================================

footer_blocks = {
    "Cumulativity": (
        "Task. Using the same reasoning pattern as the exemplars, "
        "classify the above event type with respect to Cumulativity, "
        "applying the definition uniformly across all occurrences "
        "admitted under the declared interpretive context. Return "
        "exactly one of the valid labels, with no explanation.\n"
        "Valid answers: cumulative, anti-cumulative"
    ),
    "Atomicity": (
        "Task. Using the same reasoning pattern as the exemplars, "
        "classify the above event type with respect to Atomicity, "
        "applying the definition uniformly across all occurrences "
        "admitted under the declared interpretive context. Return "
        "exactly one of the valid labels, with no explanation.\n"
        "Valid answers: atomic, anti-atomic"
    ),
    "Homeomericity": (
        "Task. Using the same reasoning pattern as the exemplars, "
        "classify the above event type with respect to Homeomericity, "
        "applying the definition uniformly across all occurrences "
        "admitted under the declared interpretive context. Return "
        "exactly one of the valid labels, with no explanation.\n"
        "Valid answers: homeomeric, anti-homeomeric"
    ),
    "Agentivity": (
        "Task. Using the same reasoning pattern as the exemplars, "
        "classify the above event type with respect to Agentivity, "
        "applying the definition uniformly across all occurrences "
        "admitted under the declared interpretive context. Return "
        "exactly one of the valid labels, with no explanation.\n"
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

def construct_fsp_prompt(event_name: str,
                          event_definition: str,
                          meta_property: str) -> str:
    """Build the full FSP user-prompt for a target event."""
    event_block = (
        f"Event type to classify.\n"
        f"  Name: {event_name}\n"
        f"  Definition: {event_definition}"
    )
    return "\n\n".join([
        GRANULARITY_PREAMBLE,
        property_definitions[meta_property],
        exemplar_blocks[meta_property],
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


def query_fsp(event_name: str,
               event_definition: str,
               meta_property: str) -> str:
    """Issue one FSP classification call. Returns a normalised label."""
    try:
        prompt = construct_fsp_prompt(event_name, event_definition,
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
            label = query_fsp(event_type, definition, meta_property)
            df.at[i, meta_property] = label
            print(f"    {meta_property}: {label}")
            time.sleep(1)  # gentle rate-limiting

        # Save after each event so progress survives interruption.
        df.to_csv(output_path, index=False)

    total_calls = len(df) * len(meta_properties)
    print()
    print("FSP independent classification complete.")
    print(f"  Events processed: {len(df)}")
    print(f"  Total LLM calls:  {total_calls} "
          f"({len(df)} events x {len(meta_properties)} properties)")
    print(f"  Output:           {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Independent Few-Shot Prompting for "
                    "perdurant meta-property classification."
    )
    parser.add_argument(
        "--input",
        default="161_FrameNet.csv",
        help="Input CSV with EventType and Generic_Definition columns.",
    )
    parser.add_argument(
        "--output",
        default="161_FSP_independent.csv",
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
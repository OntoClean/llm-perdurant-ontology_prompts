"""
Metacognitive Prompting (MC) for perdurant meta-property
classification: INDEPENDENT EXECUTION (v2).

This script issues all four property-specific MC prompts
INDEPENDENTLY for every event, with no information passed
between prompts and no constraint forcing applied. Each
property is classified as a separate task, exactly four LLM
calls per event.

Each MC prompt is a single API call. The reflection block
inside the prompt instructs the model to produce a candidate
label silently, reconsider it against the property's
characteristic structural content, confirm or revise, and
return only the final post-reflection label.

Design changes from v1.
The v1 MC prompts had three concerns that the CoT v1 -> v2
empirical episode flagged as risks:
  1. The homeomericity reflection asked for an active witness
     search ("attempt to identify a continuous temporal
     sub-interval that is not itself a complete occurrence ...").
     Active-search instructions are the wording that destroyed
     CoT v1 atomicity and homeomericity. The v2 reflection
     reframes this step as a definitional-content check that
     consults a structural inventory of three anti-homeomeric
     patterns.
  2. The agentivity reflection had six steps, which is at the
     upper end of what GPT-4 reliably executes in a single
     pass without losing focus. The v2 reflection compresses
     the agent + initiation + culmination decomposition into
     a single integrated check, retaining the orthogonality
     reminder and the three-way classification, for four steps
     total.
  3. The cumulativity trigger check fired AFTER a candidate
     was already produced, functioning as confirmation bias
     for an Anti-Cumulative candidate. The v2 reflection
     reframes the trigger check as a positive structural
     inventory that the model consults before deciding.

Two smaller improvements based on the CoT v2 experience:
  - The granularity preamble's closing sentence is updated
    to characteristic-of-type framing, matching CoT v2.
  - The system message includes an additional instruction
    that the reflection is silent and only the final label
    is emitted.

This run is the MC equivalent of the independent Direct
Prompting baseline reported in Section
app:ch6-dp-results-independent of the appendix. The
stage-wise pipeline using IC1, IC2, and the contrapositive
of IC3 is a separate run produced by a different script.

Output CSV columns per event:
  EventType, Generic_Definition,
  Cumulativity, Atomicity, Homeomericity, Agentivity

Usage:
    1. Set the OPENAI_API_KEY environment variable:
           export OPENAI_API_KEY=sk-...    (macOS/Linux)
           setx OPENAI_API_KEY sk-...      (Windows, then reopen terminal)
    2. Run from a terminal:
           python mc_independent_local_v2.py \
               --input  /path/to/161_FrameNet.csv \
               --output /path/to/161_MC_independent_v2.csv

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
    "prompt asks for silent reflection before answering, perform "
    "the reflection internally and emit only the final label. You "
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
# PROPERTY-DEFINITION BLOCKS (identical to DP, FSP, CoT)
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
# MC REFLECTION BLOCKS (v2)
# Each block targets a specific failure mode documented in the DP
# independent run, applies a candidate-then-reflect-then-revise
# pattern, and uses characteristic-of-type framing throughout.
# =====================================================================

reflection_blocks = {

    # -----------------------------------------------------------------
    # CUMULATIVITY: 5 reflection steps
    #   Step 1: candidate
    #   Step 2: positive-profile check (open-ended or stative)
    #   Step 3: anti-cumulativity structural inventory
    #           (positive structural inventory, not confirmation bias)
    #   Step 4: confirm or revise
    #   Step 5: final label
    # -----------------------------------------------------------------
    "Cumulativity": (
        "Metacognitive reflection (silent). Carry out the following "
        "steps internally, in order. Do not include the candidate, "
        "the critique, or any reasoning trace in your output. Emit "
        "only the final label.\n\n"

        "Step 1 (candidate).\n"
        "Produce an initial candidate label for the event type's "
        "Cumulativity, drawing on the property definition above.\n\n"

        "Step 2 (reflection: positive-profile check).\n"
        "Now reconsider the candidate. Does the type's definition "
        "characteristically fit one of the two positive profiles? "
        "An open-ended activity has no internal stopping criterion, "
        "so combining two occurrences yields a longer occurrence of "
        "the same type (running, walking, talking). A persistent "
        "state holds across its duration, so combining two periods "
        "of the state yields a longer period of the state (being "
        "seated, holding, containing). If the type fits either "
        "profile, the type is Cumulative.\n\n"

        "Step 3 (reflection: anti-cumulativity structural "
        "inventory).\n"
        "Consult the following inventory of structural features that "
        "characteristically defeat cumulativity at the type level:\n"
        "  - culmination point at which an occurrence is complete "
        "(running five kilometres, reaching the summit, finishing a "
        "meal);\n"
        "  - event-external individuation criterion under which each "
        "occurrence is counted separately (addressing Congress "
        "counts proceedings, not speech-time);\n"
        "  - uniqueness condition built into the type's identity "
        "criteria (winning a race admits one winner per race);\n"
        "  - punctual character with no temporal extent (crossing "
        "the finish line, arriving at a destination).\n"
        "If any one of these features is characteristic of the type, "
        "the type is Anti-Cumulative.\n\n"

        "Step 4 (confirm or revise).\n"
        "Compare the candidate from Step 1 with the conclusions of "
        "Steps 2 and 3. If they contradict the candidate, revise the "
        "label. Otherwise, confirm the candidate.\n\n"

        "Step 5 (final label).\n"
        "Return one of: cumulative, anti-cumulative."
    ),

    # -----------------------------------------------------------------
    # ATOMICITY: 5 reflection steps
    #   Step 1: candidate
    #   Step 2: atomic-positive check (3 patterns)
    #   Step 3: anti-atomic check (3 patterns)
    #   Step 4: confirm or revise
    #   Step 5: final label
    # -----------------------------------------------------------------
    "Atomicity": (
        "Metacognitive reflection (silent). Carry out the following "
        "steps internally, in order. Do not include the candidate, "
        "the critique, or any reasoning trace in your output. Emit "
        "only the final label.\n\n"

        "Step 1 (candidate).\n"
        "Produce an initial candidate label for the event type's "
        "Atomicity, at the granularity declared in the "
        "interpretive-context preamble.\n\n"

        "Step 2 (reflection: atomic-positive check).\n"
        "Does the type's definition characteristically fit one of "
        "the three atomic patterns at the declared granularity?\n"
        "  - achievement: a single boundary transition (reaching "
        "the summit, arriving, winning);\n"
        "  - decision-point: a single discrete cognitive act "
        "(choosing, selecting, deciding);\n"
        "  - state-entry: the moment of entering a state "
        "(beginning, starting, becoming).\n"
        "If the type characteristically fits one of these three "
        "patterns, the type is Atomic.\n\n"

        "Step 3 (reflection: anti-atomic check).\n"
        "If Step 2 does not apply, does the type's definition "
        "characteristically entail identifiable internal phases "
        "or sub-acts that are themselves smaller events?\n"
        "  - phased accomplishments (preparation, execution, "
        "completion);\n"
        "  - distributed activities with concurrent sub-acts by "
        "multiple participants;\n"
        "  - composite processes with parallel coordinated strands.\n"
        "If the type characteristically fits one of these three "
        "patterns, the type is Anti-Atomic.\n\n"

        "Step 4 (confirm or revise).\n"
        "Compare the candidate from Step 1 with the conclusions of "
        "Steps 2 and 3. If they contradict the candidate, revise the "
        "label. Otherwise, confirm the candidate.\n\n"

        "Step 5 (final label).\n"
        "Return one of: atomic, anti-atomic."
    ),

    # -----------------------------------------------------------------
    # HOMEOMERICITY: 5 reflection steps
    # Step 3 reframed from active witness search to definitional
    # content check, matching CoT v2 wording.
    #   Step 1: candidate
    #   Step 2: homeomeric-positive check (state or process)
    #   Step 3: anti-homeomeric structural check (definitional, not
    #           a witness search)
    #   Step 4: confirm or revise
    #   Step 5: final label
    # -----------------------------------------------------------------
    "Homeomericity": (
        "Metacognitive reflection (silent). Carry out the following "
        "steps internally, in order. Do not include the candidate, "
        "the critique, or any reasoning trace in your output. Emit "
        "only the final label.\n\n"

        "Step 1 (candidate).\n"
        "Produce an initial candidate label for the event type's "
        "Homeomericity.\n\n"

        "Step 2 (reflection: homeomeric-positive check, run "
        "first).\n"
        "Before confirming an Anti-Homeomeric candidate, ask whether "
        "the type's definition characteristically fits one of these "
        "two homeomeric profiles.\n"
        "  - persistent state: the type denotes a condition that "
        "holds across the duration of an occurrence (being seated, "
        "holding, containing, surrounding, bearing). Every "
        "continuous sub-interval of such a state is itself an "
        "instance of the same state.\n"
        "  - homogeneous process: the type denotes an open-ended "
        "activity whose temporal sub-events are themselves of the "
        "same type at the activity level (running, walking, "
        "breathing, carrying). The type-level classification is "
        "uniform across continuous sub-intervals even when "
        "instantaneous physical configurations vary.\n"
        "If the type characteristically fits either profile, the "
        "type is Homeomeric. Stop reflection here.\n\n"

        "Step 3 (reflection: anti-homeomeric structural check).\n"
        "If Step 2 does not apply, does the type's definition "
        "characteristically entail phases that differ in kind from "
        "the whole?\n"
        "  - phase-distinct accomplishments (foundation, walls, and "
        "roof in building a house);\n"
        "  - goal-directed activities with preparation, execution, "
        "and culmination (preparing for, performing, and finishing "
        "a symphony);\n"
        "  - composite distributed activities whose strands are "
        "themselves of different types (anaesthesia, cutting, and "
        "monitoring in a surgical procedure).\n"
        "If the type characteristically fits one of these patterns, "
        "the type is Anti-Homeomeric.\n\n"

        "Step 4 (confirm or revise).\n"
        "Compare the candidate from Step 1 with the conclusions of "
        "Steps 2 and 3. If they contradict the candidate, revise the "
        "label. Otherwise, confirm the candidate.\n\n"

        "Step 5 (final label).\n"
        "Return one of: homeomeric, anti-homeomeric."
    ),

    # -----------------------------------------------------------------
    # AGENTIVITY: 4 reflection steps (compressed from v1's six)
    # Steps 1-3 of v1 (agent, initiation, culmination) are folded
    # into a single integrated definitional check at Step 2.
    #   Step 1: candidate
    #   Step 2: integrated definitional check + orthogonality
    #   Step 3: three-way classification check
    #   Step 4: final label
    # -----------------------------------------------------------------
    "Agentivity": (
        "Metacognitive reflection (silent). Carry out the following "
        "steps internally, in order. Do not include the candidate, "
        "the critique, or any reasoning trace in your output. Emit "
        "only the final label.\n\n"

        "Step 1 (candidate).\n"
        "Produce an initial candidate label for the event type's "
        "Agentivity.\n\n"

        "Step 2 (reflection: integrated definitional check with "
        "orthogonality).\n"
        "Reconsider the candidate by evaluating the type's "
        "definitional content against three integrated requirements: "
        "does the type characteristically require an enduring agent "
        "(a person, group, or organisation distinct from the event), "
        "who intentionally initiates the occurrence with a goal in "
        "mind, with a culminating part at which the goal is "
        "realised?\n"
        "Crucially, judge this from the type's definition, not from "
        "its mereological profile. Two event types with identical "
        "mereological profiles (the same combinations of "
        "cumulativity, atomicity, and homeomericity) can differ in "
        "agentivity. Building-a-bridge and volcanic-eruption have "
        "the same mereological profile but differ in agentivity, "
        "because agentivity depends on intentional initiation in the "
        "type's definition, not on decomposition or aggregation "
        "behaviour.\n\n"

        "Step 3 (reflection: three-way classification).\n"
        "If the integrated check in Step 2 holds characteristically "
        "of every occurrence admitted by the type, the type is "
        "Agentive. If the type's definition characteristically "
        "excludes intentional initiation (occurrences arise from "
        "natural, mechanical, or reflexive processes), the type is "
        "Anti-Agentive. If the type's definition is itself neutral "
        "on intentional initiation, or covers both deliberate and "
        "non-deliberate cases, the type is Non-Agentive. Compare "
        "this conclusion with the candidate from Step 1; revise on "
        "contradiction, confirm otherwise.\n\n"

        "Step 4 (final label).\n"
        "Return one of: agentive, anti-agentive, non-agentive."
    ),
}


# =====================================================================
# TASK FOOTERS (per property)
# =====================================================================

footer_blocks = {
    "Cumulativity": (
        "Task. Return exactly one of the valid labels, with no "
        "explanation and no reasoning trace. The label you return "
        "must be the post-reflection label, not the initial "
        "candidate.\n"
        "Valid answers: cumulative, anti-cumulative"
    ),
    "Atomicity": (
        "Task. Return exactly one of the valid labels, with no "
        "explanation and no reasoning trace. The label you return "
        "must be the post-reflection label, not the initial "
        "candidate.\n"
        "Valid answers: atomic, anti-atomic"
    ),
    "Homeomericity": (
        "Task. Return exactly one of the valid labels, with no "
        "explanation and no reasoning trace. The label you return "
        "must be the post-reflection label, not the initial "
        "candidate.\n"
        "Valid answers: homeomeric, anti-homeomeric"
    ),
    "Agentivity": (
        "Task. Return exactly one of the valid labels, with no "
        "explanation and no reasoning trace. The label you return "
        "must be the post-reflection label, not the initial "
        "candidate.\n"
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

def construct_mc_prompt(event_name: str,
                         event_definition: str,
                         meta_property: str) -> str:
    """Build the full MC user-prompt for a target event."""
    event_block = (
        f"Event type to classify.\n"
        f"  Name: {event_name}\n"
        f"  Definition: {event_definition}"
    )
    return "\n\n".join([
        GRANULARITY_PREAMBLE,
        property_definitions[meta_property],
        event_block,
        reflection_blocks[meta_property],
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


def query_mc(event_name: str,
              event_definition: str,
              meta_property: str) -> str:
    """Issue one MC classification call. Returns a normalised label."""
    try:
        prompt = construct_mc_prompt(event_name, event_definition,
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
            label = query_mc(event_type, definition, meta_property)
            df.at[i, meta_property] = label
            print(f"    {meta_property}: {label}")
            time.sleep(1)  # gentle rate-limiting

        # Save after each event so progress survives interruption.
        df.to_csv(output_path, index=False)

    total_calls = len(df) * len(meta_properties)
    print()
    print("MC independent classification complete (v2).")
    print(f"  Events processed: {len(df)}")
    print(f"  Total LLM calls:  {total_calls} "
          f"({len(df)} events x {len(meta_properties)} properties)")
    print(f"  Output:           {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Independent Metacognitive Prompting (v2) for "
                    "perdurant meta-property classification."
    )
    parser.add_argument(
        "--input",
        default="161_FrameNet.csv",
        help="Input CSV with EventType and Generic_Definition columns.",
    )
    parser.add_argument(
        "--output",
        default="161_MC_independent_v2.csv",
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
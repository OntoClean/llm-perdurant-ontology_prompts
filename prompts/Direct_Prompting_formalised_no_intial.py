import os
import time
import pandas as pd
from openai import OpenAI

# === OpenAI API key ===
# The key is read from the OPENAI_API_KEY environment variable.
# Set it before running, for example:
#     export OPENAI_API_KEY=sk-...    (macOS/Linux)
#     setx OPENAI_API_KEY sk-...      (Windows, then reopen terminal)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise SystemExit(
        "OPENAI_API_KEY environment variable is not set. "
        "Set it to your OpenAI API key before running."
    )
client = OpenAI(api_key=OPENAI_API_KEY)

# === Load the event data ===
df = pd.read_csv("161_FrameNet.csv", encoding="ISO-8859-1")
df = df.drop_duplicates(subset=["EventType", "Generic_Definition"])

# === Meta-properties — each prompted independently for every row ===
meta_properties = ["Cumulativity", "Atomicity", "Homeomericity", "Agentivity"]

# Add output columns if they don't already exist
for col in meta_properties:
    if col not in df.columns:
        df[col] = ""

# === Interpretive context preamble (shared across all prompts) ===
INTERPRETIVE_CONTEXT_PREAMBLE = (
    "You are a highly knowledgeable assistant tasked with classifying event types "
    "according to formal ontological meta-properties. Apply each property definition "
    "uniformly across all occurrences admitted under the event type's declared meaning. "
    "Base your classification solely on the definitional content of the event type."
)

# ===============================================================
# PROMPT DEFINITIONS (aligned to formal definitions in Ch. 5)
# ===============================================================

helper_blocks = {
    "Cumulativity": (
        "Property definition: Cumulativity.\n"
        "Cumulativity concerns whether the combination of two separate occurrences of the "
        "event type is itself a single occurrence of the same type. The two type-level "
        "outcomes are defined as follows.\n\n"
        "  Cumulative. For any two occurrences x and y of the event type, their "
        "mereological sum x+y (the aggregate formed by considering the two occurrences "
        "together) is also a single occurrence of the same event type.\n\n"
        "  Anti-Cumulative. For any two occurrences x and y of the event type that are "
        "mutually independent (that is, neither x is a part of y, nor y is a part of x), "
        "their mereological sum x+y is not a single occurrence of the same event type."
    ),

    "Atomicity": (
        "Property definition: Atomicity.\n"
        "Atomicity concerns whether occurrences of the event type can be decomposed into "
        "strictly smaller proper sub-events at the declared temporal granularity. The two "
        "type-level outcomes are defined as follows.\n\n"
        "  Atomic. Every occurrence of the event type has no proper parts at the declared "
        "granularity. A proper part is a strictly smaller, identifiable constituent "
        "sub-event of the occurrence.\n\n"
        "  Anti-Atomic. Every occurrence of the event type has at least one proper part "
        "at the declared granularity.\n\n"
        "Classification depends on the intended granularity of analysis and whether "
        "meaningful sub-intervals can be identified within the event."
    ),
    "Homeomericity": (
        "Property definition: Homeomericity.\n"
        "Homeomericity concerns whether the temporal sub-events of an "
        "occurrence (the smaller events that take place during continuous "
        "portions of the occurrence's duration) are themselves complete "
        "occurrences of the same event type. The two type-level outcomes "
        "are defined as follows.\n\n"
        "  Homeomeric. For every occurrence of the event type, every "
        "temporal sub-event of that occurrence (every event that takes "
        "place during a continuous portion of the occurrence's duration) "
        "is itself a complete occurrence of the same event type.\n\n"
        "  Anti-Homeomeric. For every occurrence of the event type, there "
        "is at least one temporal sub-event of that occurrence that is "
        "not itself a complete occurrence of the same event type. Common "
        "cases include preparation phases or culminating phases that "
        "differ in kind from the whole."
    ),

    "Agentivity": (
        "Property definition: Agentivity.\n"
        "Agentivity concerns whether occurrences of the event type are, by definition, "
        "intentionally initiated by an agent acting toward a culmination. An agent is a "
        "person, group, or organisation that is an enduring entity, distinct from the "
        "event itself. To intentionally initiate an event is for the agent to bring the "
        "event about with a goal in mind, and the event must contain a culminating part "
        "at which that goal is realised. The three type-level outcomes are defined as "
        "follows.\n\n"
        "  Agentive. Every occurrence of the event type is, by definition, intentionally "
        "initiated by some agent acting toward a culmination.\n\n"
        "  Anti-Agentive. No occurrence of the event type is intentionally initiated by "
        "an agent. The event arises through natural, mechanical, or reflexive processes.\n\n"
        "  Non-Agentive. The event type admits both intentionally initiated occurrences "
        "and occurrences that are not intentionally initiated. Whether agency is present "
        "depends on the context of the particular occurrence (for example, sneezing may "
        "be either voluntary or reflexive).\n\n"
        "Important. Judge agentivity on the definitional content of the event type itself. "
        "Do not infer agentivity from the event's mereological profile, that is, from "
        "whether the event is atomic, cumulative, or homeomeric. Two event types with "
        "identical mereological profiles can differ in agentivity."
    ),
}

footer_blocks = {
    "Cumulativity": (
        "Task. Classify the above event type with respect to Cumulativity, applying the "
        "definition uniformly across all occurrences admitted under the declared "
        "interpretive context. Return exactly one of the valid labels, with no explanation.\n"
        "Valid answers: cumulative, anti-cumulative"
    ),
    "Atomicity": (
        "Task. Classify the above event type with respect to Atomicity, applying the "
        "definition uniformly across all occurrences admitted under the declared "
        "interpretive context. Return exactly one of the valid labels, with no explanation.\n"
        "Valid answers: atomic, anti-atomic"
    ),
    "Homeomericity": (
        "Task. Classify the above event type with respect to Homeomericity, applying the "
        "definition uniformly across all occurrences admitted under the declared "
        "interpretive context. Return exactly one of the valid labels, with no explanation.\n"
        "Valid answers: homeomeric, anti-homeomeric"
    ),
    "Agentivity": (
        "Task. Classify the above event type with respect to Agentivity, applying the "
        "definition uniformly across all occurrences admitted under the declared "
        "interpretive context. Return exactly one of the valid labels, with no explanation.\n"
        "Valid answers: agentive, anti-agentive, non-agentive"
    ),
}


# ===============================================================
# PROMPT CONSTRUCTION AND QUERYING
# ===============================================================

def construct_prompt(event_name, definition, meta_property):
    event_block = (
        f"Event type to classify.\n"
        f"  Name: {event_name}\n"
        f"  Definition: {definition}"
    )
    return "\n\n".join([
        helper_blocks[meta_property],
        event_block,
        footer_blocks[meta_property],
    ])


def query_meta_property_label(event_name, definition, meta_property):
    try:
        prompt = construct_prompt(event_name, definition, meta_property)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": INTERPRETIVE_CONTEXT_PREAMBLE},
                {"role": "user", "content": prompt},
            ],
            max_tokens=20,
            temperature=0.2,
            top_p=1.0,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"[Label:{meta_property}] Error for '{event_name}': {e}")
        return "error"


# ===============================================================
# MAIN LOOP — flat, independent per-property prompting
# ===============================================================

for i, row in df.iterrows():
    event_type = row["EventType"]
    definition = row["Generic_Definition"]

    print(f"Processing [{i}]: {event_type}")

    for meta_property in meta_properties:
        label = query_meta_property_label(event_type, definition, meta_property)
        df.at[i, meta_property] = label
        print(f"  {meta_property}: {label}")
        time.sleep(1)  # gentle rate-limiting

    # Save after each row so progress is not lost on interruption
    df.to_csv("161_Direct_prompting.csv", index=False)

print("Meta-property classification completed and saved.")
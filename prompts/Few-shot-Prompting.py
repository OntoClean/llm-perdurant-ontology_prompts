import pandas as pd
import openai
import time

# === Set your OpenAI API key here ===
openai.api_key = "OPEN_AI_API_KEY"

# === Load the event data ===
df = pd.read_csv("161_FrameNet.csv", encoding="ISO-8859-1")
# Ensure correct deduplication
df = df.drop_duplicates(subset=["EventType", "Generic_Definition"])

# Add output columns if they don't exist
meta_properties = ["Cumulativity", "Homeomericity", "TemporalExtent", "Agentivity"]
for col in meta_properties + [f"{m}Justification" for m in meta_properties]:
    if col not in df.columns:
        df[col] = ""

# === Helper descriptions for each meta-property ===
helper_blocks = {
    "Cumulativity": (
        "Cumulativity assesses whether multiple instances of an event type can be aggregated into a larger whole that is itself still an instance of the same type.\n"
        "- **Cumulative**: A perdurant type is cumulative if the mereological sum of two or more of its instances also qualifies as an instance of the same type. This property ensures that both individual (possibly atomic) episodes and extended aggregated occurrences can coexist within the same event type, as long as summation preserves type identity.\n"
        "  - *Example*: 'Sitting' is cumulative — two sitting periods can be summed into a longer sitting.\n"
        "- **Anti-Cumulative**: A perdurant type is anti-cumulative if the mereological sum of two or more distinct, non-overlapping instances does not result in another instance of the same type. Such occurrences remain ontologically distinct and cannot be merged without violating the identity conditions of the event type.\n"
        "  - *Example*: 'A walk from Ponte dei Sospiri to Piazza San Marco' is anti-cumulative — combining two such walks does not result in a single instance of the same type due to differing origin-destination pairs."
    
      "**1. Cumulative Example — Walking**\n"
        "- Walking for 5 minutes in the morning and walking again in the evening can be summed into a broader occurrence of 'walking.'\n"
        "- **Classification:** Cumulative\n\n"
        "**2. Cumulative Example — Searching**\n"
        "- Searching for lost keys in the morning and again in the afternoon can be aggregated as a unified 'searching' event.\n"
        "- **Classification:** Cumulative\n\n"
        "**3. Anti-Cumulative Example — Addressing Congress**\n"
        "- Addressing Congress three separate times must be enumerated ('addressed Congress three times').\n"
        "- Cannot simply say 'addressed Congress all year' without losing important distinctions.\n"
        "- **Classification:** Anti-Cumulative\n\n"
        "**4. Anti-Cumulative Example — Deciding**\n"
        "- Making one decision about a vacation and another about a career move are distinct decisions. They cannot be summed into one 'deciding' occurrence without changing the nature of the events.\n"
        "- **Classification:** Anti-Cumulative"
    
    
    ),

    "Homeomericity": (
        "Homeomericity evaluates whether every temporal part of an event instance is also an instance of the same event type. \n"
        "- **Homeomeric**: A perdurant type is homeomeric if all of its temporal subparts are instances of the same type. This reflects uniformity across all temporal intervals. Atomic events are trivially homeomeric, since they lack proper temporal parts.\n"
        "  - *Example*: 'Sitting' is homeomeric — every temporal slice of sitting is still sitting.\n"
        "- **Anti-Homeomeric**: A perdurant type is anti-homeomeric if at least one of its proper temporal parts fails to be an instance of the same type. This typically occurs in durative events with heterogeneous or structurally distinct subactivities.\n"
        "  - *Example*: 'Running a marathon' is anti-homeomeric — shorter intervals may involve warm-ups, hydration, or walking, which are not running in a strict sense."
   
            "**1. Homeomeric Example — Being Seated**\n"
            "- Dividing the time someone is seated (e.g., morning, afternoon) still results in each subinterval being an instance of 'being seated.'\n"
            "- **Classification:** Homeomeric\n\n"

            "**2. Trivially Homeomeric Example — Coming to the Finishing Line**\n"
            "- 'Coming to the Finishing Line' is an instantaneous atomic event — it has no meaningful temporal substructure, so it is trivially homeomeric.\n"
            "- **Classification:** Trivially Homeomeric\n\n"

            "**3. Anti-Homeomeric Example — Paying Attention**\n"
            "- During a lecture, subactivities like looking at slides, listening, and note-taking are distinct and not identical to the full act of 'paying attention.'\n"
            "- **Classification:** Anti-Homeomeric" 
   
    
   
    ),

    "TemporalExtent": (
        "Temporal extent evaluates whether an event type is conceptually treated as a temporally indivisible occurrence (atomic) or as one that unfolds over time with internal stages (durative).\n"
        "- **Atomic**: An event is atomic if it is temporally minimal and lacks proper temporal parts at the level of analysis. Such events are considered indivisible and are typically instantaneous or treated as such in context.\n"
        "  - *Example*: 'Death' or 'Reaching the summit' are atomic — they occur at a single point in time.\n"
        "- **Durative**: An event is durative if it extends in time and can be segmented into distinct phases or stages. These events have temporal parts and are characterized by their unfolding nature.\n"
        "  - *Example*: 'A conference' or 'A performance' are durative — they span over a period and involve internal sequences like sessions or acts."
            
            "**1. Atomic Example — Snapping Fingers**\n"
            "- 'Snapping fingers' happens as a single, indivisible action without meaningful internal stages, regardless of how finely the time is divided.\n"
            "- **Classification:** Atomic\n\n"

            "**2. Durative Example — Building a House**\n"
            "- 'Building a house' unfolds over time with identifiable phases such as planning, foundation work, construction, and finishing, each contributing to the overall event.\n"
            "- **Classification:** Durative\n\n"

            "**3. Atomic Example — Signing a Contract**\n"
            "- 'Signing a contract' is treated as a unified act where internal hand movements are not meaningfully separable at the relevant conceptual granularity.\n"
            "- **Classification:** Atomic\n\n"

            "**4. Durative Example — Learning a Language**\n"
            "- 'Learning a language' progresses through multiple identifiable stages (e.g., vocabulary acquisition, grammar mastery), making it temporally extended with internal structure.\n"
            "- **Classification:** Durative."
    
    
    
    
    ),

    "Agentivity": (
        "Agentivity refers to whether an event is initiated intentionally by an agent with a specific goal or purpose.\n"
        "- **Agentive**: The event is intentionally initiated by an agent, driven by a clear purpose or goal.\n"
        "  - *Example*: 'Writing', 'Organizing a conference', or 'Running a company' are agentive — they require purposeful initiation.\n"
        "- **Non-Agentive**: The event may involve intentional action, but it does not require goal-directed initiation in all instances. It can also occur naturally without purposeful intervention.\n"
        "  - *Example*: 'Being red' or 'Being open' are non-agentive — although they involve states, they don't require an initiating agent.\n"
        "- **Anti-Agentive**: The event occurs entirely due to external forces or natural processes, with no intentional action by an agent.\n"
        "  - *Example*: 'Rainfall' or 'Erosion' (if considered as events in a broader domain) would be anti-agentive — caused by natural forces without agency."
   
               "**1. Agentive Example — Signing a Petition**\n"
            "- 'Signing a petition' is an event that is initiated intentionally by an agent (the person signing) with a clear goal to express support for a cause. The agent deliberately initiates the event.\n"
            "- **Classification:** Agentive\n\n"

            "**2. Non-Agentive Example — Breathing**\n"
            "- 'Breathing' occurs naturally, but it can be intentional (e.g., deep breathing for relaxation) or involuntary (e.g., when asleep). While the action is often intentional, it is not always initiated with a clear goal.\n"
            "- **Classification:** Non-Agentive\n\n"

            "**3. Anti-Agentive Example — Lightning Strike**\n"
            "- 'Lightning strike' occurs entirely due to natural atmospheric processes. There is no agent or intentional action involved, as the event is a natural, uncontrolled phenomenon.\n"
            "- **Classification:** Anti-Agentive"
   
   
    )
}


footer_blocks = {
    "Cumulativity": (
        "Return only the one correct without any label or explanation.\n"
        "Valid answers are one of:\n"
        "- cumulative, anti-cumulative"
    ),
    "Homeomericity": (
        "Return only the one correct without any label or explanation.\n"
        "Valid answers are one of:\n"
        "- homeomeric, anti-homeomeric"
    ),
    "TemporalExtent": (
        "Return only the one correct without any label or explanation.\n"
        "Valid answers are one of:\n"
        "- durative, atomic"
    ),
    "Agentivity": (
        "Return only the one correct without any label or explanation.\n"
        "Valid answers are one of:\n"
        "- agentive, non-agentive, anti-agentive"
    )
    }
# === Prompt Construction with Definition ===
def construct_prompt(definition, meta_property):
    story = f"The goal is to classify the meta-property '{meta_property}' for an event defined as:\n{definition}"
    prompt = f"{helper_blocks[meta_property]}\n\n{story}\n\n{footer_blocks[meta_property]}"
    return prompt


# === Query GPT for Label ===
def query_meta_property_label(definition, meta_property):
    try:
        prompt = construct_prompt(definition, meta_property)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system", "content": "You are a highly knowledgeable assistant tasked with classifying events based on specific meta-properties."
            }, {
                "role": "user", "content": prompt
            }],
            max_tokens=20,
            temperature=0.2,
            top_p=1.0
        )
        return response['choices'][0]['message']['content'].strip().lower()
    except Exception as e:
        print(f"[Label:{meta_property}] Error for definition: {e}")
        return "error"


# === Query GPT for Justification ===
def query_meta_property_justification(definition, meta_property, label):
    try:
        explanation_prompt = (
            f"{helper_blocks[meta_property]}\n\nDefinition: {definition}\nAssigned Value: {label}\n"
            f"Provide a one-sentence reason why the event is labeled '{label}'."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system", "content": "You are an reasoning expert providing justifications for event classifications."
            }, {
                "role": "user", "content": explanation_prompt
            }],
            max_tokens=150,
            temperature=0.2,
            top_p=1.0
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[Justification:{meta_property}] Error: {e}")
        return "error"


# === Main Loop with Meta-property Tagging and Justification ===
for i, row in df.iterrows():
    event_type = row['EventType']  # Keep the event type for output
    definition = row['Generic_Definition']

    print(f"Processing definition: {definition}")

    for meta_property in meta_properties:
        # Get the label from GPT using definition
        label = query_meta_property_label(definition, meta_property)
        
        # Get the justification from GPT using definition
        #justification = query_meta_property_justification(definition, meta_property, label)
        
        # Store results in DataFrame
        df.at[i, 'EventType'] = event_type  # Ensure event type is stored in the output
        df.at[i, meta_property] = label
        #df.at[i, f"{meta_property}Justification"] = justification
        
        # Sleep for a second to avoid rate limiting
        time.sleep(1)

    # Save after each row
    df.to_csv("161_FewShot_prompting.csv", index=False)



print("Meta-property classification completed and saved.")


print("Meta-property classification completed and saved.")

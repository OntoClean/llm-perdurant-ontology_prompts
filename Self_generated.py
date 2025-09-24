import pandas as pd
import openai
import time

# === Set your OpenAI API key here ===
openai.api_key = "OPEN_AI_API_KEY"

# === Load the event data ===
df = pd.read_csv("MAVEN_Generic_Defintion_DataSet.csv", encoding="ISO-8859-1")
# Ensure correct deduplication
df = df.drop_duplicates(subset=["EventType", "Generic_Definition"])

# Add output columns if they don't exist
meta_properties = ["Cumulativity", "Homeomericity", "TemporalExtent", "Agentivity"]
for col in meta_properties + [f"{m}Justification" for m in meta_properties]:
    if col not in df.columns:
        df[col] = ""
helper_blocks = {
    "Cumulativity": (
        "Cumulativity assesses whether multiple instances of an event type can be aggregated into a larger whole that is itself still an instance of the same type.\n"
        "- **Cumulative**: A perdurant type is cumulative if the mereological sum of two or more of its instances also qualifies as an instance of the same type. This property ensures that both individual (possibly atomic) episodes and extended aggregated occurrences can coexist within the same event type, as long as summation preserves type identity.\n"
        "- **Anti-Cumulative**: A perdurant type is anti-cumulative if the mereological sum of two or more distinct, non-overlapping instances does not result in another instance of the same type. Such occurrences remain ontologically distinct and cannot be merged without violating the identity conditions of the event type."
    ),


"Homeomericity": (
    "Homeomericity evaluates whether every temporal part of an event instance is also an instance of the same event type. \n"
    "- **Homeomeric**: A perdurant type is homeomeric if all of its temporal subparts are instances of the same type. This reflects uniformity across all temporal intervals. Atomic events are trivially homeomeric, since they lack proper temporal parts.\n"
    "- **Anti-Homeomeric**: A perdurant type is anti-homeomeric if at least one of its proper temporal parts fails to be an instance of the same type. This typically occurs in durative events with heterogeneous or structurally distinct subactivities. Granularity and contextual interpretation are crucial in determining this distinction."
),


"TemporalExtent": (
    "Temporal extent evaluates whether an event type is conceptually treated as a temporally indivisible occurrence (atomic) or as one that unfolds over time with internal stages (durative).\n"
    "- **Atomic**: An event is atomic if it is temporally minimal and lacks proper temporal parts at the level of analysis. Such events are considered indivisible and are typically instantaneous or treated as such in context.\n"
    "- **Durative**: An event is durative if it extends in time and can be segmented into distinct phases or stages. These events have temporal parts and are characterized by their unfolding nature.\n"
    "Classification depends on the intended granularity of analysis and whether meaningful sub-intervals can be identified within the event."
),


    "Agentivity": (
        "Agentivity refers to whether an event is initiated intentionally by an agent with a specific goal or purpose.\n"
        "- **Agentive**: The event is intentionally initiated by an agent, driven by a clear purpose or goal.\n"
        "- **Non-Agentive**: The event may involve intentional action, but it does not require goal-directed initiation in all instances. It can also occur naturally without purposeful intervention.\n"
        "- **Anti-Agentive**: The event occurs entirely due to external forces or natural processes, with no intentional action by an agent.\n"
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

# === Self-Generated Example Prompt Construction ===
def construct_self_generated_prompt(definition, meta_property):
    story = f"The goal is to classify the meta-property '{meta_property}' based on the following definition:\n{definition}"

    self_generated_prompt = f"""
Please generate an example event that demonstrates the following meta-property: '{meta_property}'.
- For **Cumulativity**: 
    - **Cumulative**: A generated example where repeated instances can be meaningfully merged into a single, unified occurrence.
    - **Anti-Cumulative**: A generated example where repeated instances yield distinct sub-events or episodes, which cannot be merged into a single event.
- For **Homeomericity**: 
    - **Homeomeric**: A generated example where all temporal parts of the event are instances of the same type as the whole event, with no further internal breakdown.
    - **Anti-Homeomeric**: A generated example where distinct subparts deviate in structure or role, not maintaining the identity of the whole event.
- For **Temporal Extent**: 
    - **Atomic**: A generated example where the event occurs as a single, indivisible action, without meaningful internal subparts.
    - **Durative**: A generated example where the event stretches over time and consists of identifiable sub-events or phases.
- For **Agentivity**:
    - **Agentive**: A generated example where the event is initiated with purposeful action by an agent with a clear goal.
    - **Non-Agentive**: A generated example where the event may occur naturally or initiated with an intentional action.
    - **Anti-Agentive**: A generated example where the event happens entirely due to external forces or natural processes, without any intentional agent involvement.

Generate examples for both of these categories, making sure they demonstrate the key distinctions.
Compare both examples in terms of the meta-property '{meta_property}'.

{helper_blocks[meta_property]}

Definition: {definition}

{footer_blocks[meta_property]}

{story}
"""
    return self_generated_prompt


# === Query GPT for Self-Generated Example and Label ===
def query_meta_property_label_with_self_generated_example(definition, meta_property):
    try:
        prompt = construct_self_generated_prompt(definition, meta_property)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system", "content": "You are a highly knowledgeable assistant tasked with generating examples and classifying events based on specific meta-properties."
            }, {
                "role": "user", "content": prompt
            }],
            max_tokens=100,
            temperature=0.7,
            top_p=0.8
        )
        return response['choices'][0]['message']['content'].strip().lower()
    except Exception as e:
        print(f"[Label:{meta_property}] Error for definition: {e}")
        return "error"


# === Query GPT for Justification with Self-Generated Example ===
def query_meta_property_justification_with_self_generated_example(definition, meta_property, label):
    try:
        explanation_prompt = (
            f"{helper_blocks[meta_property]}\n\nEvent Definition: {definition}\nAssigned Value: {label}\n"
            f"Provide a single sentence justification for why this event is assigned the value '{label}' for the meta-property '{meta_property}'."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system", "content": "You are an ontology expert providing justifications for event classifications."
            }, {
                "role": "user", "content": explanation_prompt
            }],
            max_tokens=150,
            temperature=0.2,
            top_p=1.0
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[Justification:{meta_property}] Error for '{event_type}': {e}")
        return "error"


# === Main Loop with Self-Generated Example Prompting ===
for i, row in df.iterrows():
    event_type = row['EventType']  # Keep the event type for output
    definition = row['Generic_Definition']

    print(f"Processing event: {event_type}")

    for meta_property in meta_properties:
        # Get the label from GPT using Self-Generated Example Prompting
        label = query_meta_property_label_with_self_generated_example(definition, meta_property)
        
        # Get the justification from GPT using Self-Generated Example Prompting
        justification = query_meta_property_justification_with_self_generated_example(definition, meta_property, label)
        
        # Store results in DataFrame
        df.at[i, 'EventType'] = event_type  # Ensure event type is stored in the output
        df.at[i, meta_property] = label
        df.at[i, f"{meta_property}Justification"] = justification
        
        # Sleep for a second to avoid rate limiting
        time.sleep(1)

    # Save after each row
    df.to_csv("Self_generated_prompting_taggings_MAVEN_Generic_Defintion_DataSet.csv", index=False)

print("Meta-property classification completed and saved.")

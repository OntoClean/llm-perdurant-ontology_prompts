import pandas as pd
import openai
import time
import itertools


# === Set your OpenAI API key here ===
openai.api_key = "OPEN_AI_API_KEY"

# === Load the event data ===
df = pd.read_csv("161_FrameNet.csv", encoding="ISO-8859-1")
# # Ensure correct deduplication
# df = df.drop_duplicates(subset=["EventType", "Generic_Definition"])

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

# === Meta-Cognitive Chain-of-Thought (CoT) Questions ===
meta_cognitive_questions = {
    "Step 1": "Begin by thoroughly understanding the meta-property assignment options. Look at the choices available for classification. Carefully analyze each option to fully comprehend its meaning and criteria.",
    
    "Step 2": "Once you understand the assignment options, make an initial identification. Classify the event based on the most straightforward match from the available choices. Use the key characteristics of the event to guide this initial choice.",
    
    "Step 3": "Critically Assess Your Preliminary Choice. Reflect on the event and the classification you have chosen. Ask yourself: Does this classification align well with the characteristics of the event? Are there any nuances that suggest a different classification?",
    
    "Step 4": "Reassess Your Initial Choice (If Unsure). If you're uncertain about your preliminary choice, re-evaluate the event and consider other perspectives. Ask: Could the event be classified differently based on alternative interpretations or additional context?",
    
    "Step 5": "Confirm the Final Answer. Once you’ve critically reassessed your choice, confirm your final classification. Make sure it reflects the event’s characteristics accurately and justifiably. If you're still unsure, revisit the assignment options"
}

# === Prompt Construction with Chain-of-Thought ===
def construct_prompt_with_CoT(definition, meta_property):
    story = f"The goal is to classify the meta-property '{meta_property}' for an event defined as:\n{definition}"
    
    # Adjust the structure for Chain-of-Thought (CoT) prompting
    CoT_prompt = f"""
Let's carefully analyze the meta-property: {meta_property}.

{story}

{helper_blocks[meta_property]}
{meta_cognitive_questions}
{footer_blocks[meta_property]}


"""
    return CoT_prompt


# === Query GPT for Label with CoT Prompt ===
def query_meta_property_label_with_CoT(definition, meta_property):
    try:
        prompt = construct_prompt_with_CoT(definition, meta_property)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system", "content": "You are a highly knowledgeable assistant tasked with classifying events based on specific meta-properties using Chain-of-Thought."
            }, {
                "role": "user", "content": prompt
            }],
            max_tokens=20,
            temperature=0.2,
            top_p=0.6
        )
        return response['choices'][0]['message']['content'].strip().lower()
    except Exception as e:
        print(f"[Label:{meta_property}] Error for definition: {e}")
        return "error"


# # === Query GPT for Justification with Chain-of-Thought Prompt ===
# def query_meta_property_justification_with_CoT(definition, meta_property, label):
#     try:
#         explanation_prompt = (
#             f"{helper_blocks[meta_property]}\n\nDefinition: {definition}\nAssigned Value: {label}\n"
#             f"Provide a one sentence justification for why this event is assigned the value '{label}' for the meta-property '{meta_property}'."
#         )
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[{
#                 "role": "system", "content": "You are an ontology expert providing justifications for event classifications."
#             }, {
#                 "role": "user", "content": explanation_prompt
#             }],
#             max_tokens=150,
#             temperature=0.2,
#             top_p=1.0
#         )
#         return response['choices'][0]['message']['content'].strip()
#     except Exception as e:
#         print(f"[Justification:{meta_property}] Error: {e}")
#         return "error"


# === Main Loop with Chain-of-Thought Prompting ===
for i, row in df.iterrows():
    event_type = row['EventType']  # Keep the event type for output
    definition = row['Generic_Definition']

    print(f"Processing definition: {definition}")

    for meta_property in meta_properties:
        # Get the label from GPT using CoT
        label = query_meta_property_label_with_CoT(definition, meta_property)
        
        # Get the justification from GPT using CoT
        #justification = query_meta_property_justification_with_CoT(definition, meta_property, label)
        
        # Store results in DataFrame
        df.at[i, 'EventType'] = event_type  # Ensure event type is stored in the output
        df.at[i, meta_property] = label
        #df.at[i, f"{meta_property}Justification"] = justification
        
        # Sleep for a second to avoid rate limiting
        time.sleep(1)


    # Save after each row
    df.to_csv("161_meta_cognitive_prompting.csv", index=False)




print("Meta-property classification completed and saved.")

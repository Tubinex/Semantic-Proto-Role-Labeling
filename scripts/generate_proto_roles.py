"""Assign gold proto-role labels to SPR1 annotations

The property groupings below are based on Dowty:
- Proto-agent properties are associated with the causal, volitional participant.
- Proto-patient properties are associated with the affected, theme-like participant.

Output: spr1_with_gold.json, the same structure as spr1.json with an added
``gold_role`` field per annotation (1 = proto-agent, 0 = proto-patient).
"""
import json

INPUT_FILE = "../data/spr1.json"
OUTPUT_FILE = "../data/spr1_with_gold.json"

# Properties that are entailments for the Proto-Agent role according to Dowty
PROTO_AGENT_PROPERTIES = {
    "awareness",
    "instigation",
    "sentient",
    "volition",
    "makes_physical_contact",
    "existed_before",
    "exists_as_physical"

}

# Properties that are entailments for the Proto-Patient role according to Dowty
PROTO_PATIENT_PROPERTIES = {
    "change_of_state",
    "change_of_location",
    "created",
    "destroyed",
    "manipulated_by_another",
    "stationary",
    "predicate_changed_argument",
    "changes_possession",
    "existed_after",
    "existed_during",
    "location_of_event"

}

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

for entry_id, annotations in data.items():

    for ann in annotations:

        categories = ann["cat"]
        labels = ann["label"]
        applicable = ann["applicable"]

        agent_score = 0
        patient_score = 0

        for i, prop in enumerate(categories):

            # SPR1 marks some properties as inapplicable when the property
            # cannot meaningfully hold for this argument type.
            # These annotations are excluded from the score sum to 
            # avoid unfairly penalising an argument for properties
            # that are impossible
            if not applicable[i]:
                continue

            # Raw SPR1 Likert ratings are summed within each property cluster. 
            # Summing rather than averaging preserves the contribution
            # of each applicable property without discarding the effect of
            # having more or fewer applicable properties on one side.
            value = int(labels[i])

            if prop in PROTO_AGENT_PROPERTIES:
                agent_score += value

            elif prop in PROTO_PATIENT_PROPERTIES:
                patient_score += value

        # gold_role encoding: 1 = proto-agent, 0 = proto-patient.
        # Determined by summing raw SPR1 Likert ratings across
        # proto-agent vs proto-patient properties (inapplicable skipped).
        if agent_score > patient_score:
            ann["gold_role"] = 1
        else:
            ann["gold_role"] = 0

with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=2)

print("Finished. Output written to", OUTPUT_FILE)
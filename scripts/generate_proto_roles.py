import json

INPUT_FILE = "../data/spr1.json"
OUTPUT_FILE = "../data/spr1_with_gold.json"

PROTO_AGENT_PROPERTIES = {
    "awareness",
    "instigation",
    "sentient",
    "volition",
    "makes_physical_contact",
    "existed_before",
    "exists_as_physical"

}

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

            if not applicable[i]:
                continue

            value = int(labels[i])

            if prop in PROTO_AGENT_PROPERTIES:
                agent_score += value

            elif prop in PROTO_PATIENT_PROPERTIES:
                patient_score += value

        if agent_score > patient_score:
            ann["gold_role"] = 1
        else:
            ann["gold_role"] = 0

with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=2)

print("Finished. Output written to", OUTPUT_FILE)
import json
from collections import defaultdict
import numpy as np

from proto_roles import PROTO_AGENT_PROPERTIES, PROTO_PATIENT_PROPERTIES

def group_input(filepath):
    """
    Read input JSONL file and group entries by spr_id.
    Returns: dict {spr_id: {property: probability}}
    """
    grouped = defaultdict(dict)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            spr_id = item["spr_id"]
            property = item["property"]
            prob = item["p_entailment"]

            grouped[spr_id][property] = prob
        
    return grouped

def compute_score(property_dict, property_list):
    """
    Compute mean probability for given property group.
    """
    values = [property_dict.get(prop,0.0) for prop in property_list]
    return np.mean(values)

def predict_one(property_dict):
    """
    Predict proto-role for a single argument.
    """
    agent_score = compute_score(property_dict, PROTO_AGENT_PROPERTIES)
    patient_score = compute_score(property_dict, PROTO_PATIENT_PROPERTIES)

    if agent_score > patient_score:
        prediction = "proto-agent"
    else:
        prediction = "proto-patient"
    
    return prediction, agent_score, patient_score

def predict_all(grouped_dict):
    """
    Predict proto-role for all arguments.
    """
    results = {}

    for spr_id, property_dict in grouped_dict.items():
        prediction, agent_score, patient_score = predict_one(property_dict)

        results[spr_id] = {
            "prediction": prediction,
            "agent_score": round(agent_score, 4),
            "patient_score": round(patient_score, 4),
            "decision_margin": round(abs(agent_score - patient_score), 4),
        }
    
    return results

def save_predictions(results, output_path):
    """
    Save predictions as JSONL.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for spr_id, result in results.items():
            output_entry = {
                "spr_id": spr_id,
                **result
            }
            f.write(json.dumps(output_entry) + "\n")



# # MAIN

# if __name__ == "__main__":
#     input_file = "predictions.jsonl"
#     output_file = "proto_role_naive_pred.jsonl"

#     grouped_data = group_input(input_file)

#     results = predict_all(grouped_data)

#     save_predictions(results, output_file)



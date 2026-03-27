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
    Compute mean probability for a given property group.
    """
    # Missing properties default to 0.0 (no entailment evidence), which slightly
    # biases the mean downward when a property was not probed for this argument.
    values = [property_dict.get(prop,0.0) for prop in property_list]
    return np.mean(values)

def predict_one(property_dict):
    """
    Predict the proto-role for a single argument

    This is the naive baseline, no training is involved. 
    Proto-agent and proto-patient scores are computed as the mean 
    entailment probability across their respective property groups, 
    and the role with the higher mean is assigned. 
    
    The ``decision_margin`` represents how confidently the model separated the two roles.
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

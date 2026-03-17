import json
from collections import defaultdict
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder




def load_probing(filepath):
    """
    Load probing predictions and group them by spr_id.
    """

    grouped = defaultdict(dict)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            spr_id = item["spr_id"]
            prop = item["property"]

            grouped[spr_id][prop] = item["p_entailment"]
            

    return grouped


def load_gold_roles(filepath):
    """
    Load gold proto roles from SPRL dataset.
    """

    gold_roles = {}

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for spr_id, entries in data.items():
        entry = entries[0]
        role = entry["gold_role"]

        if role == 1:
            gold_roles[spr_id] = "proto-agent"
        else:
            gold_roles[spr_id] = "proto-patient"
    
    return gold_roles


def build_dataset(probing_data, gold_roles):
    """
    Load gold proto roles from SPRL dataset.
    """

    X_dicts = []
    y_labels = []
    spr_ids = []

    for spr_id in probing_data:
        if spr_id not in gold_roles:
            continue

        X_dicts.append(probing_data[spr_id])
        y_labels.append(gold_roles[spr_id])
        spr_ids.append(spr_id)

    return X_dicts, y_labels, spr_ids




class ProtoLabelClassifier:

    def __init__(self, property_list):
        self.property_list = property_list
        self.model = LogisticRegression(max_iter=1000)
        self.label_encoder = LabelEncoder()

    def _dicts_to_matrix(self, X_dicts):
        """
        Converts list of feature dictionaries into numpy matrix.
        """
        matrix = []

        for sample in X_dicts:
            row = [sample.get(prop, 0.0) for prop in self.property_list]
            matrix.append(row)

        return np.array(matrix)

    def train(self, X_dicts, y_labels):
        X = self._dicts_to_matrix(X_dicts)
        y = self.label_encoder.fit_transform(y_labels)
        self.model.fit(X, y)

    def predict_with_proba(self, X_dicts):
        X = self._dicts_to_matrix(X_dicts)

        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        labels = self.label_encoder.inverse_transform(y_pred)

        return labels, y_proba




def save_predictions(filepath, spr_ids, y_true, y_pred, y_proba, label_encoder):

    class_order = label_encoder.classes_

    with open(filepath, "w", encoding="utf-8") as f:

        for spr_id, gold, pred, probs in zip(spr_ids, y_true, y_pred, y_proba):

            output = {
                "spr_id": spr_id,
                "gold_proto_role": gold,
                "predicted_proto_role": pred,
                "is_correct": gold == pred
            }

            for class_name, prob in zip(class_order, probs):
                output[f"p_{class_name}"] = float(prob)

            f.write(json.dumps(output) + "\n")




# if __name__ == "__main__":

#     probing_file = "predictions.jsonl"
#     gold_file = "spr1_with_gold.json"

#     output_file = "proto_role_predictions.jsonl"

#     # load probing predictions
#     probing_data = load_probing(probing_file)

#     # load gold roles
#     gold_roles = load_gold_roles(gold_file)

#     # build dataset
#     X_dicts, y_labels, spr_ids = build_dataset(probing_data, gold_roles)

#     # define consistent property order
#     property_list = sorted(list({
#         prop
#         for sample in X_dicts
#         for prop in sample.keys()
#     }))

#     # initialize classifier
#     classifier = ProtoLabelClassifier(property_list)

#     # train model
#     classifier.train(X_dicts, y_labels)

#     # predict
#     y_pred, y_proba = classifier.predict_with_proba(X_dicts)

#     # save results
#     save_predictions(
#         output_file,
#         spr_ids,
#         y_labels,
#         y_pred,
#         y_proba,
#         classifier.label_encoder
#     )

#     print(f"Finished. Predictions saved to {output_file}")
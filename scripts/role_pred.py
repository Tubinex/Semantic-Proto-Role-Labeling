import os
import importlib.util

def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(
        name,
        os.path.join(os.path.dirname(__file__), relative_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# load both modules
supervised = load_module("superviesed_pred", "../probing/superviesed_pred.py")
naive      = load_module("naive_pred",       "../probing/naive_pred.py")

if __name__ == "__main__":

    BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
    DATA_DIR = os.path.join(BASE_DIR, "data")

    probing_file       = os.path.join(DATA_DIR, "predictions.jsonl")
    gold_file          = os.path.join(DATA_DIR, "spr1_with_gold.json")
    supervised_output  = os.path.join(DATA_DIR, "proto_role_predictions.jsonl")
    naive_output       = os.path.join(DATA_DIR, "proto_role_naive_pred.jsonl")

    # --- supervised ---
    print("Running supervised prediction...")
    probing_data = supervised.load_probing(probing_file)
    gold_roles   = supervised.load_gold_roles(gold_file)
    X_dicts, y_labels, spr_ids = supervised.build_dataset(probing_data, gold_roles)

    property_list = sorted({prop for sample in X_dicts for prop in sample.keys()})

    classifier = supervised.ProtoLabelClassifier(property_list)
    classifier.train(X_dicts, y_labels)
    y_pred, y_proba = classifier.predict_with_proba(X_dicts)

    supervised.save_predictions(supervised_output, spr_ids, y_labels, y_pred, y_proba, classifier.label_encoder)
    print(f"Supervised predictions saved to {supervised_output}")

    # --- naive ---
    print("Running naive prediction...")
    grouped_data = naive.group_input(probing_file)
    results      = naive.predict_all(grouped_data)
    naive.save_predictions(results, naive_output)
    print(f"Naive predictions saved to {naive_output}")
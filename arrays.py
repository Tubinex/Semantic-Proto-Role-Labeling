import json

raw_pairs = {}
pairs =  []
predictions = []

with open('./data/processed/pairs.jsonl', 'r') as f:
    for line in f:
        raw = str.strip(line)
        data = json.loads(raw)
        raw_pairs[data['id']] = data

with open('./data/processed/predictions.jsonl', 'r') as f:
    for line in f:
        raw = str.strip(line)
        data = json.loads(raw)

        predictions.append(1 if data['pred_bool'] else 0)
        pairs.append(1 if raw_pairs[data['id']]['label'] > 3 else 0)

correct = 0
for i in range(0, len(pairs)):
    if pairs[i] == predictions[i]:
        correct += 1

print(f'Correct labels: {correct} ({int((100 / len(pairs) * correct) * 100) / 100}%)')
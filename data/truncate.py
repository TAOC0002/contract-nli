import json

input_path = './contract-nli/train.json'
output_path = './contract-nli/train_truncated.json'
truncated_size = 50

with open(input_path) as json_file:
    data = json.load(json_file)
data['documents'] = data['documents'][:truncated_size]
with open(output_path, 'w') as outfile:
    json.dump(data, outfile, indent=4)

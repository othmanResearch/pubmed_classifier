import json




def read_bern2_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(data)


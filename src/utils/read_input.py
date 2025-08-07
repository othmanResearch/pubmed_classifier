import json
import os


def read_bern2_json(filepath) -> list:
    """
    read a json file containing one or multiple tagged elements
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def scan_json_path(jsondir) -> list:
    """
    scans a directory containing json BERN2 nnotated files and return one list
    containing all the annotations
    """
    all_entries = []
    for json in os.scandir(jsondir):
        list_of_entries = read_bern2_json(json)
        all_entries.extend(list_of_entries)
    return all_entries



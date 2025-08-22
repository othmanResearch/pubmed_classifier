import json
import os
import re
from collections import Counter
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

def read_annotated_texts(annotated_texts_file) -> list:
    """ """
    with open(annotated_texts_file, 'r') as f:
        data = f.readlines()
    return data 


def extract_placeholders(text):
    """
    Extract all unique placeholders of the form __PLACEHOLDER__ from text.

    Args:
        text (str): Multi-line text input.

    Returns:
        list of str: Unique placeholders found in the text.
    """
    # Regex: two underscores + 1+ uppercase letters/digits + two underscores
    pattern = r'__([A-Za-z0-9]+(?:_[A-Za-z0-9]+)?)__'
    matches = re.findall(pattern, text)

    # Reconstruct full placeholder with underscores and remove duplicates
    unique_placeholders = sorted(set(f"__{m}__" for m in matches))
    full_placeholders = [f"__{m}__" for m in matches]
    counts = dict(Counter(full_placeholders))
    
    return counts



from collections import Counter
import json
import tqdm
import warnings


def filter_by_probability(bern2_annotated_abstract, prob=0.95) -> None:
    """
    takes a BURN2 annotated abstrat and keeps the tags with 
    probability that goes byond a certain cutoff
    
    Args:
        bern2_annotated_abstract (list): a bern2 annotation of an abstract
        prob (float): the probability cutpff 
    """
    annotations = bern2_annotated_abstract['annotations']
    new_annotation = []
    for elem in annotations:
        try:
            if float(elem['prob']) >= prob :
                new_annotation.append(elem)
        except :
            warnings.warn("anntation with no prob vale, will be retained by default")
            new_annotation.append(elem)
        finally:
            pass
    bern2_annotated_abstract['annotations'] = new_annotation
    return bern2_annotated_abstract

def tag_no_disease_abstracts(bern2_annotated_abstract):
    """
    returns True if abstract contains disease tag or False otherwise

    Args:
        bern2_annotated_abstract (list): a bern2 annitation of an abstract
    """
    annotations = bern2_annotated_abstract['annotations']
    if any(annotation['obj'] == 'disease' for annotation in annotations ):
        return True
    else:
        return False

def remove_overlapping_annotations(bern2_abstract_annotations):
    """
    Remove overlapping annotations, keeping the longest span when overlaps occur.
    
    Args:
        bern2_abstract_annotations (list): List of annotations with 'span' dicts 
            containing 'begin' and 'end' keys. This could be the 'annotations' 
            element from BERN2 output of an abstract
    
    Returns:
        list: Non-overlapping annotations.
    """
    # Sort annotations by span start, then by longest span (descending)
    sorted_data = sorted(bern2_abstract_annotations, key=lambda x: (x['span']['begin'], -(x['span']['end'] - x['span']['begin'])))
    result = []
    for ann in sorted_data:
        overlap = False
        for kept in result:
            if not (ann['span']['end'] <= kept['span']['begin'] or ann['span']['begin'] >= kept['span']['end']):
                # Overlap detected
                overlap = True
                # If current annotation is longer, replace the kept one
                if (ann['span']['end'] - ann['span']['begin']) > (kept['span']['end'] - kept['span']['begin']):
                    result.remove(kept)
                    result.append(ann)
                break
        if  overlap == False:
            result.append(ann)

    return result

def filter_homo_sapiens(annotations):
    """
    Filter annotations to keep only those with species 'NCBITaxon:9606', 
    and update their 'obj' value to 'homo sapiens'. 
    Other annotations with different 'obj' values are kept unchanged.
    
    Args:
        annotations (list[dict]): List of annotation dictionaries.
        
    Returns:
        list[dict]: Filtered and updated annotations.
    """

    filtered = []
    for ann in annotations:
        if ann.get("obj") == "species":
            if ann.get("id") == ["NCBITaxon:9606"]:
                ann = ann.copy()  # avoid mutating the original
                ann["obj"] = "homo sapiens"
                filtered.append(ann)
        else:
            filtered.append(ann)
    return filtered

def insert_inline_tags(bern2_abstract, tag_style="placeholder"):
    """
    Insert inline entity tags into text without replacing the original mention.
    Bracketed annotations are replaced with safe placeholders (e.g., __GENE__).

    Args:
        bern2_abstract (dict): Dictionary with 'text' and 'annotations'.
        tag_style (str): 
            'placeholder' -> __ENTITY__ (safe for tokenization)
            'bracket' -> [ENTITY]
            'angle' -> <ENTITY>

    Returns:
        str: Text with inline tags added after each mention.
    """
    text = bern2_abstract["text"]
    annotations = sorted(
        bern2_abstract.get("annotations", []),
        key=lambda x: x['span']['begin'],
        reverse=True
    )

    for ann in annotations:
        start = ann['span']['begin']
        end = ann['span']['end']
        obj = ann['obj'].upper()  # e.g., "GENE", "DISEASE"

        # Determine the tag style
        if tag_style == "placeholder":
            tag = f"__{obj}__"
        elif tag_style == "angle":
            tag = f"<{obj}>"
        else:  # default bracket
            tag = f"[{obj}]"

        # Insert the tag after the mention
        text = text[:end] + tag + text[end:]

    return text


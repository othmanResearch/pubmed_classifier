

def filter_by_probability(bern2_annotated_abstract, prob=0.95) -> None:
    """
    takes a BURN2 annotated abstrat and keeps the tags with 
    probability that goes byond a certain cutoff
    
    Args:
        bern2_annotated_abstract (list): a bern2 annotation of an abstract
        prob (float): the probability cutpff 
    """
    annotations = bern2_annotated_abstract['annotations']
    new_annotaion = []
    for elem in annotations :
        if float(elem['prob']) >= prob :
            new_annotaion.append(elem)
    bern2_annotated_abstract['annotations'] = new_annotaion
    return bern2_annotated_abstract
    
    print(new_annotaion)
    #filtered_data = [item for item in annotations ]
    #print(filtered_data)

    



def filter_by_probability(bern2_annotated_abstract):
    annotations = bern2_annotated_abstract['annotations']
    new_annotaion = []
    for elem in annotations :
        if float(elem['prob']) >= 0.95 :
            new_annotaion.append(elem)
    
    print(new_annotaion)
    #filtered_data = [item for item in annotations ]
    #print(filtered_data)

    

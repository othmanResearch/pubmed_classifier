from metaflow import FlowSpec, step, Config
import sys
import os
import tqdm 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))    # set before calling internal modules
from utils import read_input, annotation

class processData(FlowSpec):

    config = Config("preprocess", required = True)
    
    @step
    def start(self):
        """ reads the path to json file  container directory into self.dataset"""
        self.dataset = read_input.scan_json_path( self.config.bern2_dataset)
        self.next(self.remove_no_disease_abstracts)
    
    @step
    def remove_no_disease_abstracts(self):
        """ retains abstracts if contain disease annotation """
        self.disease_info_abstracts = []
        for abstract in tqdm.tqdm(self.dataset):
            if annotation.tag_no_disease_abstracts(abstract):
                self.disease_info_abstracts.append(abstract)
        self.next(self.filter_tags)

    @step
    def filter_tags(self):
        """reains annotations exceeding or equals a probability cutoff """
        self.prob_filered_annotations = [annotation.filter_by_probability(abstract) for abstract in self.disease_info_abstracts]
        self.next(self.remove_ner_overlap)

    @step
    def  remove_ner_overlap(self):
        for abstract in self.prob_filered_annotations:
            abstract_annotations = abstract['annotations']
            filtered_annotions = annotation.remove_overlapping_annotations(abstract_annotations)
            abstract['annotations'] = filtered_annotions

        self.next(self.insert_tags)

    @step
    def insert_tags(self):

        self.next(self.end)


    @step
    def end(self):
        pass

if __name__=="__main__":
    processData()







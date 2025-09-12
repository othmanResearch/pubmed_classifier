from metaflow import FlowSpec, step, Config, Parameter
import sys
import os
import tqdm
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))    # set before calling internal modules
from utils import read_input, annotation

# allow INFO level of logging
logging.basicConfig(level=logging.INFO)


class processData(FlowSpec):

    config = Config("preprocess", required = True)

    
    @step
    def start(self):
        """ reads the path to json file  container directory into self.dataset"""
        full_path = os.path.abspath(self.config.bern2_dataset)
        self.dataset = read_input.scan_json_path( full_path)
        total_number = len(self.dataset)
        logging.info(f'total number of abstracts:   {total_number}')
        self.next(self.remove_short_abstracts)

    @step
    def remove_short_abstracts(self):
        """keep only significant cutoff with more than 50 words """
        self.length_filterd_abstracts = []
        for abstract in self.dataset:
            words = abstract["text"].split()
            if len(words) > 50:
                self.length_filterd_abstracts.append(abstract)
        logging.info(f"Number of retained abstracts filtered by a cutoff of 50 words: {len(self.length_filterd_abstracts)}")
        self.next(self.filter_tags)

    @step
    def filter_tags(self):
        """retains annotations exceeding or equals a probability cutoff """
        self.prob_filered_annotations = [annotation.filter_by_probability(abstract) for abstract in self.length_filterd_abstracts]
        self.next(self.keep_only_humans)

    @step
    def keep_only_humans(self):
        """Remove non human species from the list of annotations"""
        self.species_filtered_annotations =  self.prob_filered_annotations.copy()
        for abstract in tqdm.tqdm(self.species_filtered_annotations):
            abstract_annotations = abstract['annotations']
            filtered_annotions = annotation.filter_homo_sapiens(abstract_annotations)
            abstract['annotations'] = filtered_annotions
        self.next(self.remove_ner_overlap)

    @step
    def  remove_ner_overlap(self):
        for abstract in self.species_filtered_annotations:
            abstract_annotations = abstract['annotations']
            filtered_annotions = annotation.remove_overlapping_annotations(abstract_annotations)
            abstract['annotations'] = filtered_annotions

        self.next(self.insert_tags)

    @step
    def insert_tags(self):
        self.corpus_transformed = []
        self.pmids = []
        for abstract in self.species_filtered_annotations:            
            self.corpus_transformed.append(annotation.insert_inline_tags(abstract, tag_style = 'placeholder' ))
            self.pmids.append(abstract["_id"])

        self.next(self.end)


    @step
    def end(self):
        try:
            output_path = self.config.output
        except:
            logging.info('no output file was processed, will proceed with default')
            os.makedirs('./output', exist_ok=True)
            output_path = './output/preprocessed_abtracts.txt'

        with open(output_path, 'w') as file:
            for pmid,item in zip(self.pmids, self.corpus_transformed):
                file.write(item + '\t' + pmid +'\n')

if __name__=="__main__":
    processData()







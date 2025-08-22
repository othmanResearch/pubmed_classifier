from metaflow import FlowSpec, step, Config, Parameter
import spacy
import sys
import os
import tqdm
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))    # set before calling internal modules
from utils import read_input, preprocess_text

# allow INFO level of logging
logging.basicConfig(level=logging.INFO)



class preprocessAnnotText(FlowSpec):
    
    config = Config('processtext', required = True)

    @step
    def start(self):
        self.annotated_texts = read_input.read_annotated_texts(self.config.annotated_texts)
        self.next(self.generate_place_holders)

    @step
    def generate_place_holders(self):
        """ not necessary but the user can check the list of the generated annotation in the text if the will"""
        with open(self.config.annotated_texts, "r", encoding="utf-8") as f:
            text = f.read()
        unique_place_holders = read_input.extract_placeholders(text)
        self.next(self.cook_text)
    
    @step
    def cook_text(self):
        """ tokenize, remove stop words and punctuation """
        self.list_of_processed_texts = []
        for text in tqdm.tqdm(self.annotated_texts):
            #print(text)
            tokenized_text = preprocess_text.tokenize_text(text)
            sp_text = preprocess_text.remove_stopwords_punct(tokenized_text)
            self.list_of_processed_texts.append(sp_text)
        nb_texts = len(self.list_of_processed_texts)
        logging.info(f'{nb_texts} texts will be saved to pickle fornmat')
        self.next(self.end)

    @step
    def end(self):
        nb_texts = len(self.list_of_processed_texts)
        logging.info(f'{nb_texts} texts will be saved to pickle fornmat')
        

if __name__ == "__main__": 
    preprocessAnnotText()
    

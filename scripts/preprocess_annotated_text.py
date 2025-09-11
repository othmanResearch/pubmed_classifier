from metaflow import FlowSpec, step, Config, Parameter
import spacy
import sys
import os
import tqdm
import logging
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))    # set before calling internal modules
from utils import read_input, preprocess_text

# allow INFO level of logging
logging.basicConfig(level=logging.INFO)



class preprocessAnnotText(FlowSpec):
    
    config = Config('preprocess', required = True)   

    @step
    def start(self):
        dataset = read_input.read_annotated_texts(self.config.annotated_texts)
        self.annotated_texts = []
        self.pmids = []
        for line in dataset :
            splitted = line.split("\t")
            self.annotated_texts.append(splitted[0])
            self.pmids.append(splitted[1].strip())
        self.next(self.generate_place_holders)

    @step
    def generate_place_holders(self):
        """ not necessary but the user can check the list of the generated annotation in the text if they will"""
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
        self.next(self.join_tokens)

    @step
    def join_tokens(self):
        self.joined_tokenized_texts = []
        for tokenised_text in self.list_of_processed_texts:
            self.joined_tokenized_texts.append(' '.join(tokenised_text))
        self.next(self.end)

    @step
    def end(self):
        nb_texts = len(self.joined_tokenized_texts)
        logging.info(f'{nb_texts} texts and pmids will be saved to pickle format')
        
        try:
            output_path = self.config.output_pkl
            logging.info(f"Will output dataset to {self.config.output_pkl}")
        except:
            logging.info('no output file was processed, will proceed with default')
            os.makedirs('./output', exist_ok=True)
            output_path = './output/text_for_ml.pkl'
       
        with open(output_path, "wb") as f:
            # this will save both the processed texts and the pmids into one pkl object
            pickle.dump([self.joined_tokenized_texts, self.pmids], f)

        

if __name__ == "__main__": 
    preprocessAnnotText()
    

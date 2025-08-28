from metaflow import FlowSpec, step, Config, Parameter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import sys
import os
import tqdm
import logging
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))    # set before calling internal modules
from utils import read_input, annotation

# allow INFO level of logging
logging.basicConfig(level=logging.INFO)

class removeOverlap(FlowSpec):
  
    config = Config("overlap", required = True)

    @step
    def start(self):
        self.positive = read_input.read_pkl(self.config.cls1)
        logging.info(f"{len(self.positive)} biomedical texts will be processed for the positive class")
        self.negative = read_input.read_pkl(self.config.cls2)
        logging.info(f"{len(self.negative)} biomedical texts will be processed for the negative class")
        self.next(self.assign_class)
    
    @step
    def assign_class(self):
        self.processed_data = self.positive+self.negative
        self.labels = np.array(len(self.positive)*[1] + len(self.negative)*[0])
        self.next(self.vectirise_text_entries)

    @step
    def vectirise_text_entries(self):
        """ genrate embeddings based on TF-idf, ngrams are set to (1,2)"""
        tfidf_vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
        self.vectorised_data  = tfidf_vectorizer.fit_transform(self.processed_data) 
        self.next(self.subset_data)

    @step
    def subset_data(self):
        # sparate indices fron self.processed 
        self.class0_idx = np.where(self.labels == 0)[0]
        self.class1_idx = np.where(self.labels == 1)[0]
        # subset_dataset vectorised data 
        self.X0 = self.vectorised_data[self.class0_idx]
        self.X1 = self.vectorised_data[self.class1_idx]
        self.next(self.compute_cosine_matrix)

    @step 
    def compute_cosine_matrix(self):
        self.similarity_matrix = cosine_similarity(self.X0, self.X1)
        print(self.similarity_matrix)
        self.next(self.filter)

    @step
    def filter(self):
        try:
            threshold = self.config.threshold
            logging.info(f"Threshold = {self.config.threshold}")
        except:
            threshold = 0.5
            logging.info(f"Threshold will be assigned to default {threshold}")

        # For each class-0 point, check if it is too similar to any class-1 point
        overlap_mask = (self.similarity_matrix.max(axis=1) < threshold)  # True = keep
        # Filter class-0 points
        #self.filtered_class0_idx = self.class0_idx[overlap_mask]
        #print("Original class 0 points:", len(self.class0_idx))
        
        #print("Remaining class 0 points after removing overlaps:", len(self.filtered_class0_idx))
        # Combine with class-1 points to get the final filtered dataset
        #filtered_idx = np.concatenate([self.filtered_class0_idx, self.class1_idx])
        #filtered_vectorised_data = self.vectorised_data[filtered_idx]
        #print(self.vectorised_data.shape)
        #filtered_labels = self.labels[filtered_idx]

        #array = np.array(self.processed_data)[filtered_idx]

        #print("Filtered dataset shape:", filtered_vectorised_data.shape)

        self.next(self.end)

    @step
    def end(self):
        #print(self.filtered_class0_idx)

        pass




if __name__=="__main__":
    removeOverlap()



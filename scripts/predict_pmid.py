from metaflow import FlowSpec, step, Config, Parameter
import sys
import os
import logging
import joblib
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))    # set before calling internal modules
from utils import read_input, annotation

# allow INFO level of logging
logging.basicConfig(level=logging.INFO)


class predictPmids(FlowSpec):
    config = Config("predict", required=True)

    @step
    def start(self):
        """ load the predictive model and the data """
        with open(self.config.model_path, "rb") as f:
            logging.info("Loodading the predictive model")
            self.pipeline = joblib.load(f)

        with open(self.config.prediction_dataset, "rb") as f:
            prediction_dataset = pickle.load(f)
            self.texts = prediction_dataset[0]
            self.pmids = prediction_dataset[1]
        self.next(self.predict)

    @step
    def predict(self):
        preds = self.pipeline.predict(self.texts)
        print(preds)

        self.next(self.end)

    @step
    def end(self):

        pass 


if __name__ == "__main__":
    predictPmids()

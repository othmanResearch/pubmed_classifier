from metaflow import FlowSpec, step, Config, Parameter
import sys
import os
import logging
import joblib
import pickle
import pandas as pd 

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
            logging.info("Loaading the predictive model")
            self.pipeline = joblib.load(f)

        with open(self.config.prediction_dataset, "rb") as f:
            prediction_dataset = pickle.load(f)
            self.texts = prediction_dataset[0]
            self.pmids = prediction_dataset[1]
        logging.info(f"{len(self.pmids)} abstracts are set for the prediction")
        self.next(self.predict)

    @step
    def predict(self):
        preds = self.pipeline.predict(self.texts)
        self.probs = self.pipeline.predict_proba(self.texts)[:, 1] if hasattr(self.pipeline, "predict_proba") else None
        self.next(self.end)

    @step
    def end(self):
        try:
            output = os.path.abspath( self.config.output)
            logging.info(f"Output model to {output}")
        except :
            os.makedirs('./output', exist_ok=True)
            output = os.path.abspath('./output/predictions.csv')
            logging.info(f'No output file was specified, will output to {output}')
       
        df = pd.DataFrame( {"pmids": self.pmids, "pob_class_1": self.probs} )
        df.to_csv(output, index = False)
        

if __name__ == "__main__":
    predictPmids()

from metaflow import FlowSpec, step, Config, Parameter
import sys
import os
import logging
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))    # set before calling internal modules
from utils import read_input, annotation

# allow INFO level of logging
logging.basicConfig(level=logging.INFO)


class predictPmids(FlowSpec):
    config = Config("predict", required=True)

    @step
    def start(self):
        """ load the data """
        with open(self.config.model_path, "rb") as f:
            logging.info("Loodading the predictice model")
            self.pipeline = joblib.load(f)

        self.next(self.end)

    @step
    def end(self):

        pass 


if __name__ == "__main__":
    predictPmids()

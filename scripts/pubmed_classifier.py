
from metaflow import FlowSpec, step, Config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))    # set before calling internal modules
from utils import read_input

class processData(FlowSpec):

    config_process_data = Config("preprocess", required = True)
    print(config_process_data["negative"])
    @step
    def start(self):

        self.myvar = "hello"
        self.next(self.end)


    @step
    def end(self):
        pass

if __name__=="__main__":
    processData()







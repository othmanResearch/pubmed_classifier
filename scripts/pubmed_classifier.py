from metaflow import FlowSpec, step, Config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))    # set before calling internal modules
from utils import read_input, annotation

class processData(FlowSpec):

    config = Config("preprocess", required = True)
    
    @step
    def start(self):
        """ reads the path to json file  container directory into self.dataset"""
        self.dataset = read_input.scan_json_path( self.config.bern2_dataset)
        annotation.filter_by_probability(self.dataset[0])
        self.next(self.end)


    @step
    def end(self):
        pass

if __name__=="__main__":
    
    processData()







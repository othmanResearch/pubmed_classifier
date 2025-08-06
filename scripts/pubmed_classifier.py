
from metaflow import FlowSpec, step
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import read_input



class processData(FlowSpec):

    @step
    def start(self):
        self.myvar = "hello"
        self.next(self.end)


    @step
    def end(self):
        pass

if __name__=="__main__":
    processData()







from metaflow import FlowSpec, step, Config, Parameter
import sys
import os
import tqdm
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))    # set before calling internal modules
from utils import read_input, annotation

# allow INFO level of logging
logging.basicConfig(level=logging.INFO)



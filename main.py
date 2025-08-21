import argparse
from metaflow import Runner
import logging 

logging.basicConfig(level = logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Template script for handling command-line arguments")
    parser.add_argument("-m", "--mode", required=True,
                        help="Choose the mode for running the pipeline (process, train, predict)")
    
    parser.add_argument('-c', "--config", required=False, 
                        help='Config file for running the workflow')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'preprocess':
        logging.info('Running data preprocessing')
        
        # raise error if configuration file was not given
        if not args.config:
            logging.error(f"Required configuration is missing.")
            raise FileNotFoundError(f"Required configuration file is missing.")

        with Runner('./scripts/preprocess.py', show_output=False).run(config='./config/training.json') as running:
            pass


           


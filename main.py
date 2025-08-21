import argparse
from metaflow import Runner

def parse_args():
    parser = argparse.ArgumentParser(description="Template script for handling command-line arguments")
    parser.add_argument("-m", "--mode", required=True,
                        help="""Choose the mode for running the pipeline (process, train, predict ) """)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)

           


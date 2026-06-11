import argparse
import json
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from training import trainer  # noqa: E402

logging.basicConfig(level=logging.INFO)


def parse_args():
    p = argparse.ArgumentParser(description="Train the PubMed text classifier")
    p.add_argument("--config", required=True, help="Path to the training JSON config")
    p.add_argument("--test-size", type=float)
    p.add_argument("--val-size", type=float)
    p.add_argument("--random-state", type=int)
    p.add_argument("--n-splits", type=int)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    for key, val in {
        "test_size": args.test_size,
        "val_size": args.val_size,
        "random_state": args.random_state,
        "n_splits": args.n_splits,
    }.items():
        if val is not None:
            config[key] = val

    summary = trainer.run_training(config)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

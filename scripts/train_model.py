from metaflow import FlowSpec, step, Config, Parameter
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))    # set before calling internal modules
from training import trainer

logging.basicConfig(level=logging.INFO)


class trainModel(FlowSpec):

    config = Config("training", required=True)
    test_size = Parameter('test_size', default=config.test_size,
                          help='test size proportion relative to the whole dataset')
    val_size = Parameter('val_size', default=0.2,
                         help='validation size proportion relative to the train+val pool')
    random_state = Parameter('random_state', default=config.random_state)
    n_splits = Parameter('n_splits', default=10,
                         help='folds for cross-validation on the training set')

    @step
    def start(self):
        self.cfg = dict(self.config)
        self.cfg.update({
            "test_size": self.test_size,
            "val_size": self.val_size,
            "random_state": self.random_state,
            "n_splits": self.n_splits,
        })
        self.next(self.train)

    @step
    def train(self):
        """Split, cross-validate, evaluate on held-out test, and log to MLflow."""
        self.summary = trainer.run_training(self.cfg)
        logging.info("training summary: %s", self.summary)
        self.next(self.end)

    @step
    def end(self):
        logging.info("training complete")


if __name__ == "__main__":
    trainModel()

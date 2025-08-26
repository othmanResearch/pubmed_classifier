from metaflow import FlowSpec, step, Config, Parameter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score

import sys
import os
import tqdm
import logging
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))    # set before calling internal modules
from utils import read_input, annotation

# allow INFO level of logging
logging.basicConfig(level=logging.INFO)

class rainModel(FlowSpec):

    config = Config("training", required = True)
    test_size = Parameter('test_size', default=config.test_size, help='test size proportion relative to the dataset' )
    random_state = Parameter('random_state', default = config.random_state)

    @step
    def start(self):
        self.positive = read_input.read_pkl(self.config.cls1)
        logging.info(f"{len(self.positive)} biomedical texts will be processed for the positive class")
        self.negative = read_input.read_pkl(self.config.cls2)
        logging.info(f"{len(self.negative)} biomedical texts will be processed for the negative class")
        self.next(self.join_dataset)

    @step
    def join_dataset(self):
        self.processed_data = self.positive+self.negative
        self.labels = len(self.positive)*[1] + len(self.negative)*[0]
        self.next(self.split_data)

    @step
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.processed_data, self.labels, 
                                                                                test_size=self.test_size, 
                                                                                random_state=self.random_state, 
                                                                                stratify = self.labels)
        logging.info(f"""Splitting the data\n test size: {self.test_size} \n random state for splitting data: {self.random_state}""")
        logging.info(f"""Training dataset size:   {len(self.X_train)}
                         Testing dataset size:    {len(self.X_test)}
                     """)
        self.next(self.vectorise_and_build_model)

    @step 
    def vectorise_and_build_model(self):
        pipeline = Pipeline([('tfidf', TfidfVectorizer(min_df=1, ngram_range=(1, 2))),
                             ('clf', LogisticRegression(C=10, penalty='l2', solver='liblinear', random_state=self.random_state))
                             ])
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        # Accuracy scores for each fold
        cv_scores = cross_val_score(pipeline, self.processed_data, self.labels, cv=cv, scoring='accuracy')
        print("Cross-validation accuracies:", cv_scores)
        print("Mean accuracy: {:.4f} ± {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))
        # ---------------------------
        # Step 4: Train model
        # ---------------------------
        pipeline.fit(self.X_train, self.y_train)
        # Predictions
        self.y_pred = pipeline.predict(self.X_test)
        self.y_proba = pipeline.predict_proba(self.X_test)[:, 1]  # probability for positive class

        # ---------------------------
        # Step 5: Performance metrics
        # ---------------------------
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, self.y_pred))

        roc_auc = roc_auc_score(self.y_test, self.y_proba)
        print("\nROC AUC Score: {:.4f}".format(roc_auc))

        self.next(self.end)


    @step 
    def end(self):

        return

if __name__ == "__main__":
    rainModel()

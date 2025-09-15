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
import joblib
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))    # set before calling internal modules
from utils import read_input, annotation

# allow INFO level of logging
logging.basicConfig(level=logging.INFO)

class trainModel(FlowSpec):

    config = Config("training", required = True)
    test_size = Parameter('test_size', default=config.test_size, help='test size proportion relative to the dataset' )
    random_state = Parameter('random_state', default = config.random_state)
    tracing = Parameter('tracing', default= False)

    @step
    def start(self):
        self.positive_all = read_input.read_pkl(self.config.cls1) # reads only the texts not the pmids
        self.positive = self.positive_all[0]
        self.positive_pmids = self.positive_all[1]
        logging.info(f"{len(self.positive)} biomedical texts will be processed for the positive class")
        self.negative_all = read_input.read_pkl(self.config.cls2)
        self.negative = self.negative_all[0]
        self.negative_pmids = self.negative_all[1]
        logging.info(f"{len(self.negative)} biomedical texts will be processed for the negative class")
        self.next(self.join_dataset)

    @step
    def join_dataset(self):
        self.processed_data = self.positive+self.negative
        self.labels = len(self.positive)*[1] + len(self.negative)*[0]
        self.next(self.split_data)

    @step
    def split_data(self):
        indices = np.arange(len(self.processed_data))  # reauired to get tthe details about the training and validation data in tracing mode
        self.X_train, self.X_test, self.y_train, self.y_test , self.idx_train, self.idx_test = train_test_split(self.processed_data, self.labels, 
                                                                                indices, 
                                                                                test_size=self.test_size, 
                                                                                random_state=self.random_state, stratify = self.labels)
        self.next(self.vectorise_and_build_model)

    @step 
    def vectorise_and_build_model(self):
        self.pipeline = Pipeline([('tfidf', TfidfVectorizer(min_df=2, ngram_range=(1, 3))),
                             ('clf', LogisticRegression(C=10, penalty='l2', solver='liblinear', random_state=self.random_state))
                             ])
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        # Accuracy scores for each fold
        self.cv_scores = cross_val_score(self.pipeline, self.processed_data, self.labels, cv=cv, scoring='accuracy')
        # ---------------------------
        #  Train model
        # ---------------------------
        self.pipeline.fit(self.X_train, self.y_train)
        # Predictions
        self.y_pred = self.pipeline.predict(self.X_test)
        self.y_proba = self.pipeline.predict_proba(self.X_test)[:, 1]  # probability for positive class

        self.next(self.end)


    @step 
    def end(self):
        try:
           output_model = os.path.abspath( os.path.expanduser( self.config.output_model))
           logging.info(f"Output model to {output_model}")
        except :
            os.makedirs('./output', exist_ok=True)
            output_model = os.path.abspath('./output/model.pkl')
            logging.info(f'no output file was specified, will output model to {output_model}')
        
        joblib.dump(self.pipeline, output_model)

        #-----------------
        # ouptut metrics
        # ----------------
        try:
           output_roc = os.path.abspath( os.path.expanduser( self.config.output_roc_data))
           logging.info(f"Output ROC data to {output_roc}")
           df = pd.DataFrame({"y_test":self.y_test, 
                          "y_pred":self.y_pred, 
                          "y_proba":self.y_proba})
           df.to_csv(output_roc, index = False)

        except: 
            logging.info("Output ROC data will not be generated")

        #---------------
        # output tracing files for training dataset
        # ------------------
        if self.tracing == True:
            output_trace = output_model.replace(".pkl", "_trace.csv")
            all_pmids = np.array(self.positive_pmids+self.negative_pmids)
            training_pmids = all_pmids[self.idx_train]
            training_labels = np.array(self.labels)[self.idx_train]
            training_type = len(training_pmids)*['training']

            testing_pmids = all_pmids[self.idx_test]
            testing_labels = np.array(self.labels)[self.idx_test]
            testing_type = len(testing_pmids)*['testing']

            df = pd.DataFrame({"pmid": np.concatenate((training_pmids , testing_pmids)),
                  "class_label": np.concatenate((training_labels, testing_labels)),
                  "type": training_type+testing_type})
            logging.info(f"Output training and testing dataset details to {output_trace}")
            df.to_csv(output_trace, index=False)

            #-------------------
            # generate report 
            # -----------------
            report_path = output_model.replace(".pkl", "_training.log")
            logging.info(f"Otput metrics to log file: {report_path}")
            with open(report_path, "w") as log:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                log.write(f"""REPORT FOR TRAINING PUBMED CLASSIFIER \n""")
                log.write(f'{timestamp}\n')
                log.write(f"Splitting the data\n \ttest size: {self.test_size} \n\t random state for splitting data: {self.random_state}\n")
                log.write(f"\tTraining dataset size:  {len(self.X_train)} Testing dataset size:  {len(self.X_test)}")
                log.write("\nClassification Report:")
                log.write(classification_report(self.y_test, self.y_pred))

                roc_auc = roc_auc_score(self.y_test, self.y_proba)
                log.write("\nROC AUC Score: {:.4f}".format(roc_auc))
                log.write(f"Cross-validation accuracies: {self.cv_scores} \n")
                log.write("Mean accuracy: {:.4f} ± {:.4f}".format(np.mean(self.cv_scores), np.std(self.cv_scores)))



                  

if __name__ == "__main__":
    trainModel()

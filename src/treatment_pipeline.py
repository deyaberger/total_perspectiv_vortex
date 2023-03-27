"""Processing EEG data pipeline."""
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import joblib
import logging

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split

from mne.decoding import CSP

from utils import get_tpv_args, tasks
from data_parsing import Parser

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("treatment")
logger.setLevel(level=logging.INFO)

class Treatment():
    def __init__(self, args):
        # TODO replace subjects from 1 to 109 and runs to 0, 4:
        self.subjects = [args.subject] if args.train or args.predict else range(1, 109)
        self.run_idxs = [args.run_idx] if args.train or args.predict else range(0, 4)
        self.runs = [x['runs'] for i, x in enumerate(tasks) if i in self.run_idxs]

    def build_pipe(self, lda_name: str = 'classic', csp_components: int = 4) -> Pipeline:
        if (lda_name == 'shrimp'):
            lda = LDA(solver='lsqr', shrinkage='auto')
        elif lda_name == 'classic':
            lda = LDA()
        csp = CSP(n_components=csp_components)
        pipeline = Pipeline([("CSP", csp), ("LDA", lda)])
        return pipeline

    def get_pretrained_model(self, dir, run_id, subject):
        sub_dir = f"run_{run_id}"
        filename = f"S{subject:03d}.save"
        complete_path = f'{dir}/{sub_dir}/{filename}'
        if not os.path.exists(complete_path):
            return None
        pipe = joblib.load(complete_path)
        return pipe


class Train(Treatment):
    def __init__(self, args):
        super().__init__(args)
        self.training_data = None
        self.pipe = None
        self.preload_models = True
        self.train()

    def cross_val_pipeline(self, pipeline, epochs) -> float:
        targets = epochs.events[:, -1]
        epochs_data = epochs.copy().crop(tmin=1.0, tmax=4.0).get_data() # They do this cropping on MNE website. Why ?
        # TODO: keep or remove random state?
        scores = cross_val_score(pipeline, epochs_data, targets, cv=ShuffleSplit(10, test_size=0.2, random_state=42), n_jobs=None)
        return scores.mean()


    def train_pipeline(self, pipeline: Pipeline, epochs, test_size = 0.1) -> Pipeline:
        targets = epochs.events[:, -1]
        epochs_data = epochs.copy().crop(tmin=1.0, tmax=4.0).get_data() # They do this cropping on MNE website. Why ?

        x_train, x_test, y_train, y_test = train_test_split(epochs_data, targets, test_size=test_size, random_state=0)

        x_train, y_train = epochs_data, targets
        pipeline.fit(X=x_train, y=y_train)
        score = pipeline.score(X=x_test, y=y_test)
        logger.info(f"trained_pipeline_score: {score:.2f}")
        return pipeline

    def save_pipeline_to_disk(self, pipe: Pipeline, dir: str, run_id: int, subject: int, parser):
        sub_dir = f"run_{run_id}"
        parser.create_dir_if_not_exists(dir, sub_dir)
        filename = f"S{subject:03d}.save"
        complete_path = f'{dir}/{sub_dir}/{filename}'
        joblib.dump(pipe, complete_path)
        logger.info(f"Saved pipeline to: {complete_path}")

    def train(self):
        training_data = pd.DataFrame(columns=['subject', 'task_number', 'cross_val_score'])
        for run_id in self.run_idxs:
            run = tasks[run_id]['runs']
            labels = tasks[run_id]['labels']
            logger.info(f"**************** RUN: {run}")
            for subject in self.subjects:
                parser = Parser(subject=subject, run=run, mne_path="./mne_data", run_id=run_id)
                parser.motion_preprocessing(labels)
                epochs = parser.get_epochs(epochs_dir="./epochs", save=True, preload=True)
                if self.preload_models:
                    self.pipe = self.get_pretrained_model("./models", run_id, subject)
                if self.pipe is None or self.preload_models is False:
                    self.pipe = self.build_pipe(lda_name='classic', csp_components=4)
                    self.pipe = self.train_pipeline(self.pipe, epochs, test_size=0.1)
                    self.save_pipeline_to_disk(self.pipe, "./models", run_id, subject, parser)
                cross_score = self.cross_val_pipeline(self.pipe, epochs)
                logger.info(f"Run:{run},\t\tsubject:{subject},\t\tCross validation average:{cross_score:.2f}")
                model_metrics = {
                    'subject': subject,
                    'task_number': run_id,
                    'cross_val_score': cross_score,
                }
                metrics = pd.DataFrame(model_metrics, index=[0])
                training_data = pd.concat([training_data, metrics])
        parser.create_dir_if_not_exists("./results")
        training_data.to_csv("./results/training.csv")
        self.training_data = training_data


class Predict(Treatment):
    def __init__(self, args):
        super().__init__(args)
        if len(self.subjects) != 1 or len(self.run_idxs) != 1:
            logger.error("Predict class take one subject and one run only")
        self.run_id = self.run_idxs[0]
        self.subject = self.subjects[0]
        self.predict()


    def predict_epochs(self, pipe, epochs):
        epochs_data = epochs.get_data()
        events = epochs.events
        right = 0
        wrong = 0
        for i, event in enumerate(events):
            last_time = event[0]
            right_answer = event[2]
            prediction  = pipe.predict(epochs_data[i, None, :, :])[0]
            right += prediction == right_answer
            wrong += prediction != right_answer

        print(f"{right = }\n{wrong = }\n{right/(right + wrong) = }")


    def predict_epochs_proba(self, pipe, epochs, threshold: float):
        epochs_data = epochs.get_data()
        events = epochs.events
        right = 0
        wrong = 0
        predicted = 0
        ignored = 0
        ignored = 0
        for i, event in enumerate(events):
            right_answer = event[2]

            prediction_distribution = pipe.predict_proba(epochs_data[i, None, :, :])
            if np.max(prediction_distribution) > threshold:
                prediction = np.argmax(prediction_distribution) + 1
                right += prediction == right_answer
                wrong += prediction != right_answer
                predicted += 1
            else:
                ignored += 1

        print(f"{right = }\n{wrong = }\n{right/(right + wrong) = }\n{predicted / (predicted + ignored) = }")

    def predict(self):
        run = tasks[self.run_id]['runs']
        labels = tasks[self.run_id]['labels']
        parser = Parser(subject=self.subject, run=run, mne_path="./mne_data", run_id=self.run_id)
        parser.motion_preprocessing(labels)
        epochs = parser.get_epochs(epochs_dir="./epochs", save=True, preload=True)
        trained_pipeline = self.get_pretrained_model("./models", self.run_id, self.subject)
        if trained_pipeline is None:
            logger.error(f"No model was trained to make prediction for run_id={self.run_id}, subject={self.subject}")
            return
        # TODO : choose with or without threshold
        self.predict_epochs_proba(trained_pipeline, epochs, 0.7)

if __name__ == "__main__":
    args = get_tpv_args()
    if args.predict:
        T = Predict(args)
    else:
        T = Train(args)

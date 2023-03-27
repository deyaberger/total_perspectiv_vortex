"""Processing EEG data pipeline."""
from tqdm import tqdm
import pandas as pd
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
        # TODO replace subjects from 1 to 109:
        self.subjects = [args.subject] if args.train or args.predict else range(1, 10)
        self.run_idxs = [args.run_idx] if args.train or args.predict else range(0, 4)
        self.runs = [x['runs'] for i, x in enumerate(tasks) if i in self.run_idxs]
        if args.predict:
            self.predict()
        else:
            self.train()

    def build_pipe(self, lda_name: str = 'classic', csp_components: int = 4) -> Pipeline:
        if (lda_name == 'shrimp'):
            lda = LDA(solver='lsqr', shrinkage='auto')
        elif lda_name == 'classic':
            lda = LDA()
        csp = CSP(n_components=csp_components)
        pipeline = Pipeline([("CSP", csp), ("LDA", lda)])
        return pipeline


    def cross_val_pipeline(self, pipeline, epochs) -> float:
        targets = epochs.events[:, -1]
        epochs_data = epochs.copy().crop(tmin=1.0, tmax=4.0).get_data() # They do this cropping on MNE website. Why ?
        scores = cross_val_score(pipeline, epochs_data, targets, cv=ShuffleSplit(10, test_size=0.2), n_jobs=None)
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
                pipe = self.build_pipe(lda_name='classic', csp_components=10)
                cross_score = self.cross_val_pipeline(pipe, epochs)
                logger.info(f"Run:{run},\t\tsubject:{subject},\t\tCross validation average:{cross_score:.2f}")
                model_metrics = {
                    'subject': subject,
                    'task_number': run_id,
                    'cross_val_score': cross_score,
                }
                metrics = pd.DataFrame(model_metrics, index=[0])
                training_data = pd.concat([training_data, metrics])
        training_data.to_csv("./results/training.csv")

    def predict(self):
        pass



if __name__ == "__main__":
    args = get_tpv_args()
    T = Treatment(args)

"""Processing EEG data pipeline."""
from tqdm import tqdm
from utils import get_tpv_args, tasks
from data_parsing import Parser


class Treatment():
    def __init__(self, args):
        self.subjects = [args.subject] if args.train or args.predict else range(0, 109)
        self.run_idxs = [args.run_idx] if args.train or args.predict else range(0, 4)
        self.runs = [x['runs'] for i, x in enumerate(tasks) if i in self.run_idxs]
        if args.predict:
            self.predict()
        else:
            self.train()

    def get_epochs(self, run, subject, labels):
        parser = Parser(subject=subject, runs=run, mne_path="./mne_data")
        parser.load_data()
        parser.select_montage('standard_1005')
        parser.get_events()
        parser.get_annotations(labels)
        noisy_channels = ['AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'Fp1', 'Fpz', 'Fp2', 'P7', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6',
                      'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Iz']
        parser.reduce_noise(60, noisy_channels)
        parser.focus_and_clean(significant_frequencies=(7.0, 32.0))
        parser.get_events()
        print(parser.events.shape)
        picks = parser.reduce_noise()
        print(len(picks))
        print(parser.raw)
        return None


    def train(self):
        for run_id in tqdm(self.run_idxs):
            run = tasks[run_id]['runs']
            labels = tasks[run_id]['labels']
            for subject in tqdm(self.subjects):
                epochs = self.get_epochs(run, subject, labels)



    def predict(self):
        pass



if __name__ == "__main__":
    args = get_tpv_args()
    T = Treatment(args)

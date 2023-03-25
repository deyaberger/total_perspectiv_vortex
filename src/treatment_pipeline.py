"""Processing EEG data pipeline."""
from utils import get_tpv_args, EXPERIMENTS
from data_parsing import Parser


class Treatment():
    def __init__(self, args):
        self.subjects = [args.subject] if args.train or args.predict else range(0, 109)
        self.run_idxs = [args.run_idx] if args.train or args.predict else range(0, 6)
        self.runs = [x["runs"] for i, x in enumerate(EXPERIMENTS) if i in self.run_idxs]
        self.descriptions = [x["description"] for i, x in enumerate(EXPERIMENTS) if i in self.run_idxs]
        if args.predict:
            self.predict()
        else:
            self.train()

    def __str__(self):
        if len(self.subjects) == len(range(0, 109)):
            return f"Treating all subject and all runs"
        return f"Subject:\t{self.subjects[0]}\nRun:\t\t{self.runs[0]}\nDescription:\t{self.descriptions[0]}"

    def train(self):
        for run in self.runs:
            for subject in self.subjects:
                parser = Parser(subject=subject, runs=run, mne_path="./mne_data")
                parser.load_data()
                montage = parser.select_montage('standard_1005')
                noisy_channels = ['AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'Fp1', 'Fpz', 'Fp2', 'P7', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6',
                                    'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Iz']
                parser.reduce_noise(60, noisy_channels)
                picks = parser.reduce_noise()
                parser.focus_and_clean(significant_frequencies=(8.0, 40.0))
                parser.get_events()


    def predict(self):
        pass



if __name__ == "__main__":
    args = get_tpv_args()
    T = Treatment(args)
    print(T)


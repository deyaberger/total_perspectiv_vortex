"""EEG Motor Experiments dataset parsing with MNE library."""
import argparse
import matplotlib.pyplot as plt
from typing import List
import logging
import os

from mne import Epochs, pick_types, events_from_annotations, annotations_from_events, read_epochs
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne import set_log_level
set_log_level("WARNING")

logger = logging.getLogger("data_parsing")
logger.setLevel(level=logging.WARNING)

class Visualizer():
    """Class to Visualize MNE data."""

    def __init__(self):
        pass

    @classmethod
    def show_montage(self, montage):
        """Plot EEG montage."""
        montage.plot()
        plt.show()

    @classmethod
    def show_raw_data(self, raw, smooth=False):
        """Plot row data (Volts of signals for each channel along the time)."""
        logger.info("Plotting raw data\n")
        if smooth:
            raw.plot(scalings=dict(eeg=1e-4), title="Raw Data")
        else:
            raw.plot(title="Raw Data")
        plt.show()

    @classmethod
    def show_psd(self, raw, average=False):
        """Plot power spectral density (PSD)."""
        if average:
            raw.compute_psd().plot(average=True, color='blue')
        else:
            raw.compute_psd().plot()
        plt.show()


class Parser():
    """Class to Parse MNE data."""

    noisy_channels = ['AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'Fp1', 'Fpz', 'Fp2', 'P7', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6',
                            'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Iz']

    def __init__(self, run_id: int = None, **kwargs):
        """Initialize data path, subject number and run task."""
        self.subject = kwargs['subject']
        self.run = kwargs['run']
        self.mne_path = kwargs['mne_path']
        self.run_id = run_id
        self.events = None
        self.annotations = None
        self.epochs = None
        self.picks = None

    def load_data(self):
        """Load data."""
        logger.info(f"Loading data for subject {self.subject}, at run {self.run}\n")
        raw_fnames = eegbci.load_data(self.subject, self.run, path=self.mne_path)
        self.raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
        eegbci.standardize(self.raw)

    def select_montage(self, montage_name):
        """Select the montage that better fits motor imagery task."""
        logger.info(f"Selecting montage: {montage_name}\n")
        montage = make_standard_montage(montage_name)
        self.raw.set_montage(montage, on_missing='ignore')
        return montage

    def compute_psd(self):
        """Compute power spectral density (PSD)."""
        self.raw.compute_psd()

    def reduce_noise(self, noisy_freq=None, noisy_channels="bads"):
        """Reducing noise to better analyse the data.

        - noisy_freq = Power line noise is a type of interference that can be picked up by EEG electrodes due to
                        the electrical activity of nearby power lines
        - noisy_channels = Channels that do are not related to the motion cortex.
        """
        logger.info("Reducing noise:")
        if noisy_freq is not None:
            logger.info(f"Removing frequency {noisy_freq}")
            self.raw.notch_filter(noisy_freq, method="iir")
        logger.info(f"Removing bad or noisy channels: {noisy_channels}\n")
        picks = pick_types(self.raw.info, eeg=True,
                           exclude=noisy_channels)
        self.raw.pick(picks)
        return picks

    def focus_and_clean(self, significant_frequencies):
        """Focus on the frequencies that are relevant for motor imagery."""
        logger.info("Focussing on significant frequencies\n")
        start, end = significant_frequencies
        self.raw.filter(start, end)

    def get_events(self, event_id=None):
        """Get events."""
        logger.info("Getting events from annotations\n")
        if event_id:
            self.events, self.event_id = events_from_annotations(self.raw, event_id=event_id)
        else:
            self.events, self.event_id = events_from_annotations(self.raw)

    def get_annotations(self, labels=None):
        """Get annotations."""
        logger.info("Getting annotations from events\n")
        if labels:
            self.annotations = annotations_from_events(events=self.events, event_desc=labels, sfreq=self.raw.info["sfreq"])
        else:
            self.annotations = annotations_from_events(events=self.events, sfreq=self.raw.info["sfreq"])

    def motion_preprocessing(self, labels: dict,
                   montage_name: str = 'standard_1005',
                   noisy_channels: List[str] = noisy_channels, noisy_freq: int = 60):
        self.load_data()
        self.select_montage(montage_name)
        self.get_events()
        self.get_annotations(labels)
        self.reduce_noise(noisy_freq, noisy_channels)
        self.picks = self.reduce_noise()
        self.focus_and_clean(significant_frequencies=(7.0, 32.0))
        self.get_events()

    def get_epochs_path(self, dir: str, run_id: int, subject_nb: int)-> str:
        self.epochs_dir = dir
        self.epochs_run_dir = f"run_{run_id}"
        self.filename = f"S{subject_nb:03d}_epo.fif"
        return f"{self.epochs_dir}/{self.epochs_run_dir}/{self.filename}"

    def create_dir_if_not_exists(self, dir: str, sub_dir: str = None):
        if not os.path.exists(dir):
            os.mkdir(dir)
        if sub_dir and not os.path.exists(f"{dir}/{sub_dir}"):
            os.mkdir(f"{dir}/{sub_dir}")

    def get_epochs(self,
                   baseline: float = None,
                   tmin: float = -1., tmax: float = 4.0,
                   epochs_dir: str = "./epochs",
                   save=False,
                   preload=False):
        epochs_path = self.get_epochs_path(epochs_dir, self.run_id, self.subject)
        if preload and epochs_path is not None and os.path.exists(epochs_path):
            logger.info(f"reading epochs existing file: {epochs_path}")
            self.epochs = read_epochs(epochs_path)
        else:
            if baseline:
                self.epochs = Epochs(self.raw, self.events, self.event_id, tmin, tmax, proj=True, picks=self.picks, baseline=(tmin, tmin + 1.), preload=True)
            else:
                self.epochs = Epochs(self.raw, self.events, self.event_id, tmin, tmax, proj=True, picks=self.picks, preload=True)
            if save and epochs_path is not None:
                logger.info(f"Saving epochs to: {epochs_path}")
                self.create_dir_if_not_exists(self.epochs_dir, self.epochs_run_dir)
                self.epochs.save(epochs_path, overwrite=True)
        return self.epochs



def get_args():
    """Get program arguments."""
    runs_dict = {
        1: [1],
        2: [2],
        3: [3, 7, 11],
        4: [4, 8, 12],
        5: [5, 9, 13],
        6: [6, 10, 14],
    }
    parser = argparse.ArgumentParser(description='Parse EEG Motor Experiment dataset')
    parser.add_argument('--show_plots', action='store_true',
                        help='Show plots at each step of the data parsing')
    parser.add_argument('--data_path', default="./mne_data",
                        help='Specify the path to MNE data')
    parser.add_argument('--subject', default=1,
                        help='Specify number of the subject you want to analyze')
    parser.add_argument('--run', default=3, choices=[1, 2, 3, 4, 5, 6],
                        help=f"Specify this list of runs you want to analyze, cf this dictionnary: {runs_dict}")
    args = parser.parse_args()
    args.run = runs_dict[args.run]
    return args


if __name__ == '__main__':
    args = get_args()
    show_plots = args.show_plots
    parser = Parser(subject=args.subject, run=args.run, mne_path=args.data_path)
    parser.load_data()
    montage = parser.select_montage('biosemi64')
    if show_plots:
        Visualizer.show_montage(montage)
        Visualizer.show_raw_data(parser.raw)
        Visualizer.show_raw_data(parser.raw, smooth=True)
        Visualizer.show_psd(parser.raw)
    if not show_plots:
        parser.compute_psd()
    noisy_channels = ['AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'Fp1', 'Fpz', 'Fp2', 'P7', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6',
                      'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Iz']
    parser.reduce_noise(60, noisy_channels)
    if show_plots:
        Visualizer.show_psd(parser.raw)
    parser.focus_and_clean(significant_frequencies=(8.0, 40.0))
    if show_plots:
        Visualizer.show_psd(parser.raw)
        Visualizer.show_psd(parser.raw, average=True)
    logger.info("Tadaaa")